# backend_service.py
import os
import logging
from loguru import logger
from dotenv import load_dotenv

# Core LangChain libraries
from langchain_milvus import Milvus
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Direct embedding model usage
from sentence_transformers import SentenceTransformer 

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" 

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger.add("rag_backend.log", rotation="10 MB", retention="10 days", level="DEBUG")
logger.info("RAG Backend Service Initialized.")

# --- RAG Components ---

# 1. Define a Wrapper Class for the Embeddings
# This creates a class that LangChain accepts (must have embed_documents and embed_query methods)
class BGEEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Normalize embeddings is important for cosine similarity / IP
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

try:
    # Initialize the wrapper class instance
    embedding_object = BGEEmbeddings(EMBEDDING_MODEL_NAME)
    logger.success("Embedding model loaded and wrapped successfully.")
except Exception as e:
    logger.error(f"Error loading embedding model: {e}")
    raise

# 2. Milvus Vector Store and Retriever
try:
    vector_store = Milvus(
        embedding_function=embedding_object, # Pass the CLASS INSTANCE
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI, "token": os.getenv("MILVUS_TOKEN", None)},
        # Map your custom field names from your schema
        vector_field="content_vector",   
        text_field="original_content",   
        auto_id=False # You managed IDs manually in the upload script
    )
    
    milvus_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 10, # Retrieve top 4 most relevant chunks
            # NOTE: We DO NOT pass 'output_fields' here. 
            # LangChain automatically retrieves 'text_field' and all scalar metadata fields.
        }
    )
    logger.success(f"Milvus connection to {COLLECTION_NAME} successful.")
except Exception as e:
    logger.error(f"Milvus connection failed at URI {MILVUS_URI}: {e}")
    raise

# 3. Groq LLM (Llama 3 70B)
llm = ChatGroq(
    temperature=0.0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)
logger.info(f"Groq LLM Llama 70B loaded.")

# 4. Prompt Template
RAG_PROMPT_TEMPLATE = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an expert AMD AI assistant. Provide accurate answers based *only* on the provided context.
If the answer cannot be found, state: "I am unable to find a relevant answer in the available knowledge base."
When citing, list the reference URLs found in the source documents at the end of your answer.

<|start_header_id|>context<|end_header_id|>
{context}

<|start_header_id|>user<|end_header_id|>
Question: {question}

<|end_of_text|>
"""
RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# --- Helper Functions for the Chain ---

def format_docs(docs: list[Document]) -> str:
    """Formats retrieved chunks into a single string for the LLM context."""
    formatted_context = []
    logger.debug(f"Retrieved {len(docs)} chunks for formatting.")
    
    for i, doc in enumerate(docs):
        # LangChain automatically puts scalar fields like 'url' into doc.metadata
        url = doc.metadata.get("url", "No URL Provided")
        
        logger.debug(f"Chunk {i+1} Source: {url}")
        formatted_context.append(f"--- Document Source: {url} ---\n{doc.page_content}\n---")

    return "\n\n".join(formatted_context)

def get_sources(docs: list[Document]) -> list[str]:
    """Extracts unique reference URLs from the metadata."""
    sources = set(doc.metadata.get("url") for doc in docs if doc.metadata.get("url"))
    return list(sources)

# --- The Main RAG Chain ---

# Chain to generate the answer
rag_chain = (
    {"context": milvus_retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# Separate chain to just retrieve sources (for UI display)
source_chain = (
    {"docs": milvus_retriever, "question": RunnablePassthrough()}
    | RunnableLambda(lambda x: get_sources(x['docs']))
)

# The Main Callable Function
def get_rag_response(query: str):
    """Executes the RAG pipeline."""
    logger.info(f"Processing user query: {query}")
    
    # 1. Get the Answer
    try:
        answer = rag_chain.invoke(query)
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        return {"answer": f"Error generating response: {e}", "sources": []}

    # 2. Get the Sources
    try:
        sources = source_chain.invoke(query)
    except Exception as e:
        logger.warning(f"Error extracting sources: {e}")
        sources = []
    
    logger.info(f"Answer generated. Found {len(sources)} sources.")
    return {"answer": answer, "sources": sources}

if __name__ == '__main__':
    # Simple test run when executing this file directly
    test_query = "What is the collaboration between AMD and HPE?"
    logger.info(f"--- Running Test Query: {test_query} ---")
    
    response = get_rag_response(test_query)
    
    print("\n--- TEST RESULT ---")
    print(f"Answer: {response['answer']}")
    print(f"References: {response['sources']}")