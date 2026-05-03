import os
import time
import sys
import logging
from loguru import logger
from dotenv import load_dotenv

# Core LangChain libraries
from langchain_milvus import Milvus
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Lightweight

# --- Logger Setup ---
logger.remove()
logger.add(sys.stderr, level="INFO")

# --- Global Variables (Initially None) ---
# This ensures nothing runs when you just import the file
_reranker_model = None
_rag_chain = None

# --- Helper Functions ---

def _get_reranker():
    """Singleton to load reranker only when needed"""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"⏳ Loading Reranker: {RERANKER_MODEL_NAME}")
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        logger.success("✅ Reranker Loaded")
    return _reranker_model

def rerank_docs(input_data: dict) -> list[Document]:
    """Reranking Logic"""
    question = input_data["question"]
    docs = input_data["docs"]
    
    if not docs: return []
    
    reranker = _get_reranker()
    
    # Rerank logic
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = score

    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:5]] # Top 5

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join([f"Source: {d.metadata.get('url')} (Score: {d.metadata.get('rerank_score'):.4f})\n{d.page_content}" for d in docs])

# --- Main Initialization Function ---

def initialize_rag_system():
    """
    This function does the heavy lifting. 
    It loads models and connects to Milvus.
    """
    global _rag_chain
    
    if _rag_chain is not None:
        return _rag_chain

    logger.info("🔌 Connecting to RAG System...")

    # 1. Load Embeddings
    logger.info(f"⏳ Loading Embeddings: {EMBEDDING_MODEL_NAME}")
    embedding_object = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Wrapper for LangChain compatibility
    class EmbedWrapper:
        def __init__(self, model): self.model = model
        def embed_documents(self, texts): return self.model.encode(texts, normalize_embeddings=True).tolist()
        def embed_query(self, text): return self.model.encode(text, normalize_embeddings=True).tolist()
    
    wrapped_embeddings = EmbedWrapper(embedding_object)

    # 2. Connect to Milvus
    logger.info(f"⏳ Connecting to Milvus at {MILVUS_URI}...")
    try:
        vector_store = Milvus(
            embedding_function=wrapped_embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI, "token": os.getenv("MILVUS_TOKEN", None)},
            vector_field="content_vector",   
            text_field="original_content",   
            auto_id=False 
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        logger.success("✅ Milvus Connected")
    except Exception as e:
        logger.error(f"❌ Milvus Connection Failed: {e}")
        raise e

    # 3. Setup LLM & Chain
    llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    
    prompt = ChatPromptTemplate.from_template("""
    Answer based on context only.
    Context: {context}
    Question: {question}
    """)

    # Build Chain
    retrieval_branch = RunnableParallel({"docs": retriever, "question": RunnablePassthrough()})
    
    _rag_chain = (
        retrieval_branch 
        | RunnableLambda(rerank_docs) 
        | RunnableLambda(format_docs) 
        | {"context": RunnablePassthrough(), "question": RunnablePassthrough()} # Fix for prompt input
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    logger.success("🚀 RAG Chain Initialized Successfully")
    return _rag_chain

# --- Public Entry Point ---

def get_rag_response(query: str):
    """Checks if system is ready, then runs query"""
    try:
        # Lazy initialization check
        chain = initialize_rag_system()
        
        start = time.time()
        answer = chain.invoke(query)
        logger.info(f"✅ Query finished in {time.time() - start:.2f}s")
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"answer": f"System Error: {e}"}