import os
import logging
from loguru import logger
from dotenv import load_dotenv

# Core LangChain libraries
from langchain_milvus import Milvus
from langchain_core.documents import Document

# Direct embedding model usage
from sentence_transformers import SentenceTransformer 

# --- Configuration ---
load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" 

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger.add("rag_backend.log", rotation="10 MB", retention="10 days", level="DEBUG")
logger.info("Retrieval Backend Service Initialized.")

# --- RAG Components ---

# 1. Define a Wrapper Class for the Embeddings
class BGEEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Normalize embeddings is CRITICAL for Inner Product (IP) to work like Cosine Similarity
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

try:
    embedding_object = BGEEmbeddings(EMBEDDING_MODEL_NAME)
    logger.success("Embedding model loaded and wrapped successfully.")
except Exception as e:
    logger.error(f"Error loading embedding model: {e}")
    raise

# 2. Milvus Vector Store and Retriever
try:
    # --- CHANGED: Added Index and Search Parameters for IP ---
    vector_store = Milvus(
        embedding_function=embedding_object,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI, "token": os.getenv("MILVUS_TOKEN", None)},
        vector_field="content_vector",   
        text_field="original_content",   
        auto_id=False,
        # Defines how the index should be built (if creating a new collection)
        index_params={
            "metric_type": "IP",        # <--- FORCE INNER PRODUCT
            "index_type": "HNSW",       # Hierarchical Navigable Small World (Fast/Accurate)
            "params": {"M": 8, "efConstruction": 64}
        },
        # Defines default search behavior
        search_params={
            "metric_type": "IP",        # <--- ENSURE SEARCH USES IP
            "params": {"ef": 64}
        }
    )
    
    # --- CHANGED: Explicitly pass metric_type to retriever ---
    milvus_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 10,
            "param": {"metric_type": "IP", "params": {"ef": 64}} 
        }
    )
    logger.success(f"Milvus connection to {COLLECTION_NAME} successful (Metric: IP).")
except Exception as e:
    logger.error(f"Milvus connection failed at URI {MILVUS_URI}: {e}")
    raise

# --- Retrieval Function ---

def retrieve_chunks(query: str):
    """
    Embeds the query, searches Milvus using Inner Product, and returns the raw chunks.
    """
    logger.info(f"Processing IP retrieval for query: {query}")
    
    try:
        # Direct retrieval using LangChain's interface
        docs = milvus_retriever.invoke(query)
        
        results = []
        for i, doc in enumerate(docs):
            # Extract content and metadata
            content = doc.page_content
            url = doc.metadata.get("url", "No URL Provided")
            
            # Note: LangChain sometimes hides the raw score in standard output, 
            # but IP search is active based on the retriever config.
            
            results.append({
                "chunk_id": i + 1,
                "content": content,
                "source_url": url
            })
            
        logger.info(f"Successfully retrieved {len(results)} chunks.")
        return results

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []

if __name__ == '__main__':
    # Simple test run
    test_query = "What is the collaboration between AMD and HPE?"
    logger.info(f"--- Running Test Query: {test_query} ---")
    
    chunks = retrieve_chunks(test_query)
    
    print("\n--- RETRIEVED RESULTS ---")
    for chunk in chunks:
        print(f"\n[Source: {chunk['source_url']}]")
        print(f"{chunk['content'][:200]}...")