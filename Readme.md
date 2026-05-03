# 🧠 RAG Backend Service

![LangChain](https://img.shields.io/badge/Orchestration-LangChain-blue)
![Milvus](https://img.shields.io/badge/Vector%20DB-Milvus-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20(Llama3)-green)

## 📖 Overview

The `backend_service.py` acts as the core reasoning engine for the AMD AI Assistant. It implements a **Retrieval-Augmented Generation (RAG)** pipeline that:
1.  **Embeds** user queries using the BAAI/bge-base model.
2.  **Retrieves** the top 10 most semantically relevant documents from a Milvus Vector Database.
3.  **Synthesizes** an answer using the Llama 3 70B model (hosted on Groq), grounded strictly in the retrieved context.

## ⚙️ Architecture

### 1. Embedding Layer
* **Model:** `BAAI/bge-base-en-v1.5`
* **Wrapper:** Custom `BGEEmbeddings` class to adapt SentenceTransformers for LangChain compatibility.
* **Normalization:** Enabled (`normalize_embeddings=True`) to optimize for Cosine Similarity search.

### 2. Retrieval Strategy
* **Type:** Dense Semantic Search.
* **Limit (K):** 10 documents per query.
* **Fields:** matches against `content_vector`, retrieves `original_content` and `url` metadata.

### 3. Generation Layer
* **Provider:** Groq API.
* **Model:** `llama-3.3-70b-versatile`.
* **Temperature:** `0.0` (Deterministic/Factual).
* **Prompting:** Uses Llama 3 specific special tokens (`<|start_header_id|>`) to enforce strict instruction following.

## 🚀 Setup & Execution

### Prerequisites
* Python 3.10+
* A running Milvus instance (URI required)
* A Groq API Key

### Environment Variables
Create a `.env` file in the root directory:

```ini
GROQ_API_KEY=gsk_your_groq_key_here
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION_NAME=amd_knowledge_base
MILVUS_TOKEN=root:Milvus (Optional)