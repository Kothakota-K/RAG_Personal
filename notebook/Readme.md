# Semantic Chunking & Embedding Pipeline for RAG

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![Milvus](https://img.shields.io/badge/Target%20DB-Milvus-green)

## 📋 Overview

This project provides a robust data processing pipeline designed to prepare text data for **Retrieval-Augmented Generation (RAG)** systems. 

Unlike standard fixed-size chunking (which splits text arbitrarily by character count), this pipeline utilizes **Semantic Chunking**. It leverages the `BAAI/bge-base-en-v1.5` embedding model to identify distinct shifts in topic or meaning, ensuring that each resulting text chunk is a coherent, self-contained thought.

The final output is a JSON file formatted specifically for ingestion into a **Milvus** vector database.

## ✨ Key Features

* **Semantic Intelligence:** Uses `langchain-experimental` to split text based on semantic similarity rather than arbitrary character limits.
* **High-Performance Embeddings:** integrated with `sentence-transformers` to generate 768-dimension vectors using the BGE model.
* **GPU Acceleration:** Auto-detects CUDA availability and uses batch processing for rapid embedding generation.
* **Milvus Ready:** Output is structured with appropriate metadata and IDs, ready for direct insertion into a Milvus collection.

## 🛠️ Prerequisites

This script is optimized for **Google Colab** but can be adapted for local Jupyter environments.

### System Requirements
* **Runtime:** Python 3.10+
* **Hardware:** T4 GPU (or better) is highly recommended for the embedding and chunking steps.

### Dependencies
The script automatically installs the following:
* `sentence-transformers`
* `langchain`
* `langchain-experimental`
* `tqdm`

## 🚀 Usage Guide

### 1. Prepare Your Input Data
The script expects a JSON file containing an array of objects. Each object represents a document and should have the following structure:

```json
[
  {
    "content": "Full text content of the page...",
    "url": "[https://example.com/page](https://example.com/page)",
    "title": "Page Title",
    "company": "Company Name",
    "full_metadata_json": "{\"key\": \"value\"}" 
  }
]