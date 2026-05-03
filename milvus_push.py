# milvus_batch_insert_truncated.py
 
import os
import json
from dotenv import load_dotenv
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
 
# --- 1. CONFIGURATION ---
load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI", "http://192.168.1.124:19530")
COLLECTION_NAME = "web_scraped_semantic_bge_robust_v5"
JSON_FILE_PATH = "milvus_ready_semantic_bge_base_data.json"
EMBEDDING_DIM = 768
BATCH_SIZE = 500
 
def process_milvus_data():
    print("--- Starting Milvus Batch Insertion (With Truncation) ---")
 
    # --- 2. CONNECT ---
    try:
        uri_clean = MILVUS_URI.replace("http://", "").replace("https://", "")
        host, port = uri_clean.split(":")
        connections.connect(alias="default", host=host, port=port)
        print(f"✅ Connected to {host}:{port}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return
 
    # --- 3. DEFINE SCHEMA ---
    id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False)
    vector_field = FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    content_field = FieldSchema(name="original_content", dtype=DataType.VARCHAR, max_length=65535)
    url_field = FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512)
    company_field = FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=128)
    title_field = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512)
    word_count_field = FieldSchema(name="word_count", dtype=DataType.INT64)
    metadata_field = FieldSchema(name="full_metadata_json", dtype=DataType.VARCHAR, max_length=16384)
 
    schema = CollectionSchema(
        fields=[id_field, vector_field, content_field, url_field, company_field, title_field, word_count_field, metadata_field],
        description="Semantic chunks for RAG"
    )
 
    # --- 4. CREATE COLLECTION ---
    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️ Collection '{COLLECTION_NAME}' exists. Dropping for clean run...")
        utility.drop_collection(COLLECTION_NAME)
 
    print(f"🚀 Creating Collection: {COLLECTION_NAME}")
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Bounded")
 
    # --- 5. CREATE INDEX ---
    print("⚙️ Creating Index...")
    index_params = {"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
    collection.create_index(field_name="content_vector", index_params=index_params)
 
    # --- 6. LOAD DATA ---
    print(f"📂 Loading JSON file...")
    try:
        with open(JSON_FILE_PATH, 'r') as f:
            data_to_insert = json.load(f)
        total_entities = len(data_to_insert)
        print(f"✅ Loaded {total_entities} entities.")
    except Exception as e:
        print(f"❌ Failed to load JSON. Error: {e}")
        return
 
    # --- 7. BATCH INSERTION WITH SAFETY CHECKS ---
    print(f"🚀 Starting Batch Insertion...")
    total_inserted = 0
    for i in range(0, total_entities, BATCH_SIZE):
        batch = data_to_insert[i : i + BATCH_SIZE]
        # --- CRITICAL FIX: Truncate oversized fields ---
        for entity in batch:
            # Truncate content to 60,000 characters to be safe (Limit is ~65,535 bytes)
            content = entity.get("original_content", "")
            if len(content) > 60000:
                print(f"✂️ Truncating oversized chunk {entity.get('id')} (Length: {len(content)})")
                entity["original_content"] = content[:60000]
            # Truncate metadata if needed
            meta = entity.get("full_metadata_json", "")
            if len(meta) > 16000:
                entity["full_metadata_json"] = meta[:16000]
 
            # Truncate URL if needed (rare, but possible)
            url = entity.get("url", "")
            if len(url) > 256:
                entity["url"] = url[:256]
 
        try:
            collection.insert(batch)
            total_inserted += len(batch)
            print(f"   ⏳ Inserted {total_inserted}/{total_entities} entities...", end="\r")
        except Exception as e:
            print(f"\n❌ Error inserting batch starting at index {i}: {e}")
            return
 
    print(f"\n🎉 SUCCESS! All {total_inserted} entities inserted.")
    # Finalize
    collection.flush()
    collection.load()
    print("✅ Data Loaded & Indexed.")
 
if __name__ == "__main__":
    process_milvus_data()