import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load keys from the .env configuration file
load_dotenv()

def upload_vectors():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    print("--- DIAGNOSTIC CHECK ---")
    print(f"Loaded Index Name: '{index_name}'")
    print(f"Total API Key Length: {len(api_key) if api_key else 0} characters")
    print("------------------------")

    input_file = Path("data/circuit_embeddings.jsonl")
    if not input_file.exists():
        print(f"Error: {input_file} not found. Run chunk_circuits.py and embed_chunks.py first.")
        return

    # Initialize Pinecone Client
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists; if not, create it automatically!
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Creating it now on the free tier...")
        pc.create_index(
            name=index_name,
            dimension=384, # Match dimension of our local model: all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Safe free-tier region
            )
        )
        print("Waiting for index initialization...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index '{index_name}' successfully created!")

    # Connect to the targeting data index plane
    index = pc.Index(index_name)

    # Load local structures
    vectors_to_upsert = []
    print(f"Processing local embeddings from {input_file}...")
    with input_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                data = json.loads(line)
                vectors_to_upsert.append({
                    "id": data["id"],
                    "values": data["embedding"],
                    "metadata": data["metadata"]
                })

    # Batch upload to avoid server payload overflows
    batch_size = 50
    print(f"Uploading {len(vectors_to_upsert)} records in batches to index '{index_name}'...")
    
    for i in range(0, len(vectors_to_upsert), batch_size):
        chunk = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=chunk)
        print(f" Successfully pushed batch {i // batch_size + 1} ({len(chunk)} records)")

    print("\n All vectors successfully reflected inside your Pinecone database index!")

if __name__ == "__main__":
    upload_vectors()
