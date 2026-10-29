import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load environment configuration keys
load_dotenv()

def search_f1_index(query_text, top_k=3):
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key:
        print("Error: Missing PINECONE_API_KEY inside your .env configuration.")
        return

    # 1. Initialize Pinecone connection
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # 2. Local conversion of search text to embedding vector match points
    print(f"Vectorizing query: '{query_text}'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

    # 3. Query Cloud Server
    print("Fetching top semantic matches from Pinecone index...")
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # 4. Display Formatted Track Results
    print(f"\n🔍 Cloud Search Results for: '{query_text}'")
    print("=" * 60)
    for idx, match in enumerate(response["matches"], 1):
        meta = match["metadata"]
        print(f"{idx}. {meta.get('name')} (Score: {match['score']:.4f})")
        print(f"   📍 Location: {meta.get('location')}, {meta.get('country')}")
        if meta.get('alt'):
            print(f"   ⛰️ Altitude: {meta.get('alt')} meters")
        if meta.get('url'):
            print(f"   🌐 Wiki: {meta.get('url')}")
        print("-" * 60)

if __name__ == "__main__":
    search_string = input("What F1 circuit features are you looking for? ")
    search_f1_index(search_string)
