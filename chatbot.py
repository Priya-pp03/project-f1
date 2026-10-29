import os
import json
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load configurations for the Pinecone retrieval layer
load_dotenv()

def get_env_var(var_name):
    val = os.getenv(var_name)
    if not val and var_name == "PINECONE_INDEX":
        val = os.getenv("PINECONE_INDEX_NAME")
    return val

def run_chatbot():
    pinecone_key = get_env_var("PINECONE_API_KEY")
    index_name = get_env_var("PINECONE_INDEX")

    if not pinecone_key or not index_name:
        print("\n❌ Configuration Error: Missing PINECONE_API_KEY or PINECONE_INDEX in .env")
        return 1

    print("🤖 Booting Local Free F1 Racing Assistant Engine...")
    print("Connecting to Pinecone Database plane...")
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    print("Initializing embedding transformation layer ('all-MiniLM-L6-v2')...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading free local text generator model ('TinyLlama-1.1B') into memory...")
    # This downloads a highly compact model that runs smoothly and instantly on your CPU/GPU for free
    local_generator = pipeline(
        "text-generation", 
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        device_map="auto"
    )

    print("\n🏎️ Local F1 Circuit RAG Chatbot is online and 100% FREE!")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower().strip() in ['exit', 'quit']:
            print("Shutting down assistant. Goodbye!")
            break
            
        if not user_input.strip():
            continue

        print("Searching database context...")
        # A. Vectorize input text locally
        query_vector = encoder.encode(user_input, normalize_embeddings=True).tolist()
        
        # B. Retrieve details from Pinecone cloud setup
        res = index.query(vector=query_vector, top_k=2, include_metadata=True)
        
        context_blocks = []
        for match in res.get("matches", []):
            if "metadata" in match and "text" in match["metadata"]:
                context_blocks.append(match["metadata"]["text"])

        context_string = "\n---\n".join(context_blocks) if context_blocks else "No matching circuit context found."

        # C. Construct the prompt for the local text model
        prompt = (
            "<|system|>\n"
            "You are an expert F1 Assistant. Answer the user question short and concisely using ONLY this context:\n"
            f"{context_string}\n"
            f"<|user|>\n{user_input}\n"
            "<|assistant|>\n"
        )

        print("Generating answer locally...")
        try:
            outputs = local_generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
            generated_text = outputs[0]["generated_text"]
            
            # Extract just the answer portion following the assistant tag
            answer = generated_text.split("<|assistant|>\n")[-1].strip()
            print(f"\nAssistant: {answer}")
        except Exception as e:
            print(f"\n❌ Local Generation Error: {str(e)}")

if __name__ == "__main__":
    sys.exit(run_chatbot())
