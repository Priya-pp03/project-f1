import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    input_file = Path("data/chunks.jsonl")
    output_file = Path("data/circuit_embeddings.jsonl")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found. Please run chunk_circuits.py first.")
        return 1

    # 1. Load the generated text chunks
    chunks = []
    print(f"Reading chunks from {input_file}...")
    with input_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
                
    texts = [chunk["text"] for chunk in chunks]
    
    # 2. Load the free, local embedding model
    print("Loading embedding model ('all-MiniLM-L6-v2') locally...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Compute vector representations
    print(f"Generating local embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # 4. Save results back into an matching JSONL layout
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing finalized embeddings to {output_file}...")
    with output_file.open("w", encoding="utf-8") as fh:
        for i, chunk in enumerate(chunks):
            # Append the array of numerical vectors directly to the structure
            chunk["embedding"] = embeddings[i].tolist()
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print("Process complete! Everything embedded without consuming API credits.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
