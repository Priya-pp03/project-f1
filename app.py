import os
import json
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load configurations from the .env file
load_dotenv()

def get_env_var(var_name):
    val = os.getenv(var_name)
    if not val and var_name == "PINECONE_INDEX":
        val = os.getenv("PINECONE_INDEX_NAME")
    return val

# Cache models to load them into system memory only ONCE
@st.cache_resource
def load_models():
    pinecone_key = get_env_var("PINECONE_API_KEY")
    index_name = get_env_var("PINECONE_INDEX")
    
    if not pinecone_key or not index_name:
        st.error("❌ Configuration Error: Missing credentials inside your .env file.")
        return None, None, None

    # Connect to Pinecone Cloud data planes
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    # Load embedding model locally
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load 100% free text generator model locally
    local_generator = pipeline(
        "text-generation", 
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        device_map="auto"
    )
    
    return index, encoder, local_generator

# Page Layout Configuration Setup
st.set_page_config(page_title="F1 RAG Assistant", page_icon="🏎️", layout="centered")
st.title("🏎️ F1 Circuit RAG Chatbot")
st.caption("Powered completely free via local vector embeddings and local LLM execution pipelines.")

index, encoder, local_generator = load_models()

# Initialize Chat Message History Layout Array if not present in active session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Ask me anything about the Formula 1 circuits dataset. 🏁"}
    ]

# Display older past chat messages dynamically inside the layout frame
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input Box Prompt Layout mapping
if user_input := st.chat_input("Ask a question (e.g., 'Tracks in Australia'):"):
    
    # Add user message to display array state tracking
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response section
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Searching database context and computing local inference response..."):
            # A. Convert user text query locally to embedding vector maps
            query_vector = encoder.encode(user_input, normalize_embeddings=True).tolist()
            
            # B. Query Pinecone database cluster for nearby vector points
            res = index.query(vector=query_vector, top_k=2, include_metadata=True)
            
            context_blocks = []
            for match in res.get("matches", []):
                if "metadata" in match and "text" in match["metadata"]:
                    context_blocks.append(match["metadata"]["text"])

            context_string = "\n---\n".join(context_blocks) if context_blocks else "No matching circuit context found."

            # C. Frame system structural instructions template layout for LLM
            prompt = (
                "<|system|>\n"
                "You are an expert F1 Assistant. Answer the user question short and concisely using ONLY this context:\n"
                f"{context_string}\n"
                f"<|user|>\n{user_input}\n"
                "<|assistant|>\n"
            )

            # D. Fire local inference processing engine pipeline execution loops
            outputs = local_generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
            generated_text = outputs[0]["generated_text"]
            
            # Extract out system generated text content blocks cleanly splitting out tags
            answer = generated_text.split("<|assistant|>\n")[-1].strip()
            
            # Render answer directly into the active layout framework block
            response_placeholder.markdown(answer)
            
    # Record answer back into session log history arrays
    st.session_state.messages.append({"role": "assistant", "content": answer})
