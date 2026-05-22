import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    query_instruction="search_query: ",
    text_instruction="search_document: ",
    trust_remote_code=True
)

# Explicitly disable the LLM so LlamaIndex doesn't look for one
Settings.llm = None 

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

def raw_database_lookup():
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"ChromaDB folder missing at '{CHROMA_DB_PATH}'. Run ingestion script first.")

    print("Connecting to local ChromaDB engine...")
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db_client.get_collection("knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Create a pure retriever (fetches the top 3 closest text chunks)
    retriever = index.as_retriever(similarity_top_k=3)
    
    print("\n ⚡ Pure Semantic Search Engine Ready!")
    print("-" * 50)
    
    while True:
        query = input("\n Search your documents: ")
        if query.strip().lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue
            
        print("🔍 Scanning vector space...")
        
        # Retrieve raw nodes (text + metadata) directly from ChromaDB
        retrieved_nodes = retriever.retrieve(query)
        
        for i, node in enumerate(retrieved_nodes, 1):
            file_name = node.metadata.get('file_name', 'Unknown File')
            page_label = node.metadata.get('page_label', 'Unknown Page')
            
            # The score represents how closely the text matches your query conceptually
            score = round(node.score, 4) if node.score else "N/A"
            
            print(f"\n[{i}] 📄 {file_name} | Page: {page_label} | Relevance Score: {score}")
            print("-" * 60)
            print(node.text.strip())
            print("-" * 60)

if __name__ == "__main__":
    raw_database_lookup()