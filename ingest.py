import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

# Configure Local Embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Workspace Directories
DATA_DIR = "./docs"
CHROMA_DB_PATH = "./chroma_db"

def run_stable_ingestion():
    # Ensure workspace folder exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.listdir(DATA_DIR):
        print(f"Ingestion paused: Please place '.pdf' inside the '{DATA_DIR}' folder and rerun.")
        return

    print(f"Scanning '{DATA_DIR}' via robust fallback parser (pypdf)...")
    
    # By passing an empty dictionary to file_extractor, SimpleDirectoryReader 
    # automatically falls back to using pypdf for all .pdf files.
    # This completely bypasses the min() empty layout bugs.
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        recursive=True,
        file_extractor={} 
    )
    
    documents = reader.load_data()
    print(f"Successfully extracted {len(documents)} raw pages from the directory.")

    print("Initializing Persistent Local ChromaDB client...")
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create a specific collection for your maritime specification data
    chroma_collection = db_client.get_or_create_collection("knowledge_base")

    # Set up the Chroma vector store layer inside LlamaIndex
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking data and generating vector embeddings...")
    # Clean text splitting configuration for dense documentation text blocks
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[node_parser]
    )

    print(f"\nSuccess! Documentation is fully indexed inside ChromaDB at '{CHROMA_DB_PATH}'.")

if __name__ == "__main__":
    run_stable_ingestion()