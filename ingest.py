import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.readers.file import PyMuPDFReader

load_dotenv()

DATA_DIR = "./docs"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    query_instruction="search_query: ",
    text_instruction="search_document: ",
    trust_remote_code=True
)

def run_incremental_ingestion():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.listdir(DATA_DIR):
        print(f"Ingestion paused: Please place '.pdf' inside the '{DATA_DIR}' folder and rerun.")
        return

    # Connect to your persistent ChromaDB instance
    print("Connecting to Persistent Local ChromaDB client...")
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create the collection (NEVER delete it!)
    chroma_collection = db_client.get_or_create_collection("knowledge_base")
    
    # Extract filenames that have already been indexed
    existing_docs = set()
    try:
        # Fetch metadata arrays currently stored in the DB
        existing_data = chroma_collection.get(include=["metadatas"])
        if existing_data and existing_data["metadatas"]:
            for meta in existing_data["metadatas"]:
                if meta and "file_name" in meta:
                    existing_docs.add(meta["file_name"])
    except Exception as e:
        print(f"Note: Could not parse existing collection metadata: {str(e)}")

    if existing_docs:
        print(f"Found {len(existing_docs)} unique files already indexed in your database.")
    else:
        print("Database appears to be empty. Starting fresh baseline ingestion.")

    # Scan the docs directory, ignoring already-indexed files
    file_extractor = {".pdf": PyMuPDFReader()}
    
    def file_metadata_helper(file_path):
        """Helper to assign basic file tracking tags."""
        return {"file_name": os.path.basename(file_path)}

    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        recursive=True,
        file_extractor=file_extractor,
        file_metadata=file_metadata_helper
    )
    
    # Filter the files to load before hitting the heavy layout extractor
    all_files = reader.input_files
    new_files = [f for f in all_files if os.path.basename(f) not in existing_docs]

    if not new_files:
        print("\n🎉 Everything is up to date! No new files detected in the './docs' folder.")
        return

    print(f"\nDetected {len(new_files)} new files out of {len(all_files)} total items.")
    for f in new_files:
        print(f" -> Queueing: {os.path.basename(f)}")

    # Update reader to process ONLY the new additions
    reader.input_files = new_files
    documents = reader.load_data()
    
    print(f"Successfully extracted {len(documents)} new pages from the queue.")

    # Prepare storage contexts
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Chunk the text blocks and insert them into the index
    print("Chunking new data into structural node components...")
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    print(f"Writing {len(nodes)} new vector nodes directly to ChromaDB...")
    
    # Loading from vector store automatically appends nodes instead of wiping them
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    index.insert_nodes(nodes)

    print(f"\nSuccess! New documentation has been merged into ChromaDB at '{CHROMA_DB_PATH}'.")

if __name__ == "__main__":
    run_incremental_ingestion()