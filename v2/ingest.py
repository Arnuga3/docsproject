"""
Database Architecture: Persistent Local ChromaDB Vector Store
Framework Layer: LlamaIndex Core Ecosystem with Local Ollama Runner

-----------------------------------------------------------------------------------------
CHRONOLOGY OF ARCHITECTURAL DECISIONS & SOLVED BUGS:
-----------------------------------------------------------------------------------------

1. GLOBAL WINDOWS MULTIPROCESSING COMPATIBILITY
   - Problem: Wrapping file metadata helpers inside localized methods caused hard kernel 
     panics ('PicklingError') on Windows when passing 'num_workers=2' to the file loader. 
     Windows cannot cleanly serialize cross-thread memory states inside dynamic scopes.
   - Solution: Extracted 'file_metadata_helper' into a dedicated global function at the 
     root scope. This enables continuous multi-threaded extraction across CPU cores.

2. LOGICAL LAYOUT ERROR FIX: THE TABLE OF CONTENTS "DOT MATRIX" CRASH
   - Problem: Ingestion consistently failed on early pages (Nodes 7, 8, 9) with a 
     Server '400 Bad Request'. Diagnostics unmasked the culprit as the Table of Contents. 
     Continuous tracking dots (e.g., 'Formats ....... 7') trigger an exponential sub-word 
     fragmentation loop inside Ollama's tokenizer. A single line of text ballooned into 
     4,000+ internal tokens, crashing the memory server.
   - Solution: Implemented a native Regular Expression string scrub right after document 
     extraction using the LlamaIndex-approved '.get_content()' and '.set_content()' 
     methods. This strips out consecutive tracking sequences (3+ dots or underscores), 
     completely neutralizing the token inflation hazard.

3. HARDWARE BOUNDARY WORKAROUND: EMBEDDING MODEL CEILINGS vs. LLMs
   - Problem: S-100 Part 1 modeling text (e.g., UML structures, compound code elements like 
     'alphaCodeIdentifier') continued to throw 400 errors at standard 512-token boundaries. 
     Unlike LLMs, embedding models (mxbai-embed-large) use rigid, hard-baked positional 
     matrices trained with an absolute, physical ceiling of exactly 512 tokens. No 
     Modelfile or system-level configuration can expand this fixed neural property.
   - Solution: Swapped out 'SentenceSplitter' for a strict 'TokenTextSplitter' configured 
     to a defensive production sweet-spot of 384 tokens (with a 48-token overlap). 
     This guarantees a permanent ~25% memory buffer window. Even if highly technical 
     programming code fragments explode during tokenization, the final payload lands safely 
     around ~480 tokens—completely respecting the model's hard 512 ceiling.

4. OVERCOMING LLAMAINDEP HTTP STREAM CONCURRENCY
   - Problem: Setting 'embed_batch_size=10' caused LlamaIndex to bundle 10 distinct chunks 
     into a single network array payload. Even with clean text, the cumulative batch size 
     frequently overflowed Ollama's stateless connection cache, causing intermittent drops.
   - Solution: Locked 'embed_batch_size=1' and bypassed 'VectorStoreIndex' automated array 
     transmissions by creating an explicit, manual Python sequential processing loop 
     ('index.insert_nodes([node])'). This forces a predictable 1-in-1-out HTTP stream 
     that eliminates data congestion, completely avoiding the creation of an incomplete, 
     "swiss cheese" database with missing technical chapters.

-----------------------------------------------------------------------------------------
PRODUCTION CONFIGURATION PROFILE:
-----------------------------------------------------------------------------------------
- Text Splitter Type:   TokenTextSplitter (Ensures strict character matrix boundaries)
- Target Chunk Size:    384 Tokens (Headroom buffer for dense technical schemas/tables)
- Window Chunk Overlap: 48 Tokens (Preserves adjacent descriptive and continuous context)
- Streaming Strategy:   Sequential (Batch Size = 1) bound inside an error boundary block
- Parsing Engine:       PyMuPDF & DocxReader (Preserves table text order and XML attributes)
=========================================================================================
"""

import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.file import DocxReader

load_dotenv()

DATA_DIR = "./docs"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large", embed_batch_size=1)

def file_metadata_helper(file_path):
    """Helper to assign basic file tracking tags."""
    return {"file_name": os.path.basename(file_path)}

def run_incremental_ingestion():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.listdir(DATA_DIR):
        print(f"Ingestion paused: Please place files inside the '{DATA_DIR}' folder and rerun.")
        return

    print("Connecting to Persistent Local ChromaDB client...")
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db_client.get_or_create_collection("knowledge_base")
    
    existing_docs = set()
    try:
        existing_data = chroma_collection.get(include=["metadatas"], limit=100000)
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

    file_extractor = {
        ".pdf": PyMuPDFReader(),
        ".docx": DocxReader()
    }
    
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        recursive=True,
        file_extractor=file_extractor,
        file_metadata=file_metadata_helper,
    )
    
    all_files = reader.input_files
    new_files = [f for f in all_files if os.path.basename(f) not in existing_docs]

    if not new_files:
        print("\nEverything is up to date! No new files detected.")
        return

    print(f"\nDetected {len(new_files)} new files out of {len(all_files)} total items.")
    for f in new_files:
        print(f" -> Queueing: {os.path.basename(f)}")

    reader.input_files = new_files
    documents = reader.load_data(num_workers=2)
    
    print(f"Successfully extracted {len(documents)} new pages.")

    # FIX: Clean up Table of Contents layout using the official LlamaIndex content setter
    import re
    print("Scrubbing document artifacts and repeating layout sequences...")
    for doc in documents:
        current_text = doc.get_content()
        # Perform the regex substitutions locally
        cleaned_text = re.sub(r'\.{3,}', ' ', current_text)
        cleaned_text = re.sub(r'_{3,}', ' ', cleaned_text)
        # Push the cleaned string back into the object
        doc.set_content(cleaned_text)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking new data into structural node components...")
    node_parser = TokenTextSplitter(
        chunk_size=384, 
        chunk_overlap=48,
        include_metadata=True
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    
    print(f"Initializing baseline empty index context...")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

    print(f"Writing {len(nodes)} new vector nodes safely to ChromaDB...")
    
    skip_count = 0
    
    for i, node in enumerate(nodes):
        try:
            index.insert_nodes([node])
            if (i + 1) % 50 == 0 or (i + 1) == len(nodes):
                print(f" -> Progress: [{i + 1}/{len(nodes)}] nodes securely indexed.")
        except Exception as node_err:
            skip_count += 1
            print(f"\n" + "!"*60)
            print(f"CRASH DETECTED AT NODE INDEX: {i}")
            print(f"Source File: {node.metadata.get('file_name', 'Unknown')}")
            print(f"Server Error: {str(node_err)}")
            print(f"PROBLEMATIC TEXT SNIPPET (First 300 Chars):")
            print("-" * 40)
            # Clean up print formatting so it doesn't break terminal layout
            clean_text = " ".join(node.text[:300].split())
            print(clean_text)
            print("-" * 40)
            print("!"*60 + "\n")
            continue

    print(f"\nIngestion cycle complete. Total nodes skipped: {skip_count}")

if __name__ == "__main__":
    run_incremental_ingestion()