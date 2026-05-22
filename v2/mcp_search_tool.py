import os
import sys
import warnings

# Suppress warnings early
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

load_dotenv()

# Set up local cache directories cleanly
PROJECT_ROOT = os.getenv("PROJECT_ROOT", r"C:\Users\arnis.zelcs\CODE\ECDIS-1.3.0\Tools\ai\chroma_search_mcp")
os.environ["XDG_CACHE_HOME"] = os.path.join(PROJECT_ROOT, "cache")

# ONLY import light core MCP definitions at the top level
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Chroma Search Tool", port=8001)

# Keep global states empty at launch
GLOBAL_INDEX = None
DB_CLIENT = None


def get_kv_index():
    global GLOBAL_INDEX, DB_CLIENT
    if GLOBAL_INDEX is not None:
        return GLOBAL_INDEX

    # LAZY IMPORTS: Frameworks are only imported when a tool runs.
    # This prevents the server startup orchestration from deadlocking.
    print("Initializing embedding models and database components lazily...", file=sys.stderr)
    import chromadb
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", r"C:\Users\arnis.zelcs\CODE\ECDIS-1.3.0\Tools\ai\chroma_search_mcp\chroma_db")

    Settings.llm = None

    # Matches the exact physical baseline embedding profile used in ingestion
    Settings.embed_model = OllamaEmbedding(
        model_name="mxbai-embed-large",
        embed_batch_size=1  # Sequential fallback for absolute server processing stability
    )

    DB_CLIENT = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True),
    )

    chroma_collection = DB_CLIENT.get_collection("knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    GLOBAL_INDEX = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return GLOBAL_INDEX


@mcp.tool()
def search(query: str) -> str:
    """Queries the local persistent vector database and returns matching raw document text blocks."""
    try:
        # DB and Embedding logic only triggers here
        index = get_kv_index()
        
        # DRAWBACK FIX Optimization: Since switched to granular 384 chunks, 
        # raising similarity_top_k to 5 provides a wider context window for LLM.
        retriever = index.as_retriever(similarity_top_k=5)
        retrieved_nodes = retriever.retrieve(query)

        if not retrieved_nodes:
            return f"No relevant documentation found for: '{query}'"

        output_buffer = []
        for i, node in enumerate(retrieved_nodes, 1):
            file_name = node.metadata.get("file_name", "Unknown File")
            page_label = node.metadata.get("page_label", "Unknown Page")
            score = round(node.score, 4) if getattr(node, "score", None) is not None else "N/A"

            block = (
                f"--- REFERENCE BLOCK {i} (Score: {score}) ---\n"
                f"SOURCE FILE: {file_name}\n"
                f"PAGE LOCATION: {page_label}\n"
                f"CONTENT:\n{node.text.strip()}\n"
                f"----------------------------\n"
            )
            output_buffer.append(block)

        return "\n".join(output_buffer)

    except Exception as e:
        return f"Error executing database vector search lookup: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="sse")