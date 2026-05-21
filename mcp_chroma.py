# mcp_chroma.py
import sys
import os
from fastmcp import FastMCP
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize the official FastMCP Server Container
mcp = FastMCP("Knowledge Base")

# Ensure global settings match your development stack
LLM_KEY = os.getenv("API_KEY")
Settings.llm = OpenAILike(
    model="deepseek/deepseek-v4-flash:free",
    api_key=LLM_KEY,
    api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
    is_chat_model=True
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

# Expose ChromaDB search layer as an explicit tool for Cline
@mcp.tool()
def search(query: str) -> str:
    """
    Queries the persistent vector database.
    Always returns exact technical facts along with source file names and page citations.
    """
    try:
        # Enforce absolute path tracking so Cline can execute it from any root
        db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = db_client.get_collection("knowledge_base")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        retriever = index.as_retriever(similarity_top_k=5)
        
        system_template = (
            "You are an expert Software Engineering Assistant.\n"
            "Your responses must be derived EXCLUSIVELY from the provided database context chunks below.\n"
            "If the context does not contain the specific answer, say: 'I cannot find that information in the ingested documentation.'\n\n"
            "CRITICAL CITATION RULES:\n"
            "Every chunk below contains metadata headers. You MUST extract the 'file_name' and 'page_label' "
            "and append an explicit citation to every single statement. Example: [Source: file_name.pdf, Page X].\n\n"
            "Database Context:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
        )
        
        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            llm=Settings.llm,
            system_prompt=system_template
        )
        
        response = chat_engine.chat(query)
        return str(response).strip()

    except Exception as e:
        return f"Error executing database vector search lookup: {str(e)}"

if __name__ == "__main__":
    # FastMCP automatically configures the bi-directional stdio channel handler
    mcp.run()