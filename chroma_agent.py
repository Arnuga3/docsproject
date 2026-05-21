import os
import asyncio
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Use ChatEngine architecture to force strict database integration
from llama_index.core.chat_engine import ContextChatEngine

load_dotenv()

LLM_KEY = os.getenv("LLM_API_KEY")
Settings.llm = DeepSeek(model="deepseek-chat", api_key=LLM_KEY, temperature=0.0) # Temperature 0.0 eliminates creative guessing
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

async def launch_chroma_agent():
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"ChromaDB folder missing at '{CHROMA_DB_PATH}'. Run ingestion script first.")

    print("🔌 Connecting to local ChromaDB engine...")
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db_client.get_collection("knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Create a specialized retriever that pulls n dense text blocks
    retriever = index.as_retriever(similarity_top_k=5)
    
    # Build a strict System Prompt that forces metadata parsing and blocks external knowledge
    system_template = (
        "You are an expert Software Engineering Assistant.\n"
        "Your responses must be derived EXCLUSIVELY from the provided database context chunks below.\n"
        "If the context does not contain the specific answer, say: 'I cannot find that information in the ingested documentation.'\n"
        "Do not use external pre-trained knowledge or historical outdated standards if they are not in the text below.\n\n"
        
        "CRITICAL CITATION RULES:\n"
        "Every chunk below is preceded by its metadata. You MUST extract the 'file_name' and 'page_label' "
        "and append an explicit citation to every technical statement you make. Example: [Source: file_name.pdf, Page X].\n\n"
        
        "Database Context:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
    )
    
    # Instantiate the ContextEngine (Forces a database pull *before* LLM is called)
    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        llm=Settings.llm,
        system_prompt=system_template
    )

    print("\n Strict Context Engine Connected to ChromaDB successfully!")
    print("-" * 50)
    
    while True:
        query = input("\n Prompt your database: ")
        if query.strip().lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue
            
        print("🔍 Searching vector space and verifying facts...")
        
        # In ContextChatEngine, this is synchronous and stable
        response = chat_engine.chat(query)
        
        print(f"\n Agent Response:\n{response}") 

if __name__ == "__main__":
    asyncio.run(launch_chroma_agent())