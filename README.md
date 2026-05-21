## Ensure virtual environment is active before running
pip install \
  fastmcp \
  chromadb \
  python-dotenv \
  pymupdf4llm \
  sentence-transformers \
  llama-index-core \
  llama-index-readers-file \
  llama-index-vector-stores-chroma \
  llama-index-embeddings-huggingface \
  llama-index-embeddings-deepseek

## Set as Cline MCP server

```
{
  "mcpServers": {
    "s100-knowledge-base": {
      "command": "C:\\Users\\name\\project\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\username\\project\\mcp_chroma.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "API_KEY": "api_key",
        "CHROMA_DB_PATH": ""
      }
    }
  }
}
```