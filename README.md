## Ollama
Install Ollama. Pull model(s):

`ollama pull mxbai-embed-large`

## Create Python env and install dependencies

```
python -m venv .venv
source .vent/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Testing and Debugging with MCP Tools

The Model Context Protocol (MCP) ecosystem provides powerful developer utilities to test your vector database search tool locally before deploying it to an AI client.

### Interactive Browser Testing 
To launch a local, network-isolated server instance that bypasses standard Windows process management and standard input/output bottlenecks, utilize the Server-Sent Events (SSE) transport protocol on a dedicated port:

1. Launch the Inspector UI via the `fastmcp` CLI:
```bash
fastmcp dev inspector mcp_chroma.py
```

2. Start the server via the `fastmcp` CLI:
```bash
fastmcp run mcp_chroma.py --transport sse --port 8001
```
3. Change the Transport Type dropdown from stdio to SSE.

4. Set the URL target box to http://localhost:8001/sse and click Connect.

5. Once connected, navigate to the Tools tab to run mock vector database queries and evaluate similarity score outputs directly in the dashboard interface. List tools, select `search`.


## Set as Cline MCP server

```
{
  "mcpServers": {
    "knowledge-base": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

## Key Benefits: v2 vs. v1

* **Data Completeness:** v2 fixes the crash bugs (ToC dots and token inflation). Every single page and technical table is now safely indexed—no more skipped chapters or missing data.
* **Accuracy:** By reducing chunks from 512 to 384 tokens, v2 isolates technical definitions and schema attributes perfectly without mixing them with unrelated layout text.
* **Better Technical Understanding:** Moving to the `mxbai-embed-large` model gives the AI a major upgrade in reading structured data, tables, and complex UML code variables.
* **Local Privacy & Speed:** By moving completely to a local Ollama engine, searches execute instantly with zero cloud dependency.
* **Stability:** v2 handles documents sequentially with custom safety buffers, preventing network congestion and memory crashes during heavy file processing.
* **Word format support:** v2 add support for .docx documents.