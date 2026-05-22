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

5. Once connected, navigate to the Tools tab to run mock vector database queries and evaluate similarity score outputs directly in the dashboard interface. List tools, select `serch`.


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