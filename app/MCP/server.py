from mcp.server import MCPServer

from app.MCP.ReRankingTool import rerank_chunks
from app.MCP.ContextAssemblingTool import assemble_context

mcp = MCPServer("Retrieval MCP")

mcp.tool()(rerank_chunks)
mcp.tool()(assemble_context)

if __name__ == "__main__":
    mcp.run()