"""Environment-aware entrypoint for the Foam-Agent FastMCP server."""

from __future__ import annotations
import os
from .cli import run_server
from .fastmcp_server import mcp

if __name__ == "__main__":  # pragma: no cover
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "7860"))

    print(f"Starting FastMCP server transport={transport} host={host} port={port}")

    run_server(mcp, transport=transport, host=host, port=port)

