"""Console entry point for `foamagent-mcp` command.

Usage (after pip install):
    foamagent-mcp                     # stdio mode (default, for MCP clients)
    foamagent-mcp --transport http    # HTTP mode (for web clients)
    foamagent-mcp --help

Integration with AI tools:
    claude mcp add foamagent -- foamagent-mcp
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="foamagent-mcp",
        description="Foam-Agent MCP server — exposes OpenFOAM CFD simulation tools over MCP",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport method (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for HTTP transport (default: 7860)",
    )
    args = parser.parse_args()

    from foamagent.mcp.fastmcp_server import mcp

    if args.transport == "http":
        uvicorn_config = {"ws": "websockets"}
        mcp.run("http", host=args.host, port=args.port, uvicorn_config=uvicorn_config)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
