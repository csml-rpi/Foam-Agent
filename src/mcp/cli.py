"""Shared startup for the console and module MCP entry points."""

from __future__ import annotations

import argparse


def run_server(server, *, transport: str, host: str, port: int) -> None:
    if transport == "http":
        server.run("http", host=host, port=port, uvicorn_config={"ws": "websockets"})
    else:
        server.run("stdio")


def main(argv=None, *, server=None, default_transport="stdio", default_host="0.0.0.0") -> None:
    parser = argparse.ArgumentParser(
        prog="foamagent-mcp",
        description="Foam-Agent MCP server — exposes OpenFOAM CFD simulation tools over MCP",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=default_transport,
        help="Transport method (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=default_host,
        help="Host for HTTP transport (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for HTTP transport (default: 7860)",
    )
    args = parser.parse_args(argv)

    # The server imports FastMCP before adding src/ to sys.path, so the local
    # src/mcp directory cannot shadow the third-party mcp package.
    if server is None:
        from src.mcp.fastmcp_server import mcp

        server = mcp
    run_server(server, transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
