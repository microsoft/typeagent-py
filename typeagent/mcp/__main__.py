# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry point for running the MCP server as a module."""

from typeagent.mcp.server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
