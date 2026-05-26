"""
Standalone MCP server (Phase 6) — exposes the retail query tool over stdio.

Phases 1-5 used an *in-process* MCP server (`create_sdk_mcp_server`): the
tool ran inside the notebook's own Python process. This file is the
*standalone* version — a separate program. The Claude Agent SDK launches
it as a subprocess and talks to it over stdin/stdout (the "stdio"
transport). That is how real, shareable MCP servers work.

Run it directly to sanity-check that it starts:

    python mcp_servers/retail_server.py

(it will then sit waiting for MCP messages on stdin — Ctrl+C to stop.)

IMPORTANT: a stdio MCP server must NOT print to stdout — stdout is the
protocol channel, and a stray print would corrupt the messages. Keep this
file free of print() calls.
"""

import os
import sys
from pathlib import Path

# This script may be launched from any working directory, so anchor it to
# the project root (the folder that contains both 'orchestrator/' and 'data/').
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mcp.server.fastmcp import FastMCP

from orchestrator import tools

# The server's name. The agent sees this tool as: mcp__retail__query_retail
mcp = FastMCP("retail")


@mcp.tool()
def query_retail(
    year: int = 0,
    country: str = "",
    top_n: int = 10,
    group_by: str = "StockCode",
    metric: str = "revenue",
) -> list:
    """Query the retail transactions dataset for the top N entries ranked by a metric.

    Use this for any 'top N' question about products, countries, or customers.

    Args:
        year: Calendar year filter, e.g. 2011. Use 0 for all years.
        country: Country filter, e.g. "United Kingdom". Use "" for all countries.
        top_n: How many rows to return.
        group_by: One of "StockCode", "Country", "Customer ID".
        metric: One of "revenue", "Quantity".
    """
    # The standalone server is a thin wrapper — the real query logic still
    # lives in orchestrator/tools.py. The server just exposes it over MCP.
    rows = tools.query_retail(
        year=year if year != 0 else None,
        country=country if country != "" else None,
        top_n=top_n,
        group_by=group_by,
        metric=metric,
    )
    return rows


if __name__ == "__main__":
    # FastMCP.run() defaults to the stdio transport.
    mcp.run()
