"""
Agent tools.

A "tool" is just a Python function the agent is allowed to call. The agent
decides WHEN to call it based on the tool's name + description.

In Phase 1 we define ONE tool: query_retail. Phase 2 adds more.
"""

from pathlib import Path
import pandas as pd

# Cache the dataframe at module load — every notebook reuses the same one.
_RETAIL_PATH = Path("data/retail.parquet")
_df: pd.DataFrame | None = None


def get_retail_df() -> pd.DataFrame:
    """Load (or return cached) retail dataframe."""
    global _df
    if _df is None:
        _df = pd.read_parquet(_RETAIL_PATH)
    return _df


def query_retail(
    year: int | None = None,
    country: str | None = None,
    top_n: int = 10,
    group_by: str = "StockCode",
    metric: str = "revenue",
) -> list[dict]:
    """
    Return the top N rows aggregated by `group_by` and sorted by `metric`.

    Parameters
    ----------
    year     : optional year filter (e.g., 2010)
    country  : optional country filter (e.g., "United Kingdom")
    top_n    : how many rows to return
    group_by : column to group by (e.g., "StockCode", "Country", "Customer ID")
    metric   : column to sort by, descending (e.g., "revenue", "Quantity")

    Returns
    -------
    list of dicts, one per row, e.g.:
        [{"StockCode": "85123A", "revenue": 132456.78}, ...]
    """
    df = get_retail_df()

    # Drop returns (negative quantity) for revenue-style questions
    df = df[df["Quantity"] > 0]

    # Apply optional filters
    if year is not None:
        df = df[df["InvoiceDate"].dt.year == year]
    if country is not None:
        df = df[df["Country"] == country]


    # Group, sum the metric, take top N
    result = (
        df.groupby(group_by, dropna=False)[metric]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    return result.to_dict(orient="records")
