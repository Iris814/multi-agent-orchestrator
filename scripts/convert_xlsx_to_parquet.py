"""
Convert online_retail_II.xlsx to retail.parquet for fast loading.

Run once during Phase 0 setup. Parquet loads ~15x faster than .xlsx.

Usage:
    python scripts/convert_xlsx_to_parquet.py
"""

from pathlib import Path
import pandas as pd

INPUT_XLSX = Path("../eCommerce_cust_behavior/online_retail_II.xlsx")
OUTPUT_PARQUET = Path("data/retail.parquet")


def main():
    print(f"Reading {INPUT_XLSX}...")

    # Read both sheets ("Year 2009-2010" and "Year 2010-2011")
    sheets = pd.read_excel(INPUT_XLSX, sheet_name=None)
    print(f"  Found sheets: {list(sheets.keys())}")

    # Stack vertically into one dataframe
    df = pd.concat(sheets.values(), ignore_index=True)

    # Add derived revenue column (vectorized — no Python loop)
    df["revenue"] = df["Quantity"] * df["Price"]

    # Ensure InvoiceDate is a real datetime (downstream time-ops need this)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Coerce all object columns to string. Parquet can't store mixed types,
    # and several columns (Invoice, StockCode, Description) have rows with
    # numbers AND rows with letters (e.g., Invoice "C489449" = cancellation,
    # StockCode "POST" = postage).
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    # Save to columnar binary format
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    # Sanity checks
    print(f"\nWrote {OUTPUT_PARQUET}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dates:   {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    assert len(df) > 1_000_000, "Expected >1M rows — did you concat both sheets?"
    assert "revenue" in df.columns, "Did you add the revenue column?"
    print("\n[OK] Conversion successful.")


if __name__ == "__main__":
    main()
