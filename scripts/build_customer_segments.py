"""
RFM-based customer segmentation.

Reads retail.parquet, computes Recency / Frequency / Monetary per customer,
buckets into VIP / Regular / At-Risk / Churned, writes customer_segments.csv.

Used downstream by Phase 5 (RAG corpus) and Phase 8 (guardrails allowlist).

Usage:
    python scripts/build_customer_segments.py
"""

from pathlib import Path
import pandas as pd

INPUT_PARQUET = Path("data/retail.parquet")
OUTPUT_CSV = Path("data/customer_segments.csv")


def assign_segment(row):
    """Priority-ordered segment assignment."""
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "VIP"
    if r <= 2 and f >= 3:
        return "At-Risk"
    if r <= 2 and f <= 2 and m <= 2:
        return "Churned"
    return "Regular"


def main():
    print(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  Loaded {len(df):,} rows")

    # Drop anonymous rows and returns
    df = df.dropna(subset=["Customer ID"])
    df = df[df["Quantity"] > 0]
    print(f"  After cleaning: {len(df):,} rows")

    # "Today" = day after the last transaction in the dataset
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # RFM per customer
    rfm = df.groupby("Customer ID").agg(
        recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        frequency=("Invoice", "nunique"),
        monetary=("revenue", "sum"),
    )

    # Quintile scores (1-5). Recency: LOWER is better, so labels reversed.
    rfm["R_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)

    rfm["rfm_score"] = (
        rfm["R_score"].astype(str)
        + rfm["F_score"].astype(str)
        + rfm["M_score"].astype(str)
    )

    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(OUTPUT_CSV)

    # Sanity checks
    print(f"\nWrote {OUTPUT_CSV}")
    print(f"  Customers: {len(rfm):,}")
    print(f"\nSegment distribution:")
    print(rfm["segment"].value_counts())
    valid = {"VIP", "Regular", "At-Risk", "Churned"}
    assert set(rfm["segment"].unique()) <= valid, "Found an unexpected segment label"
    assert (rfm["segment"] == "VIP").sum() >= 100, "Expected at least 100 VIPs"
    print("\n[OK] Segmentation successful.")


if __name__ == "__main__":
    main()
