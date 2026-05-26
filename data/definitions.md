# Business Definitions — Customer Analytics Glossary

> This is the corpus retrieved by the RAG agent in Phase 5. Add to it as your project grows. Each definition becomes a "fact" the agent can cite.

## Customer Segments (RFM-derived)

**VIP** — Customers with high Recency, Frequency, AND Monetary scores (R>=4, F>=4, M>=4 on a 1–5 scale). These are the most valuable customers; retention is critical.

**Regular** — The default bucket. Active customers with moderate engagement. Stable revenue contributors.

**At-Risk** — Customers who *used to* purchase frequently but haven't bought recently (R<=2 AND F>=3). Strong candidates for win-back campaigns.

**Churned** — Customers with low Recency, Frequency, and Monetary (R<=2 AND F<=2 AND M<=2). Long since lapsed. Acquisition cost is typically lower for win-back than for new acquisition, but probability of reactivation is also lower.

## RFM Scoring

**Recency (R)** — Days since the customer's most recent purchase. Lower = more recent = better. Scored 1–5 via quintiles (5 = most recent).

**Frequency (F)** — Number of distinct purchase occasions. Higher = better. Scored 1–5.

**Monetary (M)** — Total revenue from the customer. Higher = better. Scored 1–5.

**RFM Score** — Concatenation of R, F, M scores (e.g., "555" = best, "111" = worst).

## Key Metrics

**Revenue** — `Quantity * Price` per line item, summed. Excludes returns (negative quantities) unless analyzing returns specifically.

**LTV (Lifetime Value)** — Total revenue from a customer across all their purchases to date. For this dataset, the observed period (Dec 2009 – Dec 2011) serves as the "lifetime."

**Retention Rate** — (Customers active in period N who were also active in period N-1) / (Customers active in period N-1). Measured monthly here.

**Cohort** — A group of customers grouped by their first-purchase month. Cohort retention tracks how long each cohort sticks around.

**Churn** — A customer is considered churned if Recency > 180 days at the snapshot date.

## Geographic Notes

The dataset is dominated by UK customers (~90%). Other notable countries: Germany, France, EIRE (Ireland), Spain. International cohorts behave differently — generally higher AOV but lower frequency.

## Data Quirks (good for guardrails to know)

- ~25% of rows have null Customer ID (typically anonymous in-store sales). Excluded from customer-level analyses.
- Negative Quantity values are returns. Excluded from revenue calculations unless analyzing returns specifically.
- "POST", "BANK CHARGES", "AMAZON FEE" are non-product StockCodes — exclude from product analyses.
- Prices in GBP (£). Convert if needed for cross-region comparison.
