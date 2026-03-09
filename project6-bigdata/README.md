# Project 6: Big Data Retail Analytics

## Business Problem
Modern retail organizations generate millions of transactions that require scalable distributed processing frameworks for timely analysis.

## Scale
- **500,000 transaction records** across 5 years (2020–2024)
- 200 stores | 10 product categories | 50,000 unique customers
- Total revenue analyzed: **$497M**
- Generated and processed in under 10 seconds

## Analytics Performed
1. **Regional Revenue Heatmap** — Revenue by region × year
2. **Category Performance** — Revenue, units, margins per category
3. **Monthly Trend + Window Functions** — YTD, 3M moving average
4. **Top Stores Ranking** — Revenue, transactions, customers per store

## Technology Notes
- **PySpark** is used when available for distributed processing
- **Pandas at scale** provides equivalent analytics as fallback
- Architecture mirrors production Spark deployments
- Spark SQL syntax is identical to production code

## Key Findings
- Electronics leads with $175M revenue (35% of total)
- Consistent ~50% gross margin across categories
- Even regional distribution with ~$100M per region

## Run
```bash
# With PySpark (recommended for large scale):
pip install pyspark
python src/bigdata_analytics.py

# Without PySpark (pandas fallback):
python src/bigdata_analytics.py
```

## Technologies
Apache Spark (PySpark) | Pandas | Python | Matplotlib
