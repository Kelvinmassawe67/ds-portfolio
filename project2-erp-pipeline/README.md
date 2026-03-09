# Project 2: ERP Data Pipeline & ETL Automation

## Business Problem
ERP systems contain valuable operational data, but extracting and preparing it for analytics requires complex, manual processes. This project demonstrates a fully automated ETL pipeline.

## Architecture
```
[ERP Source] → [Extract] → [Transform] → [Load] → [SQLite DW] → [Analytics]
```

## Pipeline Steps
1. **Extract**: Simulate ERP API pull (customers, products, invoices, line items)
2. **Transform**: Clean, enrich, compute KPIs (CLV, margins, overdue rates)
3. **Load**: Push to structured SQLite data warehouse with analytical views

## Scale
- 1,000 customers | 500 products | 5,000 invoices | 20,000+ line items
- Completed in ~8 seconds end-to-end

## Analytical Outputs
- `mart_fact_invoices` — enriched invoice fact table
- `mart_monthly_revenue` — monthly aggregated revenue
- `mart_customer_ltv` — customer lifetime value
- `mart_product_performance` — product margin analysis

## Run
```bash
python src/etl_pipeline.py
```

## Technologies
Python | Pandas | SQLite | SQL
