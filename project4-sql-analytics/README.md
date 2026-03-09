# Project 4: SQL-Based Business Analytics

## Business Problem
Organizations need complex SQL analytics to generate business intelligence from relational databases — going beyond simple queries to cohort analysis, window functions, and multi-table joins.

## SQL Queries Demonstrated
| Query | Technique | Business Value |
|-------|-----------|----------------|
| Q1 Monthly Revenue | GROUP BY + aggregates | Revenue trending |
| Q2 Customer Cohorts | CTEs + self-join | Retention analysis |
| Q3 Category Performance | Multi-table JOIN + margins | Product profitability |
| Q4 RFM Segmentation | CASE WHEN + subqueries | Customer tiering |
| Q5 Window Functions | LAG, SUM OVER, AVG OVER | MoM growth, running totals |

## Database Schema
```
customers → orders → order_items ← products
```

## Scale
- 2,000 customers | 1,000 products | 20,000 orders | ~60,000 items
- Total revenue analyzed: $396M across 4 years

## Run
```bash
python src/sql_analytics.py
```

## Technologies
Python | SQLite | Pandas | SQL Window Functions
