"""
ERP Data Pipeline & ETL Automation
Simulates extraction, transformation, and loading of ERP system data.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
import hashlib
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, 'erp_warehouse.db')


# ── Synthetic ERP Data Generation ──────────────────────────────────────────────
def generate_erp_data():
    print("  Generating ERP source data…")

    # Customers
    cities   = ['New York','Los Angeles','Chicago','Houston','Phoenix','Philadelphia','Dallas','London','Berlin','Paris']
    segments = ['Enterprise','SMB','Startup','Government','Non-profit']
    customers = pd.DataFrame({
        'customer_id':   [f'C{i:04d}' for i in range(1, 1001)],
        'company_name':  [f'Company_{i}' for i in range(1, 1001)],
        'segment':       np.random.choice(segments, 1000),
        'city':          np.random.choice(cities, 1000),
        'country':       np.random.choice(['USA','UK','Germany','France','Canada'], 1000, p=[0.5,0.2,0.1,0.1,0.1]),
        'credit_limit':  np.random.randint(5000, 500000, 1000),
        'created_date':  [(datetime(2020,1,1) + timedelta(days=int(d))).strftime('%Y-%m-%d')
                          for d in np.random.randint(0, 1460, 1000)],
        'is_active':     np.random.choice([1, 0], 1000, p=[0.92, 0.08]),
    })

    # Products
    categories   = ['Software','Hardware','Services','Consulting','Support','Training']
    product_rows = []
    for pid in range(1, 501):
        cat  = np.random.choice(categories)
        cost = round(np.random.uniform(10, 5000), 2)
        product_rows.append({
            'product_id':   f'P{pid:04d}',
            'product_name': f'{cat}_Product_{pid}',
            'category':     cat,
            'unit_cost':    cost,
            'unit_price':   round(cost * np.random.uniform(1.2, 3.0), 2),
            'stock_qty':    np.random.randint(0, 1000),
            'is_active':    np.random.choice([1, 0], p=[0.9, 0.1]),
        })
    products = pd.DataFrame(product_rows)

    # Invoices & line items
    invoice_rows, line_rows = [], []
    statuses = ['Paid','Pending','Overdue','Cancelled']
    for inv_id in range(1, 5001):
        cust_id = f'C{np.random.randint(1,1001):04d}'
        inv_date = datetime(2022,1,1) + timedelta(days=int(np.random.randint(0, 1095)))
        n_lines  = np.random.randint(1, 8)
        subtotal = 0.0

        for _ in range(n_lines):
            prod_id = f'P{np.random.randint(1,501):04d}'
            qty     = np.random.randint(1, 50)
            price   = products.loc[products['product_id']==prod_id, 'unit_price'].values
            price   = float(price[0]) if len(price) else round(np.random.uniform(50,2000),2)
            line_total = round(qty * price, 2)
            subtotal  += line_total
            line_rows.append({
                'line_id':     f'L{len(line_rows)+1:06d}',
                'invoice_id':  f'INV{inv_id:05d}',
                'product_id':  prod_id,
                'quantity':    qty,
                'unit_price':  price,
                'line_total':  line_total,
                'discount_pct': np.random.choice([0,5,10,15,20], p=[0.5,0.2,0.15,0.1,0.05]),
            })

        tax  = round(subtotal * 0.08, 2)
        disc = round(subtotal * np.random.uniform(0, 0.15), 2)
        invoice_rows.append({
            'invoice_id':   f'INV{inv_id:05d}',
            'customer_id':  cust_id,
            'invoice_date': inv_date.strftime('%Y-%m-%d'),
            'due_date':     (inv_date + timedelta(days=30)).strftime('%Y-%m-%d'),
            'subtotal':     round(subtotal, 2),
            'tax_amount':   tax,
            'discount':     disc,
            'total_amount': round(subtotal + tax - disc, 2),
            'status':       np.random.choice(statuses, p=[0.7,0.15,0.1,0.05]),
            'sales_rep':    f'REP{np.random.randint(1,26):02d}',
            'region':       np.random.choice(['North','South','East','West','International']),
        })

    invoices   = pd.DataFrame(invoice_rows)
    line_items = pd.DataFrame(line_rows)

    return customers, products, invoices, line_items


# ── ETL Transform ───────────────────────────────────────────────────────────────
def transform_data(customers, products, invoices, line_items):
    print("  Transforming data…")

    # Clean customers
    customers_clean = customers.copy()
    customers_clean['company_name'] = customers_clean['company_name'].str.strip().str.title()
    customers_clean['created_date'] = pd.to_datetime(customers_clean['created_date'])
    customers_clean['customer_age_days'] = (pd.Timestamp.now() - customers_clean['created_date']).dt.days

    # Clean invoices
    invoices_clean = invoices.copy()
    invoices_clean['invoice_date'] = pd.to_datetime(invoices_clean['invoice_date'])
    invoices_clean['due_date']     = pd.to_datetime(invoices_clean['due_date'])
    invoices_clean['year']         = invoices_clean['invoice_date'].dt.year
    invoices_clean['month']        = invoices_clean['invoice_date'].dt.month
    invoices_clean['quarter']      = invoices_clean['invoice_date'].dt.quarter
    invoices_clean['is_overdue']   = (
        (invoices_clean['status'] == 'Overdue') |
        ((invoices_clean['due_date'] < pd.Timestamp.now()) &
         (invoices_clean['status'] == 'Pending'))
    ).astype(int)

    # Fact table: invoice enriched
    fact = invoices_clean.merge(customers_clean[['customer_id','company_name','segment','country','city']],
                                on='customer_id', how='left')

    # Aggregate: monthly revenue
    monthly_rev = (invoices_clean[invoices_clean['status']=='Paid']
                   .groupby(['year','month'])
                   .agg(total_revenue=('total_amount','sum'),
                        invoice_count=('invoice_id','count'),
                        avg_invoice=('total_amount','mean'))
                   .reset_index())

    # Aggregate: customer lifetime value
    clv = (invoices_clean[invoices_clean['status'].isin(['Paid'])]
           .groupby('customer_id')
           .agg(total_spent=('total_amount','sum'),
                invoice_count=('invoice_id','count'),
                first_purchase=('invoice_date','min'),
                last_purchase=('invoice_date','max'))
           .reset_index())
    clv['customer_lifetime_days'] = (clv['last_purchase'] - clv['first_purchase']).dt.days
    clv['avg_invoice_value']      = clv['total_spent'] / clv['invoice_count']

    # Product performance
    prod_perf = (line_items.merge(products[['product_id','category','unit_cost']], on='product_id')
                 .groupby(['product_id','category'])
                 .agg(total_revenue=('line_total','sum'),
                      units_sold=('quantity','sum'),
                      order_count=('invoice_id','count'))
                 .reset_index())
    prod_perf['gross_margin'] = (prod_perf['total_revenue'] -
                                  prod_perf['units_sold'] * products.set_index('product_id')['unit_cost'].reindex(prod_perf['product_id']).values) / prod_perf['total_revenue']

    print(f"    Fact table: {len(fact):,} rows")
    print(f"    Monthly revenue: {len(monthly_rev)} periods")
    print(f"    CLV records: {len(clv):,}")
    print(f"    Product performance: {len(prod_perf):,}")

    return {'fact_invoices': fact, 'monthly_revenue': monthly_rev,
            'customer_ltv': clv, 'product_performance': prod_perf}


# ── Load to Data Warehouse ──────────────────────────────────────────────────────
def load_to_warehouse(raw_tables, analytics_tables):
    print(f"  Loading to SQLite warehouse: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Raw / staging
    for name, df in raw_tables.items():
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['datetime64']).columns:
            df_copy[col] = df_copy[col].astype(str)
        df_copy.to_sql(f'stg_{name}', conn, if_exists='replace', index=False)

    # Analytics / mart
    for name, df in analytics_tables.items():
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['datetime64']).columns:
            df_copy[col] = df_copy[col].astype(str)
        df_copy.to_sql(f'mart_{name}', conn, if_exists='replace', index=False)

    # Create analytical views
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_regional_revenue AS
        SELECT region, year, SUM(total_amount) AS revenue,
               COUNT(invoice_id) AS invoice_count,
               AVG(total_amount) AS avg_invoice
        FROM mart_fact_invoices
        WHERE status = 'Paid'
        GROUP BY region, year
    """)

    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_top_customers AS
        SELECT customer_id, company_name, segment,
               total_spent, invoice_count, avg_invoice_value,
               RANK() OVER (ORDER BY total_spent DESC) AS revenue_rank
        FROM mart_customer_ltv
        ORDER BY total_spent DESC
        LIMIT 100
    """)

    conn.commit()
    conn.close()
    print("  Warehouse loaded ✓")


# ── Pipeline Orchestrator ───────────────────────────────────────────────────────
def run_pipeline():
    start = datetime.now()
    log   = []

    def log_step(step, status, msg=''):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.append({'timestamp': ts, 'step': step, 'status': status, 'message': msg})
        print(f"  [{ts}] {step}: {status} {msg}")

    print("=" * 60)
    print("PROJECT 2 – ERP Data Pipeline & ETL Automation")
    print("=" * 60)

    print("\n[EXTRACT]")
    customers, products, invoices, line_items = generate_erp_data()
    for name, df in [('customers', customers), ('products', products),
                     ('invoices', invoices), ('line_items', line_items)]:
        df.to_csv(os.path.join(DATA_DIR, f'{name}_raw.csv'), index=False)
    log_step('Extract', 'SUCCESS', f'{len(invoices):,} invoices extracted')

    print("\n[TRANSFORM]")
    analytics = transform_data(customers, products, invoices, line_items)
    for name, df in analytics.items():
        df.to_csv(os.path.join(DATA_DIR, f'{name}.csv'), index=False)
    log_step('Transform', 'SUCCESS', f'{len(analytics)} analytical tables created')

    print("\n[LOAD]")
    raw_tables = {'customers': customers, 'products': products,
                  'invoices': invoices, 'line_items': line_items}
    load_to_warehouse(raw_tables, analytics)
    log_step('Load', 'SUCCESS', f'Data warehouse: {DB_PATH}')

    duration = (datetime.now() - start).total_seconds()
    log_step('Pipeline', 'COMPLETE', f'Duration: {duration:.1f}s')

    pd.DataFrame(log).to_csv(os.path.join(DATA_DIR, 'pipeline_log.csv'), index=False)

    print(f"\n{'='*60}")
    print(f"  Pipeline completed in {duration:.1f} seconds")
    print(f"  Records processed:")
    print(f"    Customers  : {len(customers):,}")
    print(f"    Products   : {len(products):,}")
    print(f"    Invoices   : {len(invoices):,}")
    print(f"    Line Items : {len(line_items):,}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_pipeline()
