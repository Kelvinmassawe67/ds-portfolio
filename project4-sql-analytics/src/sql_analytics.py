"""
SQL-Based Business Analytics
Demonstrates advanced SQL queries for business intelligence using SQLite.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VIZ_DIR  = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, 'analytics.db')

np.random.seed(42)


# ── Schema & Data Setup ──────────────────────────────────────────────────────────
DDL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   TEXT PRIMARY KEY,
    customer_name TEXT,
    email         TEXT,
    segment       TEXT,
    region        TEXT,
    country       TEXT,
    signup_date   TEXT,
    is_active     INTEGER
);

CREATE TABLE IF NOT EXISTS products (
    product_id   TEXT PRIMARY KEY,
    product_name TEXT,
    category     TEXT,
    subcategory  TEXT,
    unit_price   REAL,
    unit_cost    REAL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    TEXT PRIMARY KEY,
    customer_id TEXT,
    order_date  TEXT,
    ship_date   TEXT,
    status      TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id    TEXT PRIMARY KEY,
    order_id   TEXT,
    product_id TEXT,
    quantity   INTEGER,
    unit_price REAL,
    discount   REAL,
    FOREIGN KEY (order_id)   REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

def seed_database(conn):
    print("  Seeding analytics database…")
    
    segments = ['Enterprise','SMB','Consumer','Government']
    regions  = ['North America','Europe','Asia Pacific','Latin America','MEA']
    
    customers = pd.DataFrame({
        'customer_id':   [f'C{i:04d}' for i in range(1,2001)],
        'customer_name': [f'Customer_{i}' for i in range(1,2001)],
        'email':         [f'customer{i}@example.com' for i in range(1,2001)],
        'segment':       np.random.choice(segments, 2000),
        'region':        np.random.choice(regions, 2000),
        'country':       np.random.choice(['USA','UK','Germany','Japan','Brazil','India'], 2000,
                                          p=[0.4,0.15,0.15,0.1,0.1,0.1]),
        'signup_date':   [(pd.Timestamp('2020-01-01') + pd.Timedelta(days=int(d))).strftime('%Y-%m-%d')
                          for d in np.random.randint(0,1460,2000)],
        'is_active':     np.random.choice([1,0], 2000, p=[0.88,0.12]),
    })
    
    cats = ['Electronics','Clothing','Furniture','Office Supplies','Technology','Appliances']
    sub  = {'Electronics':['Phones','Laptops','Tablets'],
            'Clothing':['Men','Women','Kids'],
            'Furniture':['Chairs','Desks','Shelves'],
            'Office Supplies':['Paper','Pens','Binders'],
            'Technology':['Accessories','Storage','Networking'],
            'Appliances':['Kitchen','Laundry','HVAC']}
    
    prod_rows = []
    for i in range(1,1001):
        cat = np.random.choice(cats)
        uc  = round(np.random.uniform(5,800),2)
        prod_rows.append({
            'product_id':   f'P{i:04d}',
            'product_name': f'{cat}_Item_{i}',
            'category':     cat,
            'subcategory':  np.random.choice(sub[cat]),
            'unit_price':   round(uc * np.random.uniform(1.3,2.5),2),
            'unit_cost':    uc,
        })
    products = pd.DataFrame(prod_rows)
    
    order_rows, item_rows = [], []
    statuses = ['Delivered','Processing','Shipped','Cancelled','Returned']
    for i in range(1,20001):
        ord_date = pd.Timestamp('2021-01-01') + pd.Timedelta(days=int(np.random.randint(0,1460)))
        cust_id  = f'C{np.random.randint(1,2001):04d}'
        order_rows.append({
            'order_id':   f'ORD{i:06d}',
            'customer_id': cust_id,
            'order_date':  ord_date.strftime('%Y-%m-%d'),
            'ship_date':  (ord_date + pd.Timedelta(days=int(np.random.randint(1,15)))).strftime('%Y-%m-%d'),
            'status':      np.random.choice(statuses, p=[0.7,0.1,0.1,0.05,0.05]),
        })
        for j in range(np.random.randint(1,6)):
            prod = f'P{np.random.randint(1,1001):04d}'
            up   = float(products.loc[products['product_id']==prod,'unit_price'].values[0]) \
                   if prod in products['product_id'].values else round(np.random.uniform(20,500),2)
            item_rows.append({
                'item_id':   f'ITEM{len(item_rows)+1:07d}',
                'order_id':   f'ORD{i:06d}',
                'product_id': prod,
                'quantity':   np.random.randint(1,20),
                'unit_price': up,
                'discount':   np.random.choice([0,0.05,0.1,0.15,0.2], p=[0.5,0.2,0.15,0.1,0.05]),
            })
    
    orders     = pd.DataFrame(order_rows)
    order_items = pd.DataFrame(item_rows)
    
    customers.to_sql('customers',   conn, if_exists='replace', index=False)
    products.to_sql('products',     conn, if_exists='replace', index=False)
    orders.to_sql('orders',         conn, if_exists='replace', index=False)
    order_items.to_sql('order_items', conn, if_exists='replace', index=False)
    
    print(f"    Customers: {len(customers):,} | Products: {len(products):,} | "
          f"Orders: {len(orders):,} | Items: {len(order_items):,}")


# ── SQL Queries ──────────────────────────────────────────────────────────────────
QUERIES = {
    "Q1_Monthly_Revenue": """
        SELECT
            strftime('%Y', o.order_date) AS year,
            strftime('%m', o.order_date) AS month,
            COUNT(DISTINCT o.order_id)   AS order_count,
            COUNT(DISTINCT o.customer_id) AS unique_customers,
            ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue,
            ROUND(AVG(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS avg_order_value
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.status NOT IN ('Cancelled','Returned')
        GROUP BY year, month
        ORDER BY year, month
    """,

    "Q2_Customer_Cohort_Revenue": """
        WITH first_purchase AS (
            SELECT customer_id,
                   strftime('%Y-%m', MIN(order_date)) AS cohort_month
            FROM orders
            WHERE status NOT IN ('Cancelled','Returned')
            GROUP BY customer_id
        ),
        customer_orders AS (
            SELECT o.customer_id,
                   fp.cohort_month,
                   strftime('%Y-%m', o.order_date) AS order_month,
                   SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN first_purchase fp ON o.customer_id = fp.customer_id
            WHERE o.status NOT IN ('Cancelled','Returned')
            GROUP BY o.customer_id, fp.cohort_month, order_month
        )
        SELECT cohort_month,
               COUNT(DISTINCT customer_id) AS cohort_size,
               ROUND(SUM(revenue), 2)      AS cohort_revenue,
               ROUND(AVG(revenue), 2)      AS avg_revenue_per_customer
        FROM customer_orders
        GROUP BY cohort_month
        ORDER BY cohort_month
        LIMIT 24
    """,

    "Q3_Product_Category_Performance": """
        SELECT
            p.category,
            p.subcategory,
            COUNT(DISTINCT oi.order_id)  AS orders,
            SUM(oi.quantity)             AS units_sold,
            ROUND(SUM(oi.quantity * oi.unit_price * (1-oi.discount)), 2) AS revenue,
            ROUND(SUM(oi.quantity * (oi.unit_price - p.unit_cost)), 2)   AS gross_profit,
            ROUND(
                100.0 * SUM(oi.quantity*(oi.unit_price-p.unit_cost))
                      / NULLIF(SUM(oi.quantity*oi.unit_price*(1-oi.discount)),0)
            , 1) AS margin_pct
        FROM order_items oi
        JOIN products p    ON oi.product_id = p.product_id
        JOIN orders o      ON oi.order_id   = o.order_id
        WHERE o.status NOT IN ('Cancelled','Returned')
        GROUP BY p.category, p.subcategory
        ORDER BY revenue DESC
    """,

    "Q4_Customer_Segmentation_RFM": """
        WITH customer_metrics AS (
            SELECT
                c.customer_id,
                c.customer_name,
                c.segment,
                c.region,
                CAST(julianday('2025-01-01') - julianday(MAX(o.order_date)) AS INTEGER) AS recency_days,
                COUNT(DISTINCT o.order_id) AS frequency,
                ROUND(SUM(oi.quantity * oi.unit_price * (1-oi.discount)), 2) AS monetary
            FROM customers c
            JOIN orders o      ON c.customer_id = o.customer_id
            JOIN order_items oi ON o.order_id   = oi.order_id
            WHERE o.status NOT IN ('Cancelled','Returned')
            GROUP BY c.customer_id
        ),
        scored AS (
            SELECT *,
                CASE WHEN recency_days <= 90  THEN 'High'
                     WHEN recency_days <= 180 THEN 'Medium'
                     ELSE 'Low' END AS recency_score,
                CASE WHEN frequency >= 10 THEN 'High'
                     WHEN frequency >= 5  THEN 'Medium'
                     ELSE 'Low' END AS frequency_score,
                CASE WHEN monetary >= 5000 THEN 'High'
                     WHEN monetary >= 1000 THEN 'Medium'
                     ELSE 'Low' END AS monetary_score
            FROM customer_metrics
        )
        SELECT
            segment, region,
            recency_score, frequency_score, monetary_score,
            COUNT(*) AS customer_count,
            ROUND(AVG(monetary), 2) AS avg_clv,
            ROUND(SUM(monetary), 2) AS total_revenue
        FROM scored
        GROUP BY segment, region, recency_score, frequency_score, monetary_score
        ORDER BY total_revenue DESC
        LIMIT 30
    """,

    "Q5_Window_Functions_Running_Total": """
        WITH monthly AS (
            SELECT
                strftime('%Y-%m', o.order_date) AS month,
                ROUND(SUM(oi.quantity * oi.unit_price * (1-oi.discount)), 2) AS revenue
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.status NOT IN ('Cancelled','Returned')
            GROUP BY month
        )
        SELECT
            month,
            revenue,
            ROUND(SUM(revenue) OVER (ORDER BY month), 2)                        AS running_total,
            ROUND(AVG(revenue) OVER (ORDER BY month ROWS 2 PRECEDING), 2)       AS moving_avg_3m,
            ROUND(revenue - LAG(revenue) OVER (ORDER BY month), 2)              AS mom_change,
            ROUND(100.0*(revenue - LAG(revenue) OVER (ORDER BY month))
                      / NULLIF(LAG(revenue) OVER (ORDER BY month),0), 1)        AS mom_pct_change
        FROM monthly
        ORDER BY month
    """,
}


def run_analytics(conn):
    results = {}
    for name, sql in QUERIES.items():
        try:
            df = pd.read_sql_query(sql, conn)
            results[name] = df
            df.to_csv(os.path.join(DATA_DIR, f'{name}.csv'), index=False)
            print(f"  ✓ {name}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return results


def visualize_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('SQL Business Analytics – Key Insights', fontsize=16, fontweight='bold')

    # Monthly revenue
    if 'Q1_Monthly_Revenue' in results:
        df = results['Q1_Monthly_Revenue'].copy()
        df['period'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str))
        for yr, grp in df.groupby('year'):
            axes[0,0].plot(grp['period'], grp['revenue']/1e3, marker='o', markersize=3,
                           linewidth=2, label=str(yr))
        axes[0,0].set_title('Monthly Revenue Trend', fontweight='bold')
        axes[0,0].set_ylabel('Revenue ($K)'); axes[0,0].legend()
        axes[0,0].grid(axis='y', alpha=0.3)

    # Category performance
    if 'Q3_Product_Category_Performance' in results:
        df = results['Q3_Product_Category_Performance']
        cat_agg = df.groupby('category')[['revenue','gross_profit']].sum().sort_values('revenue', ascending=False)
        x = np.arange(len(cat_agg))
        axes[0,1].bar(x-0.2, cat_agg['revenue']/1e3,    0.4, label='Revenue',      color='#1a73e8')
        axes[0,1].bar(x+0.2, cat_agg['gross_profit']/1e3,0.4, label='Gross Profit', color='#34a853')
        axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(cat_agg.index, rotation=30, ha='right')
        axes[0,1].set_title('Revenue & Profit by Category', fontweight='bold')
        axes[0,1].set_ylabel('$K'); axes[0,1].legend()

    # Running total & moving average
    if 'Q5_Window_Functions_Running_Total' in results:
        df = results['Q5_Window_Functions_Running_Total']
        ax = axes[1,0]; ax2 = ax.twinx()
        ax.bar(range(len(df)), df['revenue']/1e3, color='#a8c4e0', alpha=0.7, label='Monthly Revenue')
        ax2.plot(range(len(df)), df['running_total']/1e6, color='#e74c3c', linewidth=2, label='Running Total')
        ax2.plot(range(len(df)), df['moving_avg_3m']/1e3, color='#f39c12', linewidth=1.5,
                 linestyle='--', label='3M Moving Avg')
        ax.set_title('Revenue with Running Total (Window Functions)', fontweight='bold')
        ax.set_ylabel('Monthly Revenue ($K)'); ax2.set_ylabel('Cumulative ($M)', color='#e74c3c')
        ax.set_xticks(range(0, len(df), 4))
        ax.set_xticklabels(df['month'].iloc[::4].tolist(), rotation=45, ha='right', fontsize=8)
        lines1, _ = ax.get_legend_handles_labels()
        lines2, _ = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, ['Monthly','Cumulative','3M Avg'], fontsize=8)

    # Customer segmentation
    if 'Q4_Customer_Segmentation_RFM' in results:
        df = results['Q4_Customer_Segmentation_RFM']
        seg_rev = df.groupby('segment')['total_revenue'].sum()
        colors  = ['#1a73e8','#34a853','#fa7b17','#ea4335']
        axes[1,1].pie(seg_rev, labels=seg_rev.index, autopct='%1.1f%%',
                      colors=colors[:len(seg_rev)], startangle=90)
        axes[1,1].set_title('Revenue Share by Customer Segment', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'sql_analytics.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Visualisation saved → {out} ✓")


def main():
    print("=" * 60)
    print("PROJECT 4 – SQL-Based Business Analytics")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)

    cur = conn.execute("SELECT COUNT(*) FROM customers")
    if cur.fetchone()[0] == 0:
        seed_database(conn)
        conn.commit()
    else:
        print("  Database already seeded ✓")

    print("\n  Running analytical queries…")
    results = run_analytics(conn)

    print("\n  Building visualisations…")
    visualize_results(results)

    # Summary
    rev_df = results.get('Q1_Monthly_Revenue')
    if rev_df is not None:
        total = rev_df['revenue'].sum()
        print(f"\n{'='*60}")
        print(f"  Total Revenue (all years) : ${total/1e6:.2f}M")
        print(f"  Avg Monthly Revenue       : ${rev_df['revenue'].mean()/1e3:.1f}K")
        print(f"  Peak Month Revenue        : ${rev_df['revenue'].max()/1e3:.1f}K")
        print(f"{'='*60}")

    conn.close()


if __name__ == '__main__':
    main()
