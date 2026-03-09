"""
Big Data Retail Analytics
Demonstrates distributed data processing using PySpark (with pandas fallback).
Analyzes large-scale retail datasets with aggregations and insights.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Try PySpark
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import *
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VIZ_DIR  = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIZ_DIR,  exist_ok=True)

np.random.seed(42)
N_RECORDS = 500_000  # Simulate large dataset


# ── Large Dataset Generation ─────────────────────────────────────────────────────
def generate_large_retail_data(n=N_RECORDS):
    print(f"  Generating {n:,} retail transaction records…")
    t0 = time.time()

    regions    = ['North', 'South', 'East', 'West', 'International']
    categories = ['Electronics', 'Clothing', 'Groceries', 'Home', 'Sports',
                  'Beauty', 'Toys', 'Automotive', 'Books', 'Food']
    stores     = [f'STR{i:03d}' for i in range(1, 201)]
    
    base_prices = {'Electronics':350,'Clothing':65,'Groceries':35,'Home':110,
                   'Sports':80,'Beauty':50,'Toys':35,'Automotive':200,'Books':25,'Food':45}

    chunk = 50_000
    all_chunks = []
    for start in range(0, n, chunk):
        end  = min(start + chunk, n)
        size = end - start

        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        trans_dates = pd.to_datetime(np.random.choice(dates, size))
        cats   = np.random.choice(categories, size)
        regs   = np.random.choice(regions, size)
        strs   = np.random.choice(stores, size)
        prices = np.array([base_prices[c] * np.random.uniform(0.7,1.4) for c in cats])
        costs  = prices * np.random.uniform(0.35, 0.65, size)
        qtys   = np.random.randint(1, 20, size)
        disc   = np.random.choice([0,0.05,0.1,0.15,0.2], size, p=[0.5,0.2,0.15,0.1,0.05])
        rev    = prices * qtys * (1 - disc)
        profit = (prices - costs) * qtys * (1 - disc)

        df = pd.DataFrame({
            'transaction_id': [f'T{start+i+1:08d}' for i in range(size)],
            'date':          trans_dates,
            'year':          trans_dates.year,
            'month':         trans_dates.month,
            'quarter':       [f'Q{d.quarter}' for d in trans_dates],
            'store_id':      strs,
            'region':        regs,
            'category':      cats,
            'quantity':      qtys,
            'unit_price':    np.round(prices, 2),
            'unit_cost':     np.round(costs, 2),
            'discount_pct':  disc * 100,
            'revenue':       np.round(rev, 2),
            'profit':        np.round(profit, 2),
            'profit_margin': np.round((prices-costs)/prices*100, 1),
            'customer_id':   [f'C{np.random.randint(1,50001):05d}' for _ in range(size)],
        })
        all_chunks.append(df)
        print(f"    {min(end,n):,}/{n:,} records", end='\r')

    full_df = pd.concat(all_chunks, ignore_index=True)
    print(f"\n  Generated in {time.time()-t0:.1f}s")
    return full_df


# ── PySpark Analysis ─────────────────────────────────────────────────────────────
def run_spark_analysis(df):
    print("  Initialising Spark session…")
    spark = SparkSession.builder \
        .appName("RetailBigDataAnalytics") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Convert to Spark DataFrame
    sdf = spark.createDataFrame(df)
    sdf.createOrReplaceTempView("retail_transactions")

    results = {}

    # 1. Revenue by region & year
    print("    Running distributed aggregations…")
    reg_yr = spark.sql("""
        SELECT region, year,
               COUNT(*) AS transactions,
               ROUND(SUM(revenue), 2) AS total_revenue,
               ROUND(SUM(profit), 2)  AS total_profit,
               ROUND(AVG(profit_margin), 2) AS avg_margin
        FROM retail_transactions
        GROUP BY region, year
        ORDER BY year, total_revenue DESC
    """).toPandas()
    results['region_year'] = reg_yr

    # 2. Category performance
    cat_perf = spark.sql("""
        SELECT category,
               SUM(quantity) AS units_sold,
               ROUND(SUM(revenue), 2) AS revenue,
               ROUND(SUM(profit), 2)  AS profit,
               ROUND(AVG(profit_margin), 2) AS avg_margin,
               COUNT(DISTINCT customer_id) AS unique_customers
        FROM retail_transactions
        GROUP BY category
        ORDER BY revenue DESC
    """).toPandas()
    results['category'] = cat_perf

    # 3. Monthly trend with window functions
    monthly = spark.sql("""
        WITH monthly_agg AS (
            SELECT year, month,
                   ROUND(SUM(revenue), 2) AS revenue,
                   COUNT(*) AS transactions
            FROM retail_transactions
            GROUP BY year, month
        )
        SELECT year, month, revenue, transactions,
               ROUND(SUM(revenue) OVER (PARTITION BY year ORDER BY month), 2) AS ytd_revenue,
               ROUND(AVG(revenue) OVER (ORDER BY year, month
                     ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS moving_avg_3m
        FROM monthly_agg
        ORDER BY year, month
    """).toPandas()
    results['monthly'] = monthly

    # 4. Top stores
    top_stores = spark.sql("""
        SELECT store_id, region,
               ROUND(SUM(revenue), 2) AS revenue,
               COUNT(*) AS transactions,
               COUNT(DISTINCT customer_id) AS customers
        FROM retail_transactions
        GROUP BY store_id, region
        ORDER BY revenue DESC
        LIMIT 20
    """).toPandas()
    results['top_stores'] = top_stores

    spark.stop()
    return results


# ── Pandas Fallback Analysis ─────────────────────────────────────────────────────
def run_pandas_analysis(df):
    print("  (PySpark not available – running equivalent pandas analytics)")
    results = {}

    # 1. Region × year
    reg_yr = df.groupby(['region','year']).agg(
        transactions=('transaction_id','count'),
        total_revenue=('revenue','sum'),
        total_profit=('profit','sum'),
        avg_margin=('profit_margin','mean')
    ).reset_index().round(2)
    results['region_year'] = reg_yr

    # 2. Category
    cat_perf = df.groupby('category').agg(
        units_sold=('quantity','sum'),
        revenue=('revenue','sum'),
        profit=('profit','sum'),
        avg_margin=('profit_margin','mean'),
        unique_customers=('customer_id','nunique')
    ).reset_index().sort_values('revenue', ascending=False).round(2)
    results['category'] = cat_perf

    # 3. Monthly trend + window functions
    monthly = df.groupby(['year','month']).agg(
        revenue=('revenue','sum'),
        transactions=('transaction_id','count')
    ).reset_index().sort_values(['year','month'])
    monthly['ytd_revenue']    = monthly.groupby('year')['revenue'].cumsum()
    monthly['moving_avg_3m']  = monthly['revenue'].rolling(3, min_periods=1).mean()
    results['monthly'] = monthly.round(2)

    # 4. Top stores
    top_stores = df.groupby(['store_id','region']).agg(
        revenue=('revenue','sum'),
        transactions=('transaction_id','count'),
        customers=('customer_id','nunique')
    ).reset_index().sort_values('revenue', ascending=False).head(20).round(2)
    results['top_stores'] = top_stores

    return results


# ── Visualisations ───────────────────────────────────────────────────────────────
def visualize(results):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Big Data Retail Analytics – PySpark / Pandas at Scale',
                 fontsize=16, fontweight='bold')

    # 1. Regional revenue heatmap
    reg_yr = results['region_year'].copy()
    pivot  = reg_yr.pivot_table(index='region', columns='year', values='total_revenue', aggfunc='sum')
    im = axes[0,0].imshow(pivot.values / 1e6, cmap='YlOrRd', aspect='auto')
    axes[0,0].set_xticks(range(len(pivot.columns))); axes[0,0].set_xticklabels(pivot.columns)
    axes[0,0].set_yticks(range(len(pivot.index)));   axes[0,0].set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            axes[0,0].text(j, i, f'${pivot.values[i,j]/1e6:.1f}M',
                           ha='center', va='center', fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=axes[0,0], label='Revenue ($M)')
    axes[0,0].set_title('Regional Revenue Heatmap ($M)', fontweight='bold')

    # 2. Category performance
    cat = results['category'].sort_values('revenue', ascending=True).tail(8)
    bars = axes[0,1].barh(cat['category'], cat['revenue']/1e6, color='#1a73e8', alpha=0.8)
    ax2b = axes[0,1].twiny()
    ax2b.plot(cat['avg_margin'], cat['category'], 'D-', color='#ea4335', linewidth=2)
    ax2b.set_xlabel('Avg Profit Margin (%)', color='#ea4335')
    ax2b.tick_params(axis='x', colors='#ea4335')
    axes[0,1].set_xlabel('Revenue ($M)'); axes[0,1].set_title('Category Revenue & Margin', fontweight='bold')

    # 3. Monthly trend (latest 2 years)
    monthly = results['monthly'].copy()
    monthly['period'] = pd.to_datetime(
        monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
    )
    recent = monthly[monthly['year'] >= monthly['year'].max()-1]
    axes[1,0].fill_between(recent['period'], recent['revenue']/1e6, alpha=0.3, color='#1a73e8')
    axes[1,0].plot(recent['period'], recent['revenue']/1e6,   label='Monthly Revenue', color='#1a73e8', lw=2)
    axes[1,0].plot(recent['period'], recent['moving_avg_3m']/1e6, '--', label='3M Moving Avg', color='#ea4335', lw=2)
    axes[1,0].set_title('Monthly Revenue Trend (2 Years)', fontweight='bold')
    axes[1,0].set_ylabel('Revenue ($M)'); axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)

    # 4. Top stores
    top = results['top_stores'].head(10)
    colors = plt.cm.tab10(np.linspace(0,1,len(top)))
    bars2 = axes[1,1].bar(range(len(top)), top['revenue']/1e3, color=colors)
    axes[1,1].set_xticks(range(len(top)))
    axes[1,1].set_xticklabels(top['store_id'], rotation=45, ha='right', fontsize=8)
    axes[1,1].set_title('Top 10 Stores by Revenue', fontweight='bold')
    axes[1,1].set_ylabel('Revenue ($K)')
    for bar in bars2:
        axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                       f'${bar.get_height():.0f}K', ha='center', fontsize=7, rotation=45)

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'bigdata_analytics.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Visualisation saved → {out} ✓")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PROJECT 6 – Big Data Retail Analytics")
    print(f"  {'(PySpark)' if HAS_SPARK else '(Pandas at scale – PySpark not installed)'}")
    print("=" * 60)

    csv_path = os.path.join(DATA_DIR, 'large_retail.csv')
    if os.path.exists(csv_path):
        print("\n  Loading existing dataset…")
        df = pd.read_csv(csv_path, parse_dates=['date'])
    else:
        print("\n[1/3] Generating large retail dataset…")
        df = generate_large_retail_data(N_RECORDS)
        df.to_csv(csv_path, index=False)
        print(f"  Saved to {csv_path}")

    print(f"\n  Dataset: {len(df):,} rows × {df.shape[1]} cols")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total Revenue : ${df['revenue'].sum()/1e6:.2f}M")
    print(f"  Total Profit  : ${df['profit'].sum()/1e6:.2f}M")
    print(f"  Avg Margin    : {df['profit_margin'].mean():.1f}%")

    print("\n[2/3] Running distributed analytics…")
    if HAS_SPARK:
        results = run_spark_analysis(df)
    else:
        results = run_pandas_analysis(df)

    for name, rdf in results.items():
        rdf.to_csv(os.path.join(DATA_DIR, f'{name}_analytics.csv'), index=False)
        print(f"  ✓ {name}: {len(rdf)} rows")

    print("\n[3/3] Building visualisations…")
    visualize(results)

    top_cat = results['category'].iloc[0]
    print(f"\n{'='*60}")
    print(f"  Records Processed: {len(df):,}")
    print(f"  Total Revenue     : ${df['revenue'].sum()/1e6:.2f}M")
    print(f"  Top Category      : {top_cat['category']} (${top_cat['revenue']/1e6:.2f}M)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
