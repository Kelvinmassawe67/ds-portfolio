"""
Retail Sales Performance Dashboard
Generates a comprehensive multi-panel sales analytics report with KPIs.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')
import os

random_seed = 42
np.random.seed(random_seed)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VIZ_DIR  = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


# ── Data Generation ─────────────────────────────────────────────────────────────
def generate_sales_data(n=50_000):
    print("  Generating synthetic retail sales data…")
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    regions   = ['North', 'South', 'East', 'West', 'International']
    categories = ['Electronics', 'Clothing', 'Groceries', 'Home & Garden',
                  'Sports', 'Beauty', 'Toys', 'Automotive']
    
    region_weights  = [0.25, 0.20, 0.22, 0.20, 0.13]
    cat_weights     = [0.18, 0.16, 0.22, 0.12, 0.10, 0.08, 0.07, 0.07]
    
    # Seasonal multipliers
    month_mult = {1:0.8,2:0.78,3:0.9,4:0.95,5:1.0,6:1.05,
                  7:1.02,8:1.0,9:0.98,10:1.05,11:1.3,12:1.5}
    
    rows = []
    for _ in range(n):
        date  = pd.Timestamp(np.random.choice(dates))
        cat   = np.random.choice(categories, p=cat_weights)
        region = np.random.choice(regions, p=region_weights)
        
        base_price = {'Electronics':350,'Clothing':65,'Groceries':45,'Home & Garden':120,
                      'Sports':85,'Beauty':55,'Toys':40,'Automotive':200}[cat]
        
        qty   = np.random.randint(1, 15)
        price = round(base_price * np.random.uniform(0.7, 1.4), 2)
        cost  = round(price * np.random.uniform(0.35, 0.65), 2)
        mult  = month_mult[date.month]
        revenue = round(price * qty * mult, 2)
        profit  = round((price - cost) * qty * mult, 2)
        
        rows.append({
            'date': date, 'year': date.year, 'month': date.month,
            'quarter': f'Q{date.quarter}', 'region': region,
            'category': cat, 'quantity': qty,
            'unit_price': price, 'unit_cost': cost,
            'revenue': revenue, 'profit': profit,
            'profit_margin': round((price-cost)/price*100, 1),
            'customer_id': f'C{np.random.randint(1,10001):05d}',
            'store_id':    f'S{np.random.randint(1,101):03d}',
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, 'sales_data.csv'), index=False)
    return df


# ── KPI Computation ─────────────────────────────────────────────────────────────
def compute_kpis(df):
    current_year = df['year'].max()
    prior_year   = current_year - 1
    
    curr = df[df['year'] == current_year]
    prev = df[df['year'] == prior_year]
    
    kpis = {
        'total_revenue':    curr['revenue'].sum(),
        'total_profit':     curr['profit'].sum(),
        'avg_margin':       curr['profit_margin'].mean(),
        'total_orders':     len(curr),
        'yoy_revenue':      (curr['revenue'].sum() - prev['revenue'].sum()) / prev['revenue'].sum() * 100,
        'yoy_profit':       (curr['profit'].sum() - prev['profit'].sum()) / prev['profit'].sum() * 100,
        'top_region':       curr.groupby('region')['revenue'].sum().idxmax(),
        'top_category':     curr.groupby('category')['revenue'].sum().idxmax(),
    }
    return kpis


# ── Dashboard ───────────────────────────────────────────────────────────────────
def build_dashboard(df):
    kpis = compute_kpis(df)
    
    # Color palette
    BLUE    = '#1a73e8'
    GREEN   = '#34a853'
    ORANGE  = '#fa7b17'
    RED     = '#ea4335'
    PURPLE  = '#9c27b0'
    DARK    = '#202124'
    LIGHT   = '#f8f9fa'
    
    fig = plt.figure(figsize=(24, 18), facecolor=LIGHT)
    fig.suptitle('Retail Sales Performance Dashboard', fontsize=22,
                 fontweight='bold', color=DARK, y=0.98)

    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.38)
    
    # ── KPI Cards (row 0) ────────────────────────────────────────────────────
    kpi_data = [
        ('Total Revenue',    f"${kpis['total_revenue']/1e6:.1f}M",   f"{kpis['yoy_revenue']:+.1f}% YoY",  BLUE),
        ('Total Profit',     f"${kpis['total_profit']/1e6:.1f}M",    f"{kpis['yoy_profit']:+.1f}% YoY",   GREEN),
        ('Avg Margin',       f"{kpis['avg_margin']:.1f}%",            'Gross Margin',                      ORANGE),
        ('Total Orders',     f"{kpis['total_orders']:,}",             f"Top: {kpis['top_region']}",        PURPLE),
    ]
    
    for i, (title, value, subtitle, color) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis('off')
        rect = FancyBboxPatch((0.02,0.05), 0.96, 0.90, boxstyle='round,pad=0.02',
                               facecolor='white', edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(0.5, 0.75, title,  ha='center', va='center', fontsize=11, color='#5f6368')
        ax.text(0.5, 0.45, value,  ha='center', va='center', fontsize=22, fontweight='bold', color=color)
        ax.text(0.5, 0.18, subtitle, ha='center', va='center', fontsize=9, color='#5f6368')
    
    # ── Monthly Revenue Trend (row 1, full width) ────────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    monthly = df.groupby(['year','month'])['revenue'].sum().reset_index()
    monthly['period'] = pd.to_datetime(monthly[['year','month']].assign(day=1))
    colors_yr = {2022:'#a8c4e0', 2023:'#5b9bd5', 2024:BLUE}
    for yr, grp in monthly.groupby('year'):
        ax1.plot(grp['period'], grp['revenue']/1e3, marker='o', markersize=4,
                 linewidth=2, color=colors_yr.get(yr, 'grey'), label=str(yr))
    ax1.fill_between(monthly[monthly['year']==2024]['period'],
                     monthly[monthly['year']==2024]['revenue']/1e3, alpha=0.15, color=BLUE)
    ax1.set_title('Monthly Revenue Trend (2022–2024)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Revenue ($K)'); ax1.legend(); ax1.grid(axis='y', alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
    
    # ── Regional Revenue (row 2, left) ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, :2])
    reg = df.groupby('region')['revenue'].sum().sort_values(ascending=True)
    bars = ax2.barh(reg.index, reg.values/1e6, color=[BLUE,GREEN,ORANGE,PURPLE,RED][:len(reg)])
    ax2.set_title('Revenue by Region ($M)', fontweight='bold', fontsize=13)
    ax2.set_xlabel('Revenue ($M)')
    for bar in bars:
        ax2.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                 f'${bar.get_width():.1f}M', va='center', fontsize=9)
    
    # ── Category Performance (row 2, right) ─────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 2:])
    cat_perf = df.groupby('category').agg(revenue=('revenue','sum'),
                                           margin=('profit_margin','mean')).reset_index()
    cat_perf = cat_perf.sort_values('revenue', ascending=False)
    x = np.arange(len(cat_perf))
    bars2 = ax3.bar(x, cat_perf['revenue']/1e6, color=BLUE, alpha=0.8, label='Revenue')
    ax3b = ax3.twinx()
    ax3b.plot(x, cat_perf['margin'], color=RED, marker='D', linewidth=2, label='Margin %')
    ax3.set_xticks(x); ax3.set_xticklabels(cat_perf['category'], rotation=35, ha='right', fontsize=9)
    ax3.set_title('Category Revenue & Margin', fontweight='bold', fontsize=13)
    ax3.set_ylabel('Revenue ($M)'); ax3b.set_ylabel('Margin (%)', color=RED)
    ax3b.tick_params(axis='y', colors=RED)
    
    # ── Quarterly Comparison (row 3, left) ───────────────────────────────────
    ax4 = fig.add_subplot(gs[3, :2])
    qtr = df.groupby(['year','quarter'])['revenue'].sum().unstack('year').fillna(0)
    qtr.plot(kind='bar', ax=ax4, color=[c for c in colors_yr.values()], width=0.6)
    ax4.set_title('Quarterly Revenue by Year', fontweight='bold', fontsize=13)
    ax4.set_xlabel('Quarter'); ax4.set_ylabel('Revenue ($)')
    ax4.tick_params(axis='x', rotation=0)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e6:.1f}M'))
    ax4.legend(title='Year')
    
    # ── Profit Margin Distribution (row 3, right) ────────────────────────────
    ax5 = fig.add_subplot(gs[3, 2:])
    for i, (region, grp) in enumerate(df.groupby('region')):
        ax5.hist(grp['profit_margin'], bins=30, alpha=0.5,
                 label=region, density=True)
    ax5.set_title('Profit Margin Distribution by Region', fontweight='bold', fontsize=13)
    ax5.set_xlabel('Profit Margin (%)'); ax5.set_ylabel('Density')
    ax5.legend(fontsize=8)
    
    out = os.path.join(VIZ_DIR, 'sales_dashboard.png')
    plt.savefig(out, bbox_inches='tight', facecolor=LIGHT, dpi=150)
    plt.close()
    print(f"  Dashboard saved → {out} ✓")
    return kpis


def main():
    print("=" * 60)
    print("PROJECT 3 – Retail Sales Performance Dashboard")
    print("=" * 60)
    
    csv_path = os.path.join(DATA_DIR, 'sales_data.csv')
    if os.path.exists(csv_path):
        print("  Loading existing sales data…")
        df = pd.read_csv(csv_path, parse_dates=['date'])
    else:
        df = generate_sales_data(50_000)
    
    print("\n  Computing KPIs…")
    kpis = build_dashboard(df)
    
    print(f"\n{'='*60}")
    print("  KEY PERFORMANCE INDICATORS")
    print(f"  Total Revenue  : ${kpis['total_revenue']/1e6:.2f}M")
    print(f"  Total Profit   : ${kpis['total_profit']/1e6:.2f}M")
    print(f"  Avg Margin     : {kpis['avg_margin']:.1f}%")
    print(f"  YoY Revenue    : {kpis['yoy_revenue']:+.1f}%")
    print(f"  Top Region     : {kpis['top_region']}")
    print(f"  Top Category   : {kpis['top_category']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
