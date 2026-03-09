"""
Customer Churn Prediction - Synthetic Data Generator
Generates realistic retail customer transaction data for churn analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

def generate_customer_data(n_customers=5000):
    """Generate synthetic retail customer transaction data."""
    
    customers = []
    transactions = []
    
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2022, 1, 1)
    
    product_categories = ['Electronics', 'Clothing', 'Groceries', 'Home & Garden',
                          'Sports', 'Books', 'Beauty', 'Toys', 'Automotive', 'Food & Beverage']
    
    for cust_id in range(1, n_customers + 1):
        # Customer profile
        age = np.random.randint(18, 75)
        gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
        city_tier = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        loyalty_years = np.random.exponential(3)
        
        # Base purchase frequency (purchases per year)
        base_freq = np.random.lognormal(mean=2.5, sigma=0.8)
        
        # Churn probability based on features
        churn_prob = 0.25
        if age > 60: churn_prob += 0.1
        if loyalty_years < 1: churn_prob += 0.15
        if city_tier == 3: churn_prob += 0.05
        if base_freq < 3: churn_prob += 0.1
        churn_prob = min(churn_prob, 0.85)
        
        is_churned = np.random.binomial(1, churn_prob)
        
        # Last activity date
        if is_churned:
            days_since_last = np.random.randint(90, 400)
        else:
            days_since_last = np.random.randint(1, 89)
        
        last_purchase = end_date - timedelta(days=days_since_last)
        cust_start = max(start_date, end_date - timedelta(days=int(loyalty_years * 365)))
        
        # Generate transactions
        n_transactions = max(1, int(base_freq * loyalty_years * (0.3 if is_churned else 1.0)))
        n_transactions = min(n_transactions, 200)
        
        preferred_category = np.random.choice(product_categories)
        
        customers.append({
            'customer_id': f'CUST{cust_id:05d}',
            'age': age,
            'gender': gender,
            'city_tier': city_tier,
            'loyalty_years': round(loyalty_years, 2),
            'preferred_category': preferred_category,
            'churned': is_churned
        })
        
        # Generate transaction records
        for t in range(n_transactions):
            trans_date = cust_start + timedelta(
                days=random.randint(0, max(1, (last_purchase - cust_start).days))
            )
            
            # Spending amount varies by category and customer profile
            base_amount = np.random.lognormal(mean=4.2, sigma=0.9)
            if preferred_category == 'Electronics':
                base_amount *= 2.5
            elif preferred_category == 'Groceries':
                base_amount *= 0.6
            
            category = np.random.choice(
                product_categories,
                p=[0.4 if c == preferred_category else 0.6/9 for c in product_categories]
            )
            
            transactions.append({
                'customer_id': f'CUST{cust_id:05d}',
                'transaction_date': trans_date.strftime('%Y-%m-%d'),
                'amount': round(base_amount, 2),
                'product_category': category,
                'items_purchased': np.random.randint(1, 10),
                'discount_used': np.random.choice([0, 1], p=[0.7, 0.3])
            })
    
    customers_df = pd.DataFrame(customers)
    transactions_df = pd.DataFrame(transactions)
    
    return customers_df, transactions_df


def compute_rfm(customers_df, transactions_df, snapshot_date='2024-12-31'):
    """Compute RFM (Recency, Frequency, Monetary) features."""
    
    snapshot = pd.Timestamp(snapshot_date)
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    
    rfm = transactions_df.groupby('customer_id').agg(
        recency=('transaction_date', lambda x: (snapshot - x.max()).days),
        frequency=('transaction_date', 'count'),
        monetary=('amount', 'sum'),
        avg_order_value=('amount', 'mean'),
        total_items=('items_purchased', 'sum'),
        discount_rate=('discount_used', 'mean'),
        unique_categories=('product_category', 'nunique')
    ).reset_index()
    
    # RFM scoring (1-5 scale)
    rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    
    # Customer segment based on RFM
    def segment_customer(row):
        if row['R_score'] >= 4 and row['F_score'] >= 4:
            return 'Champions'
        elif row['R_score'] >= 3 and row['F_score'] >= 3:
            return 'Loyal Customers'
        elif row['R_score'] >= 4:
            return 'Recent Customers'
        elif row['F_score'] >= 4:
            return 'At Risk'
        elif row['R_score'] <= 2 and row['F_score'] <= 2:
            return 'Lost Customers'
        else:
            return 'Potential Loyalists'
    
    rfm['segment'] = rfm.apply(segment_customer, axis=1)
    
    return rfm


if __name__ == '__main__':
    print("Generating synthetic retail customer data...")
    customers_df, transactions_df = generate_customer_data(5000)
    
    print(f"Generated {len(customers_df)} customers and {len(transactions_df)} transactions")
    
    rfm_df = compute_rfm(customers_df, transactions_df)
    
    customers_df.to_csv('../data/customers.csv', index=False)
    transactions_df.to_csv('../data/transactions.csv', index=False)
    rfm_df.to_csv('../data/rfm_features.csv', index=False)
    
    print("Data saved to ../data/")
    print(f"\nChurn rate: {customers_df['churned'].mean():.1%}")
    print(f"\nRFM Segments:\n{rfm_df['segment'].value_counts()}")
