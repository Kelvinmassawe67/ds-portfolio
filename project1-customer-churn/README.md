# Project 1: Customer Churn Prediction for Retail Supermarkets

## Business Problem
Retail companies depend heavily on customer retention. This project builds a machine learning system to predict which customers are at risk of churning, enabling proactive retention strategies.

## Approach
- **Feature Engineering**: RFM (Recency, Frequency, Monetary) metrics with customer segmentation
- **Models**: Logistic Regression, Random Forest, Gradient Boosting
- **Best Result**: Random Forest — ROC-AUC 0.9737, F1 0.8814

## Key Findings
- 33.5% churn rate in the dataset
- Recency is the strongest predictor of churn
- "Lost Customers" segment shows 80%+ churn probability
- 1,707 customers (34%) flagged as high-risk

## Project Structure
```
project1-customer-churn/
├── data/               # Generated customer & transaction data
├── src/
│   ├── data_generator.py   # Synthetic data generation + RFM features
│   └── churn_model.py      # EDA, model training & evaluation
├── models/             # Saved model artifacts
├── visualizations/     # EDA and model result plots
└── requirements.txt
```

## Run
```bash
pip install -r requirements.txt
python src/churn_model.py
```

## Technologies
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn
