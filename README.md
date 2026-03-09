# 📊 Data Science Portfolio

A collection of **6 professional data science projects** demonstrating real-world skills across machine learning, data engineering, SQL analytics, business intelligence, and deep learning.

---

## 🗂️ Projects Overview

| # | Project | Domain | Key Technologies |
|---|---------|--------|-----------------|
| 1 | [Customer Churn Prediction](#1-customer-churn-prediction) | ML / Classification | Scikit-learn, XGBoost, RFM Analysis |
| 2 | [ERP Data Pipeline & ETL](#2-erp-data-pipeline--etl-automation) | Data Engineering | Python, SQL, SQLite, Pandas |
| 3 | [Retail Sales Dashboard](#3-retail-sales-performance-dashboard) | Business Intelligence | Matplotlib, KPI Analytics |
| 4 | [SQL Business Analytics](#4-sql-based-business-analytics) | Analytics / SQL | SQLite, Window Functions, Python |
| 5 | [Medical X-Ray Classification](#5-medical-x-ray-image-classification) | Deep Learning / CV | CNN, TensorFlow/PyTorch, Scikit-learn |
| 6 | [Big Data Retail Analytics](#6-big-data-retail-analytics) | Big Data | PySpark, Pandas, 500K+ records |

---

## 1. Customer Churn Prediction

**Business Problem:** Retail companies lose significant revenue through customer attrition. This project builds a churn prediction system to identify at-risk customers before they leave.

**Approach:** RFM feature engineering + multi-model ML pipeline  
**Best Model:** Random Forest — ROC-AUC: **0.97**, F1: **0.88**  
**Impact:** Identified 34% of customers as high-risk, enabling targeted retention campaigns.

📁 [`project1-customer-churn/`](project1-customer-churn/)

---

## 2. ERP Data Pipeline & ETL Automation

**Business Problem:** ERP systems contain critical operational data but require manual extraction and transformation for analytics.

**Approach:** End-to-end ETL pipeline — Extract → Transform → Load into SQLite data warehouse  
**Scale:** 1,000 customers, 500 products, 5,000 invoices, 20,000+ line items  
**Impact:** Automated pipeline replacing hours of manual data prep.

📁 [`project2-erp-pipeline/`](project2-erp-pipeline/)

---

## 3. Retail Sales Performance Dashboard

**Business Problem:** Managers need quick, intuitive insights into sales trends and regional performance.

**Approach:** Multi-panel KPI dashboard with trend analysis and profitability breakdown  
**Scale:** 50,000 sales transactions across 3 years  
**Key Insight:** Electronics drives highest revenue; November–December shows 30–50% seasonal lift.

📁 [`project3-sales-dashboard/`](project3-sales-dashboard/)

---

## 4. SQL-Based Business Analytics

**Business Problem:** Business intelligence teams need complex SQL analytics over relational data.

**Approach:** 5 advanced SQL queries — cohort analysis, RFM segmentation, window functions, KPIs  
**Scale:** 2,000 customers, 20,000 orders, 60,000 order items  
**Total Revenue Analyzed:** $396M across 4 years

📁 [`project4-sql-analytics/`](project4-sql-analytics/)

---

## 5. Medical X-Ray Image Classification

**Business Problem:** Deep learning can detect patterns in medical imaging that assist clinical diagnosis.

**Approach:** CNN architecture for 4-class X-ray classification (Normal, Pneumonia, COVID-19, TB)  
**Performance:** Accuracy **94.4%**, Macro ROC-AUC **0.997**  
**Note:** Uses synthetic X-ray data for demonstration; production use requires clinical datasets.

📁 [`project5-xray-cnn/`](project5-xray-cnn/)

---

## 6. Big Data Retail Analytics

**Business Problem:** Modern organizations process datasets too large for single-machine tools.

**Approach:** Distributed processing pipeline using PySpark (pandas at scale as fallback)  
**Scale:** **500,000 transaction records**, 5 years, 200 stores, 10 categories  
**Total Revenue Analyzed:** $497M

📁 [`project6-bigdata/`](project6-bigdata/)

---

## 🛠️ Tech Stack

```
Languages   : Python 3.10+, SQL
ML/DL       : Scikit-learn, XGBoost, TensorFlow/Keras, PyTorch
Data        : Pandas, NumPy, SQLite, Apache Spark
Visualization: Matplotlib, Seaborn
Engineering : Apache Airflow (pipeline orchestration concept)
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/ds-portfolio.git
cd ds-portfolio

# Install dependencies for any project
pip install -r project1-customer-churn/requirements.txt

# Run a project
python project1-customer-churn/src/churn_model.py
```

## 📈 Results Summary

| Project | Key Metric | Value |
|---------|-----------|-------|
| Churn Prediction | ROC-AUC | 0.97 |
| ERP Pipeline | Records Processed | 26,085 |
| Sales Dashboard | Revenue Analyzed | $17.1M |
| SQL Analytics | Revenue Analyzed | $396M |
| X-Ray CNN | Accuracy | 94.4% |
| Big Data | Records Processed | 500,000 |

---

*All projects use synthetic data generated to simulate realistic business scenarios.*
