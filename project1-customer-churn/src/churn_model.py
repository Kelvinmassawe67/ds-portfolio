"""
Customer Churn Prediction Model
Trains and evaluates ML models to predict customer churn.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
import joblib
import os

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VIZ_DIR  = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
MDL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams.update({'figure.dpi': 150, 'figure.figsize': (10, 6)})


# ── Data Loading & Feature Engineering ─────────────────────────────────────────
def load_and_prepare_data():
    customers    = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'))
    rfm          = pd.read_csv(os.path.join(DATA_DIR, 'rfm_features.csv'))
    transactions = pd.read_csv(os.path.join(DATA_DIR, 'transactions.csv'))

    # Additional features from transactions
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    monthly_spend = (transactions.groupby('customer_id')
                     .resample('ME', on='transaction_date')['amount']
                     .sum().reset_index())
    spend_stability = (monthly_spend.groupby('customer_id')['amount']
                       .std().reset_index()
                       .rename(columns={'amount': 'spend_std'}))
    spend_stability['spend_std'] = spend_stability['spend_std'].fillna(0)

    # Merge all features
    df = customers.merge(rfm, on='customer_id', how='left') \
                  .merge(spend_stability, on='customer_id', how='left')
    df['spend_std'] = df['spend_std'].fillna(0)

    # Encode categoricals
    le = LabelEncoder()
    df['gender_enc']    = le.fit_transform(df['gender'])
    df['category_enc']  = le.fit_transform(df['preferred_category'].fillna('Unknown'))
    df['segment_enc']   = le.fit_transform(df['segment'].fillna('Unknown'))

    feature_cols = ['age', 'gender_enc', 'city_tier', 'loyalty_years', 'category_enc',
                    'recency', 'frequency', 'monetary', 'avg_order_value',
                    'total_items', 'discount_rate', 'unique_categories',
                    'R_score', 'F_score', 'M_score', 'RFM_score',
                    'segment_enc', 'spend_std']

    X = df[feature_cols].fillna(0)
    y = df['churned']
    return X, y, df, feature_cols


# ── Visualisations ─────────────────────────────────────────────────────────────
def plot_eda(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Customer Churn – Exploratory Data Analysis', fontsize=16, fontweight='bold')

    # 1. Churn distribution
    churn_counts = df['churned'].value_counts()
    axes[0,0].pie(churn_counts, labels=['Active','Churned'], autopct='%1.1f%%',
                  colors=['#2ecc71','#e74c3c'], startangle=90)
    axes[0,0].set_title('Churn Distribution')

    # 2. Recency vs churn
    df.boxplot(column='recency', by='churned', ax=axes[0,1])
    axes[0,1].set_title('Recency by Churn Status')
    axes[0,1].set_xlabel('Churned'); axes[0,1].set_ylabel('Days Since Last Purchase')
    plt.sca(axes[0,1]); plt.title('Recency by Churn Status')

    # 3. RFM score distribution
    df.groupby(['RFM_score','churned']).size().unstack(fill_value=0).plot(
        kind='bar', ax=axes[0,2], color=['#2ecc71','#e74c3c'])
    axes[0,2].set_title('RFM Score vs Churn')
    axes[0,2].set_xlabel('RFM Score'); axes[0,2].legend(['Active','Churned'])

    # 4. Customer segments
    segment_churn = df.groupby('segment')['churned'].mean().sort_values(ascending=False)
    segment_churn.plot(kind='barh', ax=axes[1,0], color='#3498db')
    axes[1,0].set_title('Churn Rate by Customer Segment')
    axes[1,0].set_xlabel('Churn Rate')

    # 5. Monetary vs Frequency scatter
    sample = df.sample(min(800, len(df)), random_state=42)
    scatter = axes[1,1].scatter(sample['frequency'], sample['monetary'],
                                 c=sample['churned'], cmap='RdYlGn_r', alpha=0.5, s=20)
    axes[1,1].set_xlabel('Purchase Frequency'); axes[1,1].set_ylabel('Total Spend ($)')
    axes[1,1].set_title('Frequency vs Monetary (coloured by churn)')
    plt.colorbar(scatter, ax=axes[1,1])

    # 6. Age distribution
    df[df['churned']==0]['age'].hist(ax=axes[1,2], alpha=0.6, bins=25,
                                     color='#2ecc71', label='Active')
    df[df['churned']==1]['age'].hist(ax=axes[1,2], alpha=0.6, bins=25,
                                     color='#e74c3c', label='Churned')
    axes[1,2].set_title('Age Distribution by Churn'); axes[1,2].legend()
    axes[1,2].set_xlabel('Age')

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'eda_overview.png'), bbox_inches='tight')
    plt.close()
    print("✓ EDA plot saved")


def plot_model_results(results, X_test, y_test, best_name, best_model, feature_cols):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')

    # 1. ROC curves
    for name, info in results.items():
        fpr, tpr, _ = roc_curve(y_test, info['proba'])
        axes[0,0].plot(fpr, tpr, label=f"{name} (AUC={info['roc_auc']:.3f})", lw=2)
    axes[0,0].plot([0,1],[0,1],'k--', lw=1)
    axes[0,0].set_xlabel('False Positive Rate'); axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curves'); axes[0,0].legend(fontsize=9)

    # 2. Model comparison bar chart
    names = list(results.keys())
    aucs  = [results[n]['roc_auc'] for n in names]
    f1s   = [results[n]['f1'] for n in names]
    x = np.arange(len(names)); w = 0.35
    axes[0,1].bar(x-w/2, aucs, w, label='ROC-AUC', color='#3498db')
    axes[0,1].bar(x+w/2, f1s,  w, label='F1 Score', color='#e67e22')
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(names, rotation=15)
    axes[0,1].set_ylim(0, 1); axes[0,1].set_title('Model Comparison')
    axes[0,1].legend()

    # 3. Confusion matrix for best model
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0],
                xticklabels=['Active','Churned'], yticklabels=['Active','Churned'])
    axes[1,0].set_title(f'Confusion Matrix – {best_name}')
    axes[1,0].set_ylabel('Actual'); axes[1,0].set_xlabel('Predicted')

    # 4. Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'named_steps'):
        est = best_model.named_steps.get('classifier') or list(best_model.named_steps.values())[-1]
        importances = est.feature_importances_ if hasattr(est, 'feature_importances_') else None
    else:
        importances = None

    if importances is not None:
        fi = pd.Series(importances, index=feature_cols).sort_values(ascending=True).tail(12)
        fi.plot(kind='barh', ax=axes[1,1], color='#9b59b6')
        axes[1,1].set_title(f'Feature Importance – {best_name}')
    else:
        axes[1,1].text(0.5, 0.5, 'N/A for this model', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'model_results.png'), bbox_inches='tight')
    plt.close()
    print("✓ Model results plot saved")


# ── Training & Evaluation ──────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, feature_cols):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
    }

    if HAS_XGB:
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos, eval_metric='logloss',
            use_label_encoder=False, random_state=42
        )

    results = {}
    best_auc, best_name, best_model = 0, None, None

    for name, model in models.items():
        print(f"  Training {name}…")
        Xtr = X_train_sc if name == 'Logistic Regression' else X_train
        Xte = X_test_sc  if name == 'Logistic Regression' else X_test

        model.fit(Xtr, y_train)
        proba  = model.predict_proba(Xte)[:, 1]
        y_pred = model.predict(Xte)
        auc    = roc_auc_score(y_test, proba)
        f1     = f1_score(y_test, y_pred)

        results[name] = {'model': model, 'proba': proba, 'roc_auc': auc, 'f1': f1}
        print(f"    ROC-AUC={auc:.4f}  F1={f1:.4f}")

        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model

    return results, best_name, best_model


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PROJECT 1 – Customer Churn Prediction")
    print("=" * 60)

    # Generate data if missing
    customers_path = os.path.join(DATA_DIR, 'customers.csv')
    if not os.path.exists(customers_path):
        print("Generating synthetic data…")
        import sys; sys.path.insert(0, os.path.dirname(__file__))
        from data_generator import generate_customer_data, compute_rfm
        os.makedirs(DATA_DIR, exist_ok=True)
        cust, trans = generate_customer_data(5000)
        rfm = compute_rfm(cust, trans)
        cust.to_csv(os.path.join(DATA_DIR, 'customers.csv'), index=False)
        trans.to_csv(os.path.join(DATA_DIR, 'transactions.csv'), index=False)
        rfm.to_csv(os.path.join(DATA_DIR, 'rfm_features.csv'), index=False)
        print("  Data generated ✓")

    print("\n[1/4] Loading & preparing data…")
    X, y, df, feature_cols = load_and_prepare_data()
    print(f"  Dataset: {X.shape[0]} customers × {X.shape[1]} features")
    print(f"  Churn rate: {y.mean():.1%}")

    print("\n[2/4] Generating EDA visualisations…")
    plot_eda(df)

    print("\n[3/4] Training models…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    results, best_name, best_model = train_models(X_train, X_test, y_train, y_test, feature_cols)

    print("\n[4/4] Saving artefacts…")
    plot_model_results(results, X_test, y_test, best_name, best_model, feature_cols)

    joblib.dump(best_model, os.path.join(MDL_DIR, 'churn_model.pkl'))
    print(f"  Best model ({best_name}) saved ✓")

    # Churn probability scores
    all_proba = best_model.predict_proba(X)[:, 1]
    scores = df[['customer_id']].copy()
    scores['churn_probability'] = all_proba
    scores['risk_tier'] = pd.cut(all_proba, bins=[0, .3, .6, 1.0],
                                  labels=['Low Risk', 'Medium Risk', 'High Risk'])
    scores.to_csv(os.path.join(DATA_DIR, 'churn_scores.csv'), index=False)
    print("  Churn scores saved ✓")

    print("\n" + "="*60)
    print(f"  Best Model : {best_name}")
    print(f"  ROC-AUC    : {results[best_name]['roc_auc']:.4f}")
    print(f"  F1 Score   : {results[best_name]['f1']:.4f}")
    high_risk = (scores['risk_tier'] == 'High Risk').sum()
    print(f"  High-Risk Customers: {high_risk} ({high_risk/len(scores):.1%})")
    print("="*60)


if __name__ == '__main__':
    main()
