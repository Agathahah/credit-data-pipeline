import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, roc_curve
)
import xgboost as xgb
from src.utils.db import get_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "revolving_util", "age", "past_due_30_59", "debt_ratio",
    "monthly_income", "open_credit_lines", "times_90_days_late",
    "real_estate_loans", "past_due_60_89", "num_dependents",
    "total_past_due", "delinquency_score", "high_util_flag",
    "very_high_util_flag", "monthly_debt", "disposable_income",
    "income_per_dependent", "low_income_flag", "age_util_interaction",
    "credit_diversity", "has_real_estate_loan", "log_monthly_income",
    "log_revolving_util", "log_monthly_debt", "interest_adjusted_debt",
    "rate_util_risk", "macro_stress_flag", "inflation_rate",
    "lending_interest_rate", "gdp_growth_rate", "unemployment_rate"
]
TARGET = "default_flag"

def train():
    os.makedirs("docs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    engine = get_engine()
    df = pd.read_sql("SELECT * FROM credit_features", engine)
    logger.info(f"Loaded {len(df)} rows from feature store")

    available = [c for c in FEATURE_COLS if c in df.columns]
    logger.info(f"Features available: {len(available)}")

    X = df[available].select_dtypes(include=[np.number])
    y = df[TARGET]

    logger.info(f"Default rate : {y.mean():.2%}")
    logger.info(f"Class counts : {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)

    logger.info(f"\n{'='*50}")
    logger.info(f"ROC-AUC        : {auc:.4f}")
    logger.info(f"Avg Precision  : {ap:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Save metrics
    metrics = {
        "roc_auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(X_train.columns),
        "features_used": list(X_train.columns),
        "default_rate": round(float(y.mean()), 4)
    }
    with open("docs/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to docs/model_metrics.json")

    # Save model
    with open("models/xgb_credit_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to models/xgb_credit_model.pkl")

    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — Credit Default Model", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("docs/roc_curve.png", dpi=150)
    plt.close()

    # Plot 2: Feature Importance
    imp_df = pd.DataFrame({
        "feature": list(X_train.columns),
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True).tail(20)
    plt.figure(figsize=(10, 8))
    plt.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
    plt.xlabel("Importance Score", fontsize=12)
    plt.title("Top 20 Feature Importances — XGBoost", fontsize=14)
    plt.tight_layout()
    plt.savefig("docs/feature_importance.png", dpi=150)
    plt.close()

    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title("Confusion Matrix", fontsize=14)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("docs/confusion_matrix.png", dpi=150)
    plt.close()

    logger.info("All plots saved to docs/")
    logger.info(f"\n🎯 Final ROC-AUC: {auc:.4f}")
    return model, metrics

if __name__ == "__main__":
    train()
