"""Stage 5: train and evaluate the default model.

Evaluation protocol
-------------------
* 60/20/20 stratified split into train, validation and test.
* Early stopping and threshold selection use the **validation** split only.
  The test split is touched once, at the end. (The first version early-stopped
  on the test split, which makes the reported test score optimistic.)
* A logistic-regression baseline is trained on the same split so the XGBoost
  number has something to be compared with.
* Zero-variance columns are dropped and listed in the metrics file.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.features.build_features import BASE_FEATURES, ENGINEERED_FEATURES, MACRO_FEATURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET = "default_flag"
SEED = 42
XGB_PARAMS = dict(
    n_estimators=1000, max_depth=5, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, eval_metric="aucpr",
    early_stopping_rounds=50, random_state=SEED, n_jobs=-1, verbosity=0,
)


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def select_features(df: pd.DataFrame, include_macro: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Return numeric model features and the list of constant columns removed."""
    wanted = BASE_FEATURES + ENGINEERED_FEATURES + (MACRO_FEATURES if include_macro else [])
    X = df[[c for c in wanted if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant:
        logger.warning("Dropping zero-variance features: %s", constant)
    return X.drop(columns=constant), constant


def split_data(X: pd.DataFrame, y: pd.Series) -> Splits:
    """60/20/20 stratified split."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=SEED, stratify=y_tmp
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(ks_2samp(proba[y_true == 1], proba[y_true == 0]).statistic)


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold that maximises F1 on the data it is given (use validation data)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    top_decile = proba >= np.quantile(proba, 0.9)
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "ks": round(ks_statistic(y, proba), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "base_rate": round(float(y.mean()), 4),
        "threshold": round(threshold, 4),
        "precision_at_threshold": round(float(tp / max(tp + fp, 1)), 4),
        "recall_at_threshold": round(float(tp / max(tp + fn, 1)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "defaults_captured_in_top_10pct": round(float(y[top_decile].sum() / max(y.sum(), 1)), 4),
    }


def train_baseline(s: Splits) -> tuple[object, dict]:
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(s.X_train, s.y_train)
    thr = best_f1_threshold(np.asarray(s.y_val), model.predict_proba(s.X_val)[:, 1])
    return model, evaluate(s.y_test, model.predict_proba(s.X_test)[:, 1], thr)


def train_xgboost(s: Splits) -> tuple[xgb.XGBClassifier, dict]:
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(s.X_train, s.y_train, eval_set=[(s.X_val, s.y_val)], verbose=False)
    thr = best_f1_threshold(np.asarray(s.y_val), model.predict_proba(s.X_val)[:, 1])
    metrics = evaluate(s.y_test, model.predict_proba(s.X_test)[:, 1], thr)
    metrics["best_iteration"] = int(model.best_iteration)
    return model, metrics


def save_plots(model: xgb.XGBClassifier, s: Splits, threshold: float, out_dir: str) -> None:
    proba = model.predict_proba(s.X_test)[:, 1]
    fpr, tpr, _ = roc_curve(s.y_test, proba)
    prec, rec, _ = precision_recall_curve(s.y_test, proba)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, lw=2, label=f"XGBoost AUC={roc_auc_score(s.y_test, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    axes[0].legend()
    axes[1].plot(rec, prec, lw=2, label=f"AP={average_precision_score(s.y_test, proba):.3f}")
    axes[1].axhline(s.y_test.mean(), color="k", ls="--", lw=1, label="random")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=130)
    plt.close(fig)

    imp = pd.Series(model.feature_importances_, index=s.X_train.columns).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    imp.plot.barh(ax=ax)
    ax.set_title("XGBoost feature importance (gain)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=130)
    plt.close(fig)

    cm = confusion_matrix(s.y_test, (proba >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No default", "Default"],
           yticklabels=["No default", "Default"], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix @ threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=130)
    plt.close(fig)


def train_from_frame(df: pd.DataFrame, docs_dir: str = "docs", models_dir: str = "models",
                     make_plots: bool = True) -> tuple[xgb.XGBClassifier, dict]:
    """Train baseline + XGBoost from the feature table and write artifacts."""
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    X, constant = select_features(df)
    y = df[TARGET].astype(int)
    s = split_data(X, y)
    _, baseline = train_baseline(s)
    model, xgb_metrics = train_xgboost(s)

    metrics = {
        "dataset": "Give Me Some Credit (Kaggle), single snapshot, no application dates",
        "rows": {"train": len(s.X_train), "validation": len(s.X_val), "test": len(s.X_test)},
        "default_rate": round(float(y.mean()), 4),
        "n_features": int(X.shape[1]),
        "features_used": list(X.columns),
        "constant_features_dropped": constant,
        "macro_features_excluded": MACRO_FEATURES,
        "logistic_regression_baseline": baseline,
        "xgboost": xgb_metrics,
        "versions": {
            "python": platform.python_version(), "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
        },
    }
    with open(os.path.join(docs_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    model.save_model(os.path.join(models_dir, "xgb_credit_model.json"))
    if make_plots:
        save_plots(model, s, xgb_metrics["threshold"], docs_dir)
    logger.info("Baseline test: %s", baseline)
    logger.info("XGBoost test : %s", xgb_metrics)
    return model, metrics


def canonical_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by every column so the random split does not depend on database row order.

    ``SELECT *`` without ORDER BY returns rows in an engine-specific order (SQLite and
    PostgreSQL differ), and ``train_test_split`` with a fixed seed is only reproducible
    when its input order is fixed. Rows are unique after de-duplication, so this order is total.
    """
    cols = sorted(df.columns)
    return df.sort_values(cols, kind="mergesort", na_position="last").reset_index(drop=True)


def train(engine=None) -> tuple[xgb.XGBClassifier, dict]:
    """Load ``credit_features`` from the database and train."""
    from src.utils.db import get_engine

    engine = engine or get_engine()
    df = canonical_order(pd.read_sql("SELECT * FROM credit_features", engine))
    logger.info("Loaded %d rows from credit_features", len(df))
    return train_from_frame(df)


if __name__ == "__main__":
    train()
