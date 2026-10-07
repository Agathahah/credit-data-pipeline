"""Stage 4: deterministic feature engineering.

Every feature here is a row-wise formula, so it can be computed before the
train/test split without leaking information. Features that need statistics
learned from data (quantiles, medians) are not built here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.db import get_engine, get_row_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_FEATURES = [
    "revolving_util", "age", "past_due_30_59", "debt_ratio", "monthly_income",
    "open_credit_lines", "times_90_days_late", "real_estate_loans", "past_due_60_89",
    "num_dependents",
]
ENGINEERED_FEATURES = [
    "total_past_due", "delinquency_score", "high_util_flag", "very_high_util_flag",
    "monthly_debt", "disposable_income", "income_per_dependent", "age_util_interaction",
    "credit_diversity", "has_real_estate_loan", "log_monthly_income", "log_revolving_util",
    "log_monthly_debt", "income_missing_flag", "past_due_sentinel_flag",
]
# Kept in the feature table for analytics, excluded from the model because the
# dataset has one snapshot year (see enrich_with_macro.py).
MACRO_FEATURES = [
    "interest_adjusted_debt", "inflation_rate", "lending_interest_rate",
    "gdp_growth_rate", "unemployment_rate",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns. Missing inputs stay missing (XGBoost handles NaN)."""
    df = df.copy()
    past_due = df[["past_due_30_59", "past_due_60_89", "times_90_days_late"]]

    df["total_past_due"] = past_due.sum(axis=1, min_count=1)
    df["delinquency_score"] = (
        df["past_due_30_59"] * 1 + df["past_due_60_89"] * 2 + df["times_90_days_late"] * 3
    )
    df["high_util_flag"] = (df["revolving_util"] > 0.7).astype(int)
    df["very_high_util_flag"] = (df["revolving_util"] > 0.9).astype(int)

    # DebtRatio in this dataset is debt / monthly income, so monthly debt is
    # only meaningful when income is known.
    df["monthly_debt"] = df["debt_ratio"] * df["monthly_income"]
    df["disposable_income"] = df["monthly_income"] - df["monthly_debt"]
    df["income_per_dependent"] = df["monthly_income"] / (df["num_dependents"].fillna(0) + 1)

    df["age_group"] = pd.cut(
        df["age"], bins=[17, 30, 45, 60, 100],
        labels=["young_adult", "mid_career", "senior", "elderly"],
    ).astype(str)
    df["age_util_interaction"] = df["age"] * df["revolving_util"]

    df["credit_diversity"] = df["open_credit_lines"] + df["real_estate_loans"]
    df["has_real_estate_loan"] = (df["real_estate_loans"] > 0).astype(int)

    df["log_monthly_income"] = np.log1p(df["monthly_income"].clip(lower=0))
    df["log_revolving_util"] = np.log1p(df["revolving_util"].clip(lower=0))
    df["log_monthly_debt"] = np.log1p(df["monthly_debt"].clip(lower=0))

    if "lending_interest_rate" in df.columns:
        df["interest_adjusted_debt"] = df["monthly_debt"] * (1 + df["lending_interest_rate"] / 100)

    logger.info("Features built: %d engineered, %d total columns", len(ENGINEERED_FEATURES), df.shape[1])
    return df


if __name__ == "__main__":
    engine = get_engine()
    enriched = pd.read_sql("SELECT * FROM credit_enriched", engine)
    feats = build_features(enriched)
    feats.to_sql("credit_features", engine, if_exists="replace", index=False, chunksize=5000)
    logger.info("[OK] credit_features: %s rows", get_row_count(engine, "credit_features"))
