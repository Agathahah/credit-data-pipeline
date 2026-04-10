import pandas as pd
import numpy as np
import logging
from src.utils.db import get_engine, get_row_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building feature store...")
    df = df.copy()

    # ── CREDIT BEHAVIOR ──────────────────────────────────────────
    df["total_past_due"] = (
        df.get("past_due_30_59", pd.Series(0, index=df.index)) +
        df.get("past_due_60_89", pd.Series(0, index=df.index)) +
        df.get("times_90_days_late", pd.Series(0, index=df.index))
    )
    df["delinquency_score"] = (
        df.get("past_due_30_59", pd.Series(0, index=df.index)) * 1 +
        df.get("past_due_60_89", pd.Series(0, index=df.index)) * 2 +
        df.get("times_90_days_late", pd.Series(0, index=df.index)) * 3
    )
    df["high_util_flag"]      = (df["revolving_util"] > 0.7).astype(int)
    df["very_high_util_flag"] = (df["revolving_util"] > 0.9).astype(int)

    # ── FINANCIAL CAPACITY ────────────────────────────────────────
    df["monthly_debt"]        = df["debt_ratio"] * df["monthly_income"]
    df["disposable_income"]   = df["monthly_income"] - df["monthly_debt"]
    df["income_per_dependent"] = df["monthly_income"] / (df["num_dependents"] + 1)
    df["low_income_flag"]     = (
        df["monthly_income"] < df["monthly_income"].quantile(0.25)
    ).astype(int)

    # ── DEMOGRAPHIC ───────────────────────────────────────────────
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 45, 60, 100],
        labels=["young_adult", "mid_career", "senior", "elderly"]
    ).astype(str)
    df["age_util_interaction"] = df["age"] * df["revolving_util"]

    # ── CREDIT PORTFOLIO ──────────────────────────────────────────
    df["credit_diversity"]     = (
        df.get("open_credit_lines", pd.Series(0, index=df.index)) +
        df.get("real_estate_loans", pd.Series(0, index=df.index))
    )
    df["has_real_estate_loan"] = (
        df.get("real_estate_loans", pd.Series(0, index=df.index)) > 0
    ).astype(int)

    # ── LOG TRANSFORMS ────────────────────────────────────────────
    df["log_monthly_income"]  = np.log1p(df["monthly_income"])
    df["log_revolving_util"]  = np.log1p(df["revolving_util"])
    df["log_monthly_debt"]    = np.log1p(df["monthly_debt"].clip(lower=0))

    # ── MACRO-CREDIT INTERACTIONS ─────────────────────────────────
    if "lending_interest_rate" in df.columns:
        df["interest_adjusted_debt"] = df["monthly_debt"] * (
            1 + df["lending_interest_rate"] / 100
        )
        rate_med = df["lending_interest_rate"].median()
        df["rate_util_risk"] = (
            (df["lending_interest_rate"] > rate_med) & (df["revolving_util"] > 0.5)
        ).astype(int)

    if "unemployment_rate" in df.columns:
        unemp_med = df["unemployment_rate"].median()
        df["macro_stress_flag"] = (df["unemployment_rate"] > unemp_med).astype(int)

    new_features = [
        "total_past_due", "delinquency_score", "high_util_flag", "very_high_util_flag",
        "monthly_debt", "disposable_income", "income_per_dependent", "low_income_flag",
        "age_group", "age_util_interaction", "credit_diversity", "has_real_estate_loan",
        "log_monthly_income", "log_revolving_util", "log_monthly_debt",
        "interest_adjusted_debt", "rate_util_risk", "macro_stress_flag"
    ]
    logger.info(f"New features built: {len(new_features)}")
    logger.info(f"Total columns    : {len(df.columns)}")
    return df

if __name__ == "__main__":
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM credit_enriched", engine)
    df_feat = build_features(df)
    df_feat.to_sql("credit_features", engine, if_exists="replace", index=False, chunksize=5000)
    logger.info(f"[OK] credit_features: {get_row_count(engine, 'credit_features')} rows, "
                f"{len(df_feat.columns)} cols")
    print(list(df_feat.columns))
