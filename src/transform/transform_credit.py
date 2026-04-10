import pandas as pd
import numpy as np
import logging
from src.utils.db import get_engine, get_row_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quality_checks(df: pd.DataFrame, stage: str = "raw") -> dict:
    logger.info(f"\n{'='*50}")
    logger.info(f"DATA QUALITY REPORT — {stage.upper()}")
    logger.info(f"{'='*50}")
    report = {
        "stage"         : stage,
        "total_rows"    : len(df),
        "total_cols"    : len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_counts"   : df.isnull().sum().to_dict(),
        "null_pct"      : (df.isnull().mean() * 100).round(2).to_dict(),
    }
    if "age" in df.columns:
        report["invalid_age"]     = int(((df["age"] < 18) | (df["age"] > 100)).sum())
    if "revolving_util" in df.columns:
        report["extreme_util"]    = int((df["revolving_util"] > 1).sum())
    if "monthly_income" in df.columns:
        report["negative_income"] = int((df["monthly_income"] < 0).sum())
    for key, val in report.items():
        logger.info(f"  {key}: {val}")
    return report

def transform_credit(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running transformation...")
    initial = len(df)

    df = df.drop_duplicates()
    logger.info(f"  Dedup: removed {initial - len(df)} rows")

    df["monthly_income"]  = df["monthly_income"].fillna(df["monthly_income"].median())
    df["num_dependents"]  = df["num_dependents"].fillna(0).astype(int)

    before = len(df)
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    logger.info(f"  Invalid age: removed {before - len(df)} rows")

    for col in ["revolving_util", "debt_ratio", "monthly_income"]:
        if col in df.columns:
            cap = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap)

    for col in ["past_due_30_59", "past_due_60_89", "times_90_days_late"]:
        if col in df.columns:
            df[col] = df[col].clip(upper=90)

    # Assign year untuk join ke macro data
    # Dataset Give Me Some Credit adalah snapshot tahun 2010
    df["data_year"] = 2010

    logger.info(f"  Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM credit_raw", engine)
    run_quality_checks(df, "raw")
    df_clean = transform_credit(df)
    run_quality_checks(df_clean, "cleaned")
    df_clean.to_sql("credit_cleaned", engine, if_exists="replace", index=False, chunksize=5000)
    logger.info(f"[OK] credit_cleaned: {get_row_count(engine, 'credit_cleaned')} rows")
