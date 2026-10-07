"""Stage 2: deterministic cleaning of the raw credit table.

Only rules that do not learn anything from the data live here (deduplication,
invalid ages, sentinel codes). Anything that is *estimated* from the data
(median imputation, percentile caps) belongs inside the training pipeline so
it is fitted on the training split only. Doing it here, before the split,
lets test-set statistics leak into training.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.db import get_engine, get_row_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PAST_DUE_COLS = ["past_due_30_59", "past_due_60_89", "times_90_days_late"]
# Give Me Some Credit stores 96 and 98 in the past-due counters. They are
# reporting codes, not real counts of late payments.
PAST_DUE_SENTINELS = (96, 98)
MIN_AGE, MAX_AGE = 18, 100
# Give Me Some Credit is a single snapshot with no application date. The
# competition data is commonly dated to around 2010; every row gets the same
# year, so any join on year attaches the same macro values to every borrower.
SNAPSHOT_YEAR = 2010


def run_quality_checks(df: pd.DataFrame, stage: str = "raw") -> dict:
    """Log and return simple data quality statistics for one stage."""
    report: dict = {
        "stage": stage,
        "total_rows": int(len(df)),
        "total_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_pct": {k: float(v) for k, v in (df.isnull().mean() * 100).round(2).items()},
    }
    if "age" in df.columns:
        report["invalid_age"] = int(((df["age"] < MIN_AGE) | (df["age"] > MAX_AGE)).sum())
    if "revolving_util" in df.columns:
        report["revolving_util_above_1"] = int((df["revolving_util"] > 1).sum())
    present = [c for c in PAST_DUE_COLS if c in df.columns]
    if present:
        report["past_due_sentinel_rows"] = int(df[present].isin(PAST_DUE_SENTINELS).any(axis=1).sum())
    logger.info("Quality report [%s]: %s", stage, report)
    return report


def transform_credit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply deterministic cleaning rules.

    Returns:
        The cleaned frame and a dict that counts rows removed or changed by
        each rule, so the README can quote numbers produced by the code.
    """
    audit: dict = {"rows_in": int(len(df))}
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    audit["removed_duplicates"] = int(before - len(df))

    before = len(df)
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    audit["removed_invalid_age"] = int(before - len(df))

    present = [c for c in PAST_DUE_COLS if c in df.columns]
    sentinel_mask = df[present].isin(PAST_DUE_SENTINELS).any(axis=1)
    audit["past_due_sentinel_rows_set_to_nan"] = int(sentinel_mask.sum())
    df["past_due_sentinel_flag"] = sentinel_mask.astype(int)
    for col in present:
        df[col] = df[col].where(~df[col].isin(PAST_DUE_SENTINELS), np.nan)

    df["income_missing_flag"] = df["monthly_income"].isna().astype(int)
    df["data_year"] = SNAPSHOT_YEAR

    audit["rows_out"] = int(len(df))
    logger.info("Transform audit: %s", audit)
    return df.reset_index(drop=True), audit


if __name__ == "__main__":
    engine = get_engine()
    raw = pd.read_sql("SELECT * FROM credit_raw", engine)
    run_quality_checks(raw, "raw")
    clean, _ = transform_credit(raw)
    run_quality_checks(clean, "cleaned")
    clean.to_sql("credit_cleaned", engine, if_exists="replace", index=False, chunksize=5000)
    logger.info("[OK] credit_cleaned: %s rows", get_row_count(engine, "credit_cleaned"))
