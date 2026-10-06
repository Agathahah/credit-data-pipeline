"""End-to-end credit risk pipeline.

Stages:
  1. Ingestion  - CSV + World Bank API -> credit_raw, macro_indicators
  2. Transform  - deterministic cleaning -> credit_cleaned
  3. Enrich     - join macro indicators -> credit_enriched
  4. Features   - row-wise feature engineering -> credit_features
  5. Training   - logistic baseline + XGBoost -> docs/, models/

Every stage writes its row counts to docs/pipeline_report.json, so numbers in
the README can be checked against an artifact produced by the code.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import pandas as pd

from src.features.build_features import build_features
from src.ingestion.ingest_credit import ingest_credit_csv
from src.ingestion.ingest_macro import ingest_macro_indicators
from src.models.train import train
from src.transform.enrich_with_macro import enrich_with_macro
from src.transform.transform_credit import run_quality_checks, transform_credit
from src.utils.db import get_engine, get_row_count

os.makedirs("docs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("docs/pipeline.log")],
)
logger = logging.getLogger(__name__)


def run(csv_path: str = "data/raw/cs-training.csv") -> dict:
    start = time.time()
    engine = get_engine()
    report: dict = {"stages": {}}

    logger.info("[STAGE 1/5] Ingestion")
    ingest_credit_csv(csv_path, engine=engine)
    ingest_macro_indicators(engine=engine)
    report["stages"]["credit_raw"] = get_row_count(engine, "credit_raw")
    report["stages"]["macro_indicators"] = get_row_count(engine, "macro_indicators")

    logger.info("[STAGE 2/5] Transform + quality checks")
    raw = pd.read_sql("SELECT * FROM credit_raw", engine)
    report["quality_raw"] = run_quality_checks(raw, "raw")
    clean, report["transform_audit"] = transform_credit(raw)
    clean.to_sql("credit_cleaned", engine, if_exists="replace", index=False, chunksize=5000)
    report["stages"]["credit_cleaned"] = get_row_count(engine, "credit_cleaned")

    logger.info("[STAGE 3/5] Enrich with macro indicators")
    _, report["enrichment"] = enrich_with_macro(engine)
    report["stages"]["credit_enriched"] = get_row_count(engine, "credit_enriched")

    logger.info("[STAGE 4/5] Feature engineering")
    enriched = pd.read_sql("SELECT * FROM credit_enriched", engine)
    feats = build_features(enriched)
    feats.to_sql("credit_features", engine, if_exists="replace", index=False, chunksize=5000)
    report["stages"]["credit_features"] = get_row_count(engine, "credit_features")

    logger.info("[STAGE 5/5] Model training")
    _, metrics = train(engine)
    report["xgboost_test"] = {k: metrics["xgboost"][k] for k in ("roc_auc", "pr_auc", "ks")}
    report["baseline_test"] = {
        k: metrics["logistic_regression_baseline"][k] for k in ("roc_auc", "pr_auc", "ks")
    }
    report["seconds"] = round(time.time() - start, 1)

    with open("docs/pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("PIPELINE COMPLETED in %.1fs: %s", report["seconds"], report["stages"])
    return report


if __name__ == "__main__":
    run(os.getenv("CREDIT_CSV", "data/raw/cs-training.csv"))
