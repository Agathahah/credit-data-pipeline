"""
End-to-End Data Engineering Pipeline for Credit Risk ML
========================================================
Run this to execute all 5 pipeline stages in sequence.

Stages:
  1. Ingestion  — CSV + World Bank API → PostgreSQL
  2. Transform  — Clean + quality check → PostgreSQL
  3. Enrich     — Join credit + macro data → PostgreSQL
  4. Features   — Build 18 features → PostgreSQL feature store
  5. Training   — XGBoost model → docs/ + models/
"""
import logging
import sys
import time
import os
import pandas as pd

os.makedirs("docs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("docs/pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

def run():
    total_start = time.time()
    logger.info("="*60)
    logger.info("CREDIT RISK DATA PIPELINE — START")
    logger.info("="*60)

    # Stage 1
    logger.info("\n[STAGE 1/5] Ingestion")
    from src.ingestion.ingest_credit import ingest_credit_csv
    from src.ingestion.ingest_macro import ingest_macro_indicators
    ingest_credit_csv()
    ingest_macro_indicators()

    # Stage 2
    logger.info("\n[STAGE 2/5] Transform + Quality Check")
    from src.transform.transform_credit import run_quality_checks, transform_credit
    from src.utils.db import get_engine, get_row_count
    engine = get_engine()
    df_raw = pd.read_sql("SELECT * FROM credit_raw", engine)
    run_quality_checks(df_raw, "raw")
    df_clean = transform_credit(df_raw)
    df_clean.to_sql("credit_cleaned", engine, if_exists="replace", index=False, chunksize=5000)
    logger.info(f"credit_cleaned: {get_row_count(engine, 'credit_cleaned')} rows")

    # Stage 3
    logger.info("\n[STAGE 3/5] Enrich with Macro Indicators")
    from src.transform.enrich_with_macro import enrich_with_macro
    enrich_with_macro()

    # Stage 4
    logger.info("\n[STAGE 4/5] Feature Engineering")
    from src.features.build_features import build_features
    df_enriched = pd.read_sql("SELECT * FROM credit_enriched", engine)
    df_feat = build_features(df_enriched)
    df_feat.to_sql("credit_features", engine, if_exists="replace", index=False, chunksize=5000)
    logger.info(f"credit_features: {get_row_count(engine, 'credit_features')} rows, "
                f"{len(df_feat.columns)} cols")

    # Stage 5
    logger.info("\n[STAGE 5/5] Model Training")
    from src.models.train import train
    model, metrics = train()

    elapsed = time.time() - total_start
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED")
    logger.info("="*60)
    logger.info(f"  Total time        : {elapsed:.1f}s")
    logger.info(f"  ROC-AUC           : {metrics['roc_auc']}")
    logger.info(f"  Avg Precision     : {metrics['average_precision']}")
    logger.info(f"  Features used     : {metrics['n_features']}")
    logger.info(f"  Training rows     : {metrics['n_train']}")
    logger.info(f"  DB tables created : credit_raw, macro_indicators,")
    logger.info(f"                      credit_cleaned, credit_enriched, credit_features")
    logger.info(f"  Artifacts saved   : docs/ and models/")
    logger.info("="*60)

if __name__ == "__main__":
    run()
