"""
Stage 6: Export credit_features from PostgreSQL to Google BigQuery
==================================================================
This stage adds a cloud analytics layer on top of the local PostgreSQL pipeline.
BigQuery enables scalable SQL analytics and potential integration with
Looker Studio, Vertex AI, and other GCP services.
"""

import os
import logging
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
from src.utils.db import get_engine

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_bq_client() -> bigquery.Client:
    credentials_path = os.getenv("GCP_CREDENTIALS_PATH")
    project_id = os.getenv("GCP_PROJECT_ID")

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    client = bigquery.Client(credentials=credentials, project=project_id)
    logger.info(f"BigQuery client initialized for project: {project_id}")
    return client

def create_dataset_if_not_exists(client: bigquery.Client, dataset_id: str):
    project_id = os.getenv("GCP_PROJECT_ID")
    full_dataset_id = f"{project_id}.{dataset_id}"

    try:
        client.get_dataset(full_dataset_id)
        logger.info(f"Dataset {full_dataset_id} already exists")
    except Exception:
        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = "US"
        dataset.description = "Credit risk ML pipeline — feature store and analytics tables"
        client.create_dataset(dataset)
        logger.info(f"[OK] Dataset created: {full_dataset_id}")

def export_table_to_bigquery(
    client: bigquery.Client,
    df: pd.DataFrame,
    table_name: str,
    dataset_id: str,
    project_id: str
):
    full_table_id = f"{project_id}.{dataset_id}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True
    )

    logger.info(f"Uploading {len(df)} rows to {full_table_id}...")
    job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
    job.result()

    table = client.get_table(full_table_id)
    logger.info(f"[OK] {full_table_id}: {table.num_rows} rows, {len(table.schema)} columns")
    return table

def run_validation_query(client: bigquery.Client, dataset_id: str, project_id: str):
    """Run basic validation queries on BigQuery to confirm data integrity."""
    logger.info("Running validation queries on BigQuery...")

    queries = {
        "total_rows": f"""
            SELECT COUNT(*) as total_rows
            FROM `{project_id}.{dataset_id}.credit_features`
        """,
        "default_rate": f"""
            SELECT
                ROUND(AVG(default_flag) * 100, 2) as default_rate_pct,
                COUNT(*) as total_records,
                SUM(default_flag) as total_defaults
            FROM `{project_id}.{dataset_id}.credit_features`
        """,
        "feature_stats": f"""
            SELECT
                ROUND(AVG(revolving_util), 4) as avg_revolving_util,
                ROUND(AVG(monthly_income), 2) as avg_monthly_income,
                ROUND(AVG(delinquency_score), 4) as avg_delinquency_score,
                ROUND(AVG(interest_adjusted_debt), 2) as avg_interest_adjusted_debt
            FROM `{project_id}.{dataset_id}.credit_features`
        """
    }

    for query_name, query in queries.items():
        result = client.query(query).result()
        rows = list(result)
        logger.info(f"  [{query_name}]: {dict(rows[0])}")

def export_to_bigquery():
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET", "credit_risk")

    logger.info("[STAGE 6] Exporting PostgreSQL feature store to Google BigQuery")
    logger.info(f"  Project : {project_id}")
    logger.info(f"  Dataset : {dataset_id}")

    # 1. Read from PostgreSQL
    engine = get_engine()
    logger.info("Reading credit_features from PostgreSQL...")
    df = pd.read_sql("SELECT * FROM credit_features", engine)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from PostgreSQL")

    # 2. Clean column types for BigQuery compatibility
    # BigQuery does not accept object columns that contain mixed types
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    # 3. Initialize BigQuery client
    client = get_bq_client()

    # 4. Create dataset if it does not exist
    create_dataset_if_not_exists(client, dataset_id)

    # 5. Export main feature store table
    export_table_to_bigquery(client, df, "credit_features", dataset_id, project_id)

    # 6. Export a summary/aggregated table for analytics
    logger.info("Building summary table by age group...")
    summary_df = df.groupby("age_group").agg(
        total_borrowers=("default_flag", "count"),
        default_rate=("default_flag", "mean"),
        avg_monthly_income=("monthly_income", "mean"),
        avg_revolving_util=("revolving_util", "mean"),
        avg_delinquency_score=("delinquency_score", "mean"),
        avg_interest_adjusted_debt=("interest_adjusted_debt", "mean")
    ).reset_index()
    summary_df["default_rate"] = summary_df["default_rate"].round(4)
    summary_df["avg_monthly_income"] = summary_df["avg_monthly_income"].round(2)

    export_table_to_bigquery(client, summary_df, "credit_summary_by_age_group", dataset_id, project_id)

    # 7. Run validation queries
    run_validation_query(client, dataset_id, project_id)

    logger.info("\n" + "="*60)
    logger.info("[OK] BigQuery export complete")
    logger.info(f"  Tables created:")
    logger.info(f"  - {project_id}.{dataset_id}.credit_features")
    logger.info(f"  - {project_id}.{dataset_id}.credit_summary_by_age_group")
    logger.info(f"  View in console:")
    logger.info(f"  https://console.cloud.google.com/bigquery?project={project_id}")
    logger.info("="*60)

if __name__ == "__main__":
    export_to_bigquery()
