import pandas as pd
import logging
from src.utils.db import get_engine, get_row_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def enrich_with_macro():
    logger.info("Joining credit data with macroeconomic indicators...")
    engine = get_engine()

    df_credit = pd.read_sql("SELECT * FROM credit_cleaned", engine)
    df_macro  = pd.read_sql("SELECT * FROM macro_indicators", engine)

    logger.info(f"Credit shape : {df_credit.shape}")
    logger.info(f"Macro shape  : {df_macro.shape}")
    logger.info(f"Macro cols   : {list(df_macro.columns)}")

    df_macro = df_macro.rename(columns={"year": "data_year"})

    # LEFT JOIN: setiap row kredit diperkaya dengan konteks makroekonomi tahun tersebut
    df_enriched = pd.merge(df_credit, df_macro, on="data_year", how="left")

    macro_cols = ["inflation_rate", "lending_interest_rate", "gdp_growth_rate", "unemployment_rate"]
    for col in macro_cols:
        if col in df_enriched.columns:
            df_enriched[col] = df_enriched[col].fillna(df_enriched[col].median())

    logger.info(f"Enriched shape : {df_enriched.shape}")

    sample = df_enriched[["age", "monthly_income", "default_flag",
                           "inflation_rate", "lending_interest_rate"]].head(5)
    logger.info(f"\nSample enriched data:\n{sample.to_string()}")

    df_enriched.to_sql("credit_enriched", engine, if_exists="replace", index=False, chunksize=5000)
    logger.info(f"[OK] credit_enriched: {get_row_count(engine, 'credit_enriched')} rows")
    return df_enriched

if __name__ == "__main__":
    enrich_with_macro()
