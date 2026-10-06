"""Stage 3: attach World Bank macro indicators to each credit row.

Important limitation (found during the October 2026 audit): the Give Me Some
Credit dataset has no application date, so every row carries the same
``data_year``. The join therefore attaches one identical set of macro values
to every borrower. Those columns have zero variance and cannot help a model
separate defaulters from non-defaulters. The stage is kept because it shows
how a second source is integrated, and ``enrichment_report`` makes the
limitation measurable instead of hiding it.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy.engine import Engine

from src.utils.db import get_engine, get_row_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MACRO_COLS = ["inflation_rate", "lending_interest_rate", "gdp_growth_rate", "unemployment_rate"]


def enrich(df_credit: pd.DataFrame, df_macro: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Left-join macro indicators on ``data_year`` and report how informative they are."""
    macro = df_macro.rename(columns={"year": "data_year"})
    enriched = df_credit.merge(macro, on="data_year", how="left")
    present = [c for c in MACRO_COLS if c in enriched.columns]
    report = {
        "distinct_years_in_credit_data": int(df_credit["data_year"].nunique()),
        "rows_without_macro_match": int(enriched[present].isna().all(axis=1).sum()) if present else 0,
        "macro_columns_distinct_values": {c: int(enriched[c].nunique()) for c in present},
    }
    if report["distinct_years_in_credit_data"] <= 1:
        logger.warning(
            "Credit data has a single snapshot year; macro columns are constant "
            "and carry no signal for the model: %s",
            report["macro_columns_distinct_values"],
        )
    return enriched, report


def enrich_with_macro(engine: Engine | None = None) -> tuple[pd.DataFrame, dict]:
    """Read cleaned credit + macro tables, join them, write ``credit_enriched``."""
    engine = engine or get_engine()
    df_credit = pd.read_sql("SELECT * FROM credit_cleaned", engine)
    df_macro = pd.read_sql("SELECT * FROM macro_indicators", engine)
    enriched, report = enrich(df_credit, df_macro)
    enriched.to_sql("credit_enriched", engine, if_exists="replace", index=False, chunksize=5000)
    logger.info("[OK] credit_enriched: %s rows", get_row_count(engine, "credit_enriched"))
    return enriched, report


if __name__ == "__main__":
    enrich_with_macro()
