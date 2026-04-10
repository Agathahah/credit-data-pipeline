import requests
import pandas as pd
from functools import reduce
import logging
from src.utils.db import get_engine, get_row_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://api.worldbank.org/v2/country/US/indicator"

INDICATORS = {
    "FP.CPI.TOTL.ZG"  : "inflation_rate",
    "FR.INR.LEND"      : "lending_interest_rate",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_rate",
    "SL.UEM.TOTL.ZS"   : "unemployment_rate",
}

def fetch_indicator(code: str, name: str, start: int = 2000, end: int = 2014) -> pd.DataFrame:
    url = f"{BASE_URL}/{code}"
    params = {"format": "json", "date": f"{start}:{end}", "per_page": 100}
    logger.info(f"  Fetching {name} from World Bank API...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if len(data) < 2 or not data[1]:
        logger.warning(f"  No data for {name}")
        return pd.DataFrame()
    records = [
        {"year": int(item["date"]), name: float(item["value"])}
        for item in data[1] if item.get("value") is not None
    ]
    df = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    logger.info(f"  Got {len(df)} records for {name}")
    return df

def ingest_macro_indicators():
    logger.info("[SOURCE 2] Fetching macroeconomic indicators from World Bank API")
    dfs = [fetch_indicator(code, name) for code, name in INDICATORS.items() if True]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        raise ValueError("No macro data fetched. Check internet connection.")
    macro_df = reduce(lambda l, r: pd.merge(l, r, on="year", how="outer"), dfs)
    macro_df = macro_df.sort_values("year").reset_index(drop=True)
    macro_df = macro_df.ffill().bfill()
    logger.info(f"Macro data shape: {macro_df.shape}")
    logger.info(f"\n{macro_df.to_string()}")
    engine = get_engine()
    macro_df.to_sql("macro_indicators", engine, if_exists="replace", index=False)
    logger.info(f"[OK] macro_indicators: {get_row_count(engine, 'macro_indicators')} rows loaded")
    return macro_df

if __name__ == "__main__":
    ingest_macro_indicators()
