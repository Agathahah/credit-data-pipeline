import pandas as pd
import logging
from src.utils.db import get_engine, get_row_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLUMN_RENAME = {
    'SeriousDlqin2yrs': 'default_flag',
    'RevolvingUtilizationOfUnsecuredLines': 'revolving_util',
    'age': 'age',
    'NumberOfTime30-59DaysPastDueNotWorse': 'past_due_30_59',
    'DebtRatio': 'debt_ratio',
    'MonthlyIncome': 'monthly_income',
    'NumberOfOpenCreditLinesAndLoans': 'open_credit_lines',
    'NumberOfTimes90DaysLate': 'times_90_days_late',
    'NumberRealEstateLoansOrLines': 'real_estate_loans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'past_due_60_89',
    'NumberOfDependents': 'num_dependents'
}

def ingest_credit_csv(filepath: str = "data/raw/cs-training.csv"):
    logger.info(f"[SOURCE 1] Loading CSV: {filepath}")
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.rename(columns=COLUMN_RENAME)

    logger.info(f"Shape        : {df.shape}")
    logger.info(f"Columns      : {list(df.columns)}")
    logger.info(f"Null counts  :\n{df.isnull().sum()}")
    logger.info(f"Default rate : {df['default_flag'].mean():.2%}")

    engine = get_engine()
    df.to_sql('credit_raw', engine, if_exists='replace', index=False, chunksize=5000)
    logger.info(f"[OK] credit_raw: {get_row_count(engine, 'credit_raw')} rows loaded")
    return df

if __name__ == "__main__":
    ingest_credit_csv()
