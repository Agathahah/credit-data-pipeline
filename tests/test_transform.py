import numpy as np

from src.transform.enrich_with_macro import enrich
from src.transform.transform_credit import SNAPSHOT_YEAR, transform_credit


def test_transform_removes_duplicates_and_invalid_ages(raw_credit):
    clean, audit = transform_credit(raw_credit)
    assert audit["removed_duplicates"] == 10
    assert audit["removed_invalid_age"] == 3
    assert audit["rows_out"] == len(clean) == len(raw_credit) - 13
    assert clean["age"].between(18, 100).all()


def test_sentinel_codes_become_missing_and_are_flagged(raw_credit):
    clean, audit = transform_credit(raw_credit)
    assert not clean["past_due_30_59"].isin([96, 98]).any()
    assert audit["past_due_sentinel_rows_set_to_nan"] == 5
    assert clean["past_due_sentinel_flag"].sum() == 5


def test_transform_does_not_learn_statistics(raw_credit):
    """Imputation belongs in training; missing income must survive cleaning."""
    clean, _ = transform_credit(raw_credit)
    assert clean["monthly_income"].isna().sum() == clean["income_missing_flag"].sum() > 0


def test_single_snapshot_year_makes_macro_constant(raw_credit, macro):
    clean, _ = transform_credit(raw_credit)
    enriched, report = enrich(clean, macro)
    assert report["distinct_years_in_credit_data"] == 1
    assert set(report["macro_columns_distinct_values"].values()) == {1}
    assert (enriched["data_year"] == SNAPSHOT_YEAR).all()
    assert np.isclose(enriched["lending_interest_rate"], 4.0).all()
