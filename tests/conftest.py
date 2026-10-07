"""Shared fixtures: a small synthetic dataset with the Give Me Some Credit schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_credit() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({
        "revolving_util": rng.uniform(0, 1.2, n),
        "age": rng.integers(21, 90, n),
        "past_due_30_59": rng.poisson(0.3, n),
        "debt_ratio": rng.uniform(0, 1.5, n),
        "monthly_income": rng.normal(6000, 2000, n).clip(500),
        "open_credit_lines": rng.integers(0, 20, n),
        "times_90_days_late": rng.poisson(0.1, n),
        "real_estate_loans": rng.integers(0, 4, n),
        "past_due_60_89": rng.poisson(0.1, n),
        "num_dependents": rng.integers(0, 5, n).astype(float),
    })
    logit = -3 + 2.5 * df["revolving_util"] + 0.8 * df["past_due_30_59"] + 1.2 * df["times_90_days_late"]
    df["default_flag"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[:49, "monthly_income"] = np.nan          # missing income
    df.loc[50:54, "past_due_30_59"] = 98            # sentinel code
    df.loc[55:57, "age"] = 0                        # invalid age
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)  # 10 duplicates
    return df


@pytest.fixture
def macro() -> pd.DataFrame:
    """Two-year macro table (test fixture values, not real statistics)."""
    return pd.DataFrame({
        "year": [2009, 2010],
        "inflation_rate": [1.0, 2.0],
        "lending_interest_rate": [3.0, 4.0],
        "gdp_growth_rate": [1.5, 2.5],
        "unemployment_rate": [8.0, 9.0],
    })
