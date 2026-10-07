import json

import numpy as np

from src.features.build_features import build_features
from src.models.train import best_f1_threshold, select_features, train_from_frame
from src.transform.enrich_with_macro import enrich
from src.transform.transform_credit import transform_credit


def _features(raw_credit, macro):
    clean, _ = transform_credit(raw_credit)
    enriched, _ = enrich(clean, macro)
    return build_features(enriched)


def test_features_are_row_wise(raw_credit, macro):
    feats = _features(raw_credit, macro)
    row = feats.iloc[100]
    assert row["total_past_due"] == row["past_due_30_59"] + row["past_due_60_89"] + row["times_90_days_late"]
    assert np.isclose(row["monthly_debt"], row["debt_ratio"] * row["monthly_income"])


def test_macro_features_are_excluded_by_default(raw_credit, macro):
    X, _ = select_features(_features(raw_credit, macro))
    assert "lending_interest_rate" not in X.columns
    assert "interest_adjusted_debt" not in X.columns


def test_constant_columns_are_dropped_when_macro_is_requested(raw_credit, macro):
    X, dropped = select_features(_features(raw_credit, macro), include_macro=True)
    assert "lending_interest_rate" in dropped
    assert "lending_interest_rate" not in X.columns


def test_best_f1_threshold_is_inside_score_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.4, 0.9])
    assert 0.1 <= best_f1_threshold(y, p) <= 0.9


def test_training_writes_artifacts_and_beats_random(raw_credit, macro, tmp_path):
    feats = _features(raw_credit, macro)
    _, metrics = train_from_frame(feats, docs_dir=str(tmp_path), models_dir=str(tmp_path), make_plots=False)
    saved = json.loads((tmp_path / "model_metrics.json").read_text())
    assert saved["xgboost"]["roc_auc"] == metrics["xgboost"]["roc_auc"]
    assert metrics["xgboost"]["roc_auc"] > 0.6
    assert metrics["xgboost"]["pr_auc"] > metrics["default_rate"]
    assert (tmp_path / "xgb_credit_model.json").exists()


def test_split_does_not_depend_on_row_order():
    import numpy as np
    import pandas as pd

    from src.models.train import canonical_order, split_data

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.integers(0, 5, 400), "y": rng.integers(0, 2, 400)})
    df.loc[::17, "a"] = np.nan
    shuffled = df.sample(frac=1, random_state=1)
    s1 = split_data(*(lambda d: (d[["a", "b"]], d["y"]))(canonical_order(df)))
    s2 = split_data(*(lambda d: (d[["a", "b"]], d["y"]))(canonical_order(shuffled)))
    pd.testing.assert_frame_equal(s1.X_test.reset_index(drop=True), s2.X_test.reset_index(drop=True))
