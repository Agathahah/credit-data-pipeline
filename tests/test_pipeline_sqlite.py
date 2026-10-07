"""End-to-end run of all five stages against SQLite (no PostgreSQL needed)."""

import json
import os

import run_pipeline


def test_pipeline_end_to_end(raw_credit, macro, tmp_path, monkeypatch):
    csv = tmp_path / "cs-training.csv"
    raw_credit.rename(columns={"default_flag": "SeriousDlqin2yrs"}).to_csv(csv)
    macro_csv = tmp_path / "macro.csv"
    macro.to_csv(macro_csv, index=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'pipeline.db'}")
    monkeypatch.setenv("MACRO_CSV", str(macro_csv))
    monkeypatch.chdir(tmp_path)
    os.makedirs("docs", exist_ok=True)

    report = run_pipeline.run(str(csv))

    assert report["stages"]["credit_raw"] == len(raw_credit)
    assert report["stages"]["credit_features"] == len(raw_credit) - 13
    assert report["enrichment"]["distinct_years_in_credit_data"] == 1
    assert json.loads((tmp_path / "docs" / "pipeline_report.json").read_text())["stages"]
