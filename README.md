# Credit Data Pipeline

A five-stage ETL and modelling pipeline for credit default prediction: CSV and a REST API go into PostgreSQL, every stage writes its own table, and the last stage trains and evaluates a model with a baseline for comparison.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Status](https://img.shields.io/badge/status-learning%20project-lightgrey)

## What this project is (and is not)

It is a portfolio project that shows how raw data moves through staged, auditable tables into a model, and how to run the same code locally, in Docker and in a local Kubernetes cluster.

It is not a production credit system. The dataset is a public Kaggle snapshot with no application dates, the model is not served behind an API, and there is no monitoring of live traffic.

## Pipeline

```
data/raw/cs-training.csv ──► credit_raw ──► credit_cleaned ──► credit_enriched ──► credit_features ──► model + docs/
World Bank API ───────────► macro_indicators ─────┘
```

| Stage | Table | Rows (last run) | What happens |
|---|---|---:|---|
| 1. Ingestion | `credit_raw` | 150,000 | Kaggle CSV loaded as-is, columns renamed |
| 1. Ingestion | `macro_indicators` | 15 | US inflation, lending rate, GDP growth, unemployment 2000–2014 |
| 2. Transform | `credit_cleaned` | 149,377 | 609 duplicates and 14 invalid ages removed; 225 rows with past-due codes 96/98 set to missing and flagged |
| 3. Enrich | `credit_enriched` | 149,377 | Macro indicators joined on `data_year` |
| 4. Features | `credit_features` | 149,377 | 15 row-wise engineered features |
| 5. Training | `docs/`, `models/` | 89,625 / 29,876 / 29,876 | Train / validation / test, stratified |

Row counts come from `docs/pipeline_report.json`, which `run_pipeline.py` writes on every run.

## Results (test split, last run)

| Model | ROC-AUC | PR-AUC | KS | Defaults caught in top 10% of scores |
|---|---:|---:|---:|---:|
| Logistic regression (baseline) | 0.861 | 0.379 | 0.569 | 54.1% |
| XGBoost | **0.866** | **0.412** | **0.578** | **55.4%** |

Default rate is 6.7%, so a random ranking has PR-AUC ≈ 0.067. At the decision threshold chosen on the validation split (0.21), XGBoost flags 2,539 of 29,876 test borrowers with 39.9% precision and 50.6% recall. Full numbers, including confusion matrices and library versions, are in [`docs/model_metrics.json`](docs/model_metrics.json).

**Reading the result:** XGBoost beats the baseline mostly on PR-AUC (+0.033). ROC-AUC differs by only 0.005, which says most of the signal is captured by the original ten columns and that the gain from boosting is real but modest.

## Audit, October 2026: what changed and why

I re-ran this project from a clean environment and found problems in the first version. They are listed here because finding them was the most useful part of the project.

| # | Problem in v1 | Evidence | Fix |
|---|---|---|---|
| 1 | **Macro features carried no information.** The dataset has no dates, so every row got `data_year = 2010` and identical macro values. `interest_adjusted_debt` was `monthly_debt × 1.0325`, perfectly correlated (r = 1.0) with `monthly_debt`. The README claimed these features improved the model. | `enrichment` block in `docs/pipeline_report.json`: every macro column has 1 distinct value. Removing all 7 macro columns changed test ROC-AUC by 0.0005 (0.8685 → 0.8680). | Macro features are excluded from the model and the limitation is logged. The macro stage stays as an integration example. A dataset with application dates is needed for macro features to matter. |
| 2 | **Early stopping used the test set.** | `eval_set=[(X_test, y_test)]` with `early_stopping_rounds` | Separate validation split for early stopping and threshold choice. Test set is used once. |
| 3 | **Statistics learned before the split.** Median imputation and 99th-percentile caps were fitted on all rows. | `transform_credit.py` v1 | Cleaning is now rule-based only. Missing values stay missing (XGBoost handles them; the baseline imputes inside its own pipeline). |
| 4 | **No baseline.** A single AUC had nothing to be compared with. | — | Logistic regression trained on the same split. |
| 5 | **Docker and local results differed** (0.8691 vs 0.8690), explained as floating point. | `requirements_docker.txt` pinned XGBoost 2.0.3 and NumPy 1.26; local used XGBoost 3.2 and NumPy 2.4. | One `requirements.txt` for every environment. |
| 6 | **Kubernetes ran a batch job as a Deployment** with `restartPolicy: Always`, so the pipeline would restart forever. The Namespace was declared after objects that use it. Database password was committed in base64. | `k8s/` v1 | `Job` with `OnFailure`, namespace applied first, Secret created with `kubectl` (see `k8s/README.md`). |
| 7 | **Password hard-coded** in `Dockerfile` and `docker-compose.yml`. | — | Read from `.env`, which is git-ignored. Container runs as a non-root user. |
| 8 | **README numbers did not match the code** (e.g. "491 invalid ages", 149,391 cleaned rows). | Actual: 14 invalid ages, 149,377 rows. | README numbers are now copied from `docs/pipeline_report.json` and `docs/model_metrics.json`. |
| 9 | **No tests, no CI.** | — | 10 pytest tests on synthetic data, including an end-to-end run on SQLite; GitHub Actions runs lint, tests and a Docker build. |

## How to run

You need the Kaggle file `cs-training.csv` from [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) in `data/raw/`.

### 1. Tests only (no database, no Docker, about 5 seconds)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q          # 10 passed
ruff check src tests run_pipeline.py
```

### 2. Full pipeline with Docker Compose

```bash
cp .env.example .env         # edit DB_PASSWORD
docker compose up --build    # PostgreSQL + one pipeline run, then exits
cat docs/pipeline_report.json
```

### 3. Full pipeline against your own database

```bash
cp .env.example .env         # point DB_* at your PostgreSQL, or set DATABASE_URL
python run_pipeline.py
```

Behind a firewall that blocks `api.worldbank.org`, export the four indicators to a CSV (`year,inflation_rate,lending_interest_rate,gdp_growth_rate,unemployment_rate`) and set `MACRO_CSV=path/to/file.csv`.

### 4. Local Kubernetes

See [`k8s/README.md`](k8s/README.md).

## Project structure

```
src/
  ingestion/   ingest_credit.py, ingest_macro.py, export_to_bigquery.py (optional)
  transform/   transform_credit.py (rules only), enrich_with_macro.py
  features/    build_features.py (row-wise formulas)
  models/      train.py (split, baseline, XGBoost, metrics, plots)
  utils/       db.py (DATABASE_URL or DB_* variables)
tests/         synthetic fixtures, unit tests, end-to-end SQLite test
k8s/           namespace, postgres, job
docs/          model_metrics.json, pipeline_report.json, plots
```

## Limitations and next steps

- The dataset is a single snapshot. A dataset with application dates would allow a time-based split and meaningful macro features.
- Probabilities from XGBoost are not calibrated. Calibration is needed before scores are read as default probabilities.
- The model is trained but not served. A FastAPI scoring endpoint with input validation is the next step.
- The BigQuery export (`requirements-bigquery.txt`) needs a service-account key and is not covered by tests.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), Kaggle: 150,000 borrowers, 10 features, 6.7% serious delinquency within two years. Macro indicators: [World Bank Open Data](https://data.worldbank.org/).

---

Agatha Ulina Silalahi · [LinkedIn](https://www.linkedin.com/in/agatha-silalahi-722507215/) · [GitHub](https://github.com/Agathahah)
