# 🏦 End-to-End Data Engineering Pipeline for Credit Risk ML

A production-style data engineering project that builds a full ML pipeline
for credit default prediction — integrating two data sources via a
multi-stage PostgreSQL pipeline.

---

## Architecture

```
[SOURCE 1] Kaggle CSV (150K rows)          [SOURCE 2] World Bank REST API
       │                                           │   (inflation, interest rate,
       │                                           │    GDP growth, unemployment)
       ▼                                           ▼
  credit_raw (PostgreSQL)              macro_indicators (PostgreSQL)
       │                                           │
       └──────────────────┬────────────────────────┘
                          ▼
                   credit_cleaned
                (transform + quality checks)
                          │
                          ▼
                   credit_enriched
                (JOIN credit + macro data)
                          │
                          ▼
                   credit_features
                (18 engineered features)
                          │
                          ▼
                 XGBoost Model
                (ROC-AUC: 0.8691)
```

## 🔧 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Database | PostgreSQL + SQLAlchemy |
| Data Processing | pandas, numpy |
| External API | World Bank REST API |
| ML Model | XGBoost, scikit-learn |
| Visualization | matplotlib, seaborn |

---

## Project Structure

```
credit-data-pipeline/
├── src/
│   ├── ingestion/
│   │   ├── ingest_credit.py        # SOURCE 1: CSV → PostgreSQL
│   │   └── ingest_macro.py         # SOURCE 2: World Bank API → PostgreSQL
│   ├── transform/
│   │   ├── transform_credit.py     # Clean and quality check
│   │   └── enrich_with_macro.py    # JOIN credit and macro data
│   ├── features/
│   │   └── build_features.py       # 18 engineered features
│   ├── models/
│   │   └── train.py                # XGBoost training and evaluation
│   └── utils/
│       └── db.py                   # DB connection helper
├── docs/
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── model_metrics.json
├── run_pipeline.py                 # Run all 5 stages end-to-end
└── requirements.txt
```

## 🚀 How to Run

```bash
# 1. Clone and setup
git clone https://github.com/Agathahah/credit-data-pipeline.git
cd credit-data-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Setup PostgreSQL
brew install postgresql@14
brew services start postgresql@14
psql postgres -c "CREATE USER dataengineer WITH PASSWORD 'de_password123';"
psql postgres -c "CREATE DATABASE credit_risk_db OWNER dataengineer;"

# 3. Create .env file
cat > .env << 'EOF'
DB_HOST=localhost
DB_PORT=5432
DB_NAME=credit_risk_db
DB_USER=dataengineer
DB_PASSWORD=de_password123
EOF

# 4. Download dataset to data/raw/cs-training.csv
# https://www.kaggle.com/c/GiveMeSomeCredit/data

# 5. Run full pipeline
python run_pipeline.py
```

---

## 📊 Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.8691 |
| Average Precision | 0.4132 |
| Training rows | 119,501 |
| Test rows | 29,876 |
| Features engineered | 31 |
| PostgreSQL tables | 5 |
| Data sources | 2 (CSV + REST API) |

---

## 📈 Evaluation Plots

![ROC Curve](docs/roc_curve.png)
![Feature Importance](docs/feature_importance.png)
![Confusion Matrix](docs/confusion_matrix.png)

---

## 🔍 Key Design Decisions

**Multi-source integration** — Credit data (CSV) is enriched with
macroeconomic indicators (World Bank API), simulating enterprise-grade
data integration from heterogeneous sources.

**Staged PostgreSQL pipeline** — Each transformation stage writes to a
separate table (credit_raw, credit_cleaned, credit_enriched, credit_features),
enabling auditability and easy debugging of each stage.

**Macro-credit interaction features** — Features like interest_adjusted_debt
and rate_util_risk are only possible because of the API integration,
demonstrating why multi-source pipelines matter for ML feature quality.

**Class imbalance handling** — scale_pos_weight in XGBoost handles the
6.7% default rate without oversampling, keeping training data distribution natural.
