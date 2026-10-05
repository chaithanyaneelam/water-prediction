# WATERNET - Water Quality Monitoring & Potability Prediction

Software-only, fully local system that predicts whether water is **potable**, estimates
**pH** when unmeasured, and flags **abnormal patterns** - from either a single form entry
or a CSV upload. Built on the real Kaggle "Water Potability" dataset (3,276 rows).

## Quick start

```bash
py -3.11 -m venv .venv                     # once
.venv\Scripts\activate                     # Windows (source .venv/bin/activate on Linux)
pip install -r requirements.txt

# 1) Put the Kaggle CSV at data/real/water_potability.csv, then train everything:
python -m backend.ml.train_all

# Irrigation module (separate real dataset, Telangana groundwater - file is included):
python -m backend.ml.load_irrigation_dataset
python -m backend.ml.train_irrigation

# 2) Start the site (Flask serves frontend + API - ONE command):
python run.py                              # -> http://127.0.0.1:5000

# 3) Optional demo simulator (replays REAL held-out test rows through the API):
python simulator/simulate.py -n 10 -i 2
```

Tests:

```bash
python -m pytest tests/ -q                 # 35 tests: API, validation, leakage, fault-injection, irrigation rules
python -m backend.ml.train_all --smoke     # end-to-end pipeline check on a tiny test-only fixture
```

Sample bulk-upload file: **`sample_bulk_upload.csv`** (project root) - upload it on the
Bulk Upload page to see the whole flow with 10 example readings.

## Pages

| Page | What it shows |
|---|---|
| Dashboard | totals, % potable, % anomalies, graphs 24-27, recent predictions (polls every 10 s) |
| Predict | 9-input form (pH optional) -> decision + probability (g29), predicted pH, anomaly + reason, radar (g28), out-of-limit params (g30), treatment text |
| Bulk Upload | CSV -> validated predictions table, summary charts, CSV download |
| History | searchable/filterable/paginated table, export CSV, readings-over-time chart (g25) |
| Model Comparison | graphs 7-23 in tabs: RF vs Decision Tree vs XGBoost, pH regressor vs baseline, Isolation Forest |
| Dataset Explorer | graphs 1-6 + summary statistics |
| Irrigation | USSL/RSC rule-engine verdict + ML cross-check on real Telangana groundwater data, USSL diagram, confusion matrix |

## API

`POST /api/predict`, `POST /api/predict/bulk`, `GET /api/readings`, `GET /api/stats`,
`GET /api/models/metrics`, `GET /api/dataset/summary`, `GET /api/plots/<name>`,
`GET /api/export/readings.csv`, `GET /api/health`.

## Honesty policy (important)

- The 20% test set is split **first** and never used for tuning; all preprocessing
  (imputation, outlier clipping, scaling, SMOTE) lives inside pipelines fitted on
  **training folds only**.
- **Potability is never a feature** - enforced by a leakage guard plus dedicated pytest tests.
- Expected classifier accuracy is **~65-70%**, because the 9 chemical parameters correlate
  only weakly with the label in this dataset. We do not chase inflated numbers.
- The pH regressor is used in the app **only if it clearly beats a predict-the-mean
  baseline**; otherwise the app falls back to median imputation (decision computed at
  training time and stored in `outputs/models/ph_regressor.json`).
- Guideline limits (WHO/BIS/EPA) are **display-only** and editable in
  `backend/ml/config.py`. They never create labels or features. Many potable-labelled
  samples exceed them (e.g. Solids averages ~22,000 mg/L).
- Accounts register with an email address; prediction reports are emailed via
  the Brevo API (TextBelt SMS remains supported for delivery when a channel is
  configured). Without keys, reports are kept in the internal outbox
  (see `.env.example`). No IoT, no Docker/cloud, no synthetic training data.
  The tiny CSV in `tests/fixtures/` is **test-only** and never used for training.

## Structure

```
backend/app/      Flask factory, blueprints, models, services, templates, static
backend/ml/       config, load_dataset, preprocess, trainers, plots, saved_models, outputs
data/real/        water_potability.csv (place the Kaggle file here)
tests/            pytest suite + test-only fixtures
docs/             architecture, model details, graphs, limitations
simulator/        optional demo replay of held-out test rows
run.py            one-command site
```

Database: SQLite (`waternet.db`) by default; switch via `DATABASE_URL` in `.env`
(see `.env.example` for a MySQL example). Tables: `water_quality_dataset`, `readings`,
`predictions`, `model_runs`.
