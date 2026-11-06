# ECG Arrhythmia Early-Warning App

Machine-learning–based early detection of life-threatening cardiac arrhythmias from
wearable-style, two-lead ECG features. **MSDS 545 – Introduction to Computational Software
Engineering, Fall 2026** · Chelsea Narcisse

A trained scikit-learn classifier is serialized to `.pkl`, served by a **FastAPI** batch
endpoint, and consumed by a **Streamlit** dashboard where a researcher uploads a CSV of
heartbeats. The dashboard shows the risk distribution, the predicted AAMI classes, a
beat-by-beat "rhythm strip", per-patient summaries, and a sortable table of flagged high-risk
beats. *(Catalog mapping: Tier 2, "Sepsis Risk Prediction Dashboard" pattern.)*

> Academic prototype for research and education. Not a medical device.

---

## Project structure

```text
.
├── backend/                     FastAPI inference service
│   ├── app.py                   endpoints (/predict-batch, /models, /history, …)
│   ├── db.py                    SQLite log of batch runs
│   ├── cardio_ml/               shared ML package (used by training AND the API)
│   │   ├── constants.py         AAMI classes, column names, sampling rates
│   │   ├── features.py          ★ feature engineering
│   │   ├── schema.py            data-validation (schema) checks
│   │   ├── rules.py             rule-based triage fallback
│   │   ├── predictor.py         validate → engineer → predict → risk score
│   │   └── wrappers.py          estimator wrapper (lets XGBoost use string labels)
│   ├── models/                  model_*.pkl + metrics.json (created by training)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                    Streamlit dashboard
│   ├── app.py                   layout only
│   ├── dashboard_utils.py       API client + Plotly charts (unit tested)
│   ├── sample_data/             demo upload files + EDA sample
│   ├── requirements.txt
│   └── Dockerfile
├── training/
│   ├── train_models.py          train, evaluate, serialize
│   └── make_report_figures.py   PNG figures for the written report
├── tests/                       pytest suite (TDD)
├── reports/figures/             report figures
├── data/raw/                    put the Kaggle CSVs here (git-ignored)
├── .github/workflows/ci.yml     GitHub Actions: tests on every push + Docker build
├── docker-compose.yml
└── pytest.ini
```

---

## Step-by-step build guide

The steps follow the SDLC phases in the proposal, updated with the professor's guidance:
focus on the ML and feature engineering, show the predictions in Streamlit through FastAPI,
use pytest, and **skip** the feature-union pipeline and the polymorphic data loader.

### Step 1: Planning: data and API contract
* **Dataset:** *ECG Arrhythmia Classification Dataset* (Sakib et al., Kaggle). This project
  uses `MIT-BIH Arrhythmia Database.csv` (100,689 beats, 44 patients, 360 Hz) for training
  and evaluation. `INCART 2-lead Arrhythmia Database.csv` (another hospital, 257 Hz) is used
  for external validation.
* **Targets:** AAMI classes **N** (normal), **SVEB** (supraventricular ectopic),
  **VEB** (ventricular ectopic), **F** (fusion), **Q** (unclassifiable).
* **API contract:** `POST /predict-batch` takes a list of beats (32 numeric features each,
  plus an optional `record` id). It returns one prediction per beat (class, class
  probabilities, risk score = 1 − P(N), risk tier) and a batch summary.

### Step 2: Data cleaning and EDA (`training/train_models.py`, dashboard tab 1)
* No missing values in MIT-BIH, and all RR intervals are positive.
* **4 redundant columns found:** `1_pre-RR` = `0_pre-RR`, `1_post-RR` = `0_post-RR`, and
  `*_qrs_morph0` = `*_qPeak` in 100% of beats. All 4 are dropped.
* **Severe class imbalance:** 89.5% of beats are N. This is why the models use class
  weighting and why **balanced accuracy** is the main metric (plain accuracy rewards
  always predicting "normal").
* **Q is excluded.** There are only 15 Q beats in the whole database (8 in training). With
  class balancing they produced hundreds of false Q alarms, so Q is left out, as in most
  inter-patient studies.

### Step 3: Feature engineering (`backend/cardio_ml/features.py`) ★
Raw features are in *samples* and are *absolute* values. The engineered features describe each
beat **relative to the patient's own rhythm**, which is how a cardiologist reads an ECG:

| Group | Features | Why |
|---|---|---|
| Unit conversion | `*_qrs_ms`, `*_qt_ms`, `pre_rr_ms` … | Makes databases recorded at different Hz comparable |
| Rhythm / prematurity | `rr_ratio`, `rr_norm_local` (RR ÷ mean of the previous 10 beats), `rr_norm_record`, `rr_diff_ms`, `heart_rate_bpm` | Ectopic beats arrive **early**, then a compensatory pause follows |
| Intervals | `qtc_bazett_ms`, `wide_qrs_flag` (QRS > 120 ms) | Ventricular beats have a **wide** QRS |
| Morphology | QRS amplitude, R/S and T/R ratios, QRS energy and slope, cross-lead R difference | Beat shape |

The local rhythm baseline is **causal**: it only uses previous beats, as a real-time wearable
would have to. The same function runs during training and inside the API, so there is no
training/serving skew.

### Step 4: Model training and evaluation (`python training/train_models.py`)
* **Models:** Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost if it
  is installed. The rule-based triage is the baseline.
* **Inter-patient split (de Chazal et al., 2004):** train on 22 patients (DS1), test on
  22 *different* patients (DS2). A random split leaks the patient's identity into the test
  set, so it is only reported to show how inflated it is.

**Results on unseen patients (DS2, 49,683 beats):**

| Model | Accuracy | Balanced acc. | Macro F1 | AUC (abnormal) | Recall N / SVEB / VEB / F |
|---|---|---|---|---|---|
| Rule-based triage | 0.890 | 0.500 | 0.438 | 0.855 | 0.95 / 0.78 / 0.27 / 0.00 |
| Logistic Regression | 0.725 | 0.527 | 0.458 | 0.894 | 0.74 / 0.23 / 0.89 / 0.26 |
| Random Forest | 0.925 | 0.527 | 0.516 | 0.920 | 0.97 / 0.08 / 0.92 / 0.14 |
| **Gradient Boosting (served)** | **0.847** | **0.737** | **0.523** | 0.896 | 0.87 / 0.22 / **0.94** / **0.92** |

**Key findings (for the Results & Discussion section):**
1. **Feature engineering works for tree models.** Gradient-boosting balanced accuracy goes
   from 0.562 (raw features only) to **0.737** (+17.5 points). Logistic regression does *not*
   benefit (0.704 → 0.527): the ratio features are non-linear and heavy-tailed.
2. **The evaluation design matters.** The same model scores **0.985** accuracy / 0.951
   balanced accuracy on a random split, but 0.847 / 0.737 on unseen patients. That gap is
   patient leakage.
3. **Sampling-rate conversion matters.** On the external INCART data (another hospital,
   257 Hz), balanced accuracy is 0.548 with the correct sample→ms conversion and 0.465
   without it.
4. **SVEB is the hard class.** Training records 201 and 203 are atrial-fibrillation patients
   whose irregular beats are labeled N, so the model learns that "early beat" can still be
   normal. The transparent rule (narrow QRS + RR < 85% of local rhythm) catches 78% of SVEB
   beats, which suggests a hybrid model as future work.
5. **Ventricular beats are what matter most for life-threatening rhythms, and VEB/F recall
   is high (0.94 / 0.92).**

Figures for the report: `python training/make_report_figures.py` → `reports/figures/`.

### Step 5: FastAPI backend (`backend/app.py`)
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + loaded models |
| GET | `/schema` | required input columns |
| GET | `/models` | models with headline metrics |
| GET | `/models/{name}/metrics` | confusion matrix, per-class metrics, feature importance |
| GET | `/metrics/summary` | dataset, ablation, leakage and external-validation results |
| POST | `/predict-batch?model_type=hgb` | **score a whole batch in one request** |
| GET | `/history` | recent batch runs (SQLite; aggregates only, no patient data) |

* Models are loaded **once** at startup.
* Pydantic validates types; `cardio_ml/schema.py` adds data-validation checks (NaN, inf,
  physiologic RR range, batch size). Bad input gets a 422 with every error listed.
* **Graceful degradation:** if no ML model is loaded, requests fall back to the rule-based
  triage.
* Performance: about 16,000 beats in about 0.2 s of server time (requirement: > 10,000
  without timeout).

### Step 6: Streamlit dashboard (`frontend/app.py`)
Tabs: **Data & EDA** · **Feature engineering** (scatter of prematurity vs QRS width, ablation) ·
**Model performance** (comparison, confusion matrix, feature importance, leakage table) ·
**Batch prediction** (upload → `/predict-batch` → risk histogram, predicted classes, rhythm
strip, per-patient table, flagged-beat table, CSV download, and accuracy if the file has
labels) · **History** (SQLite trend). All predictions come from the API, never from a local
model.

### Step 7: Testing (TDD with pytest)
```bash
pytest                 # all tests
pytest -m "not slow"   # fast: mocked model, no .pkl needed
```
* `test_features.py`: unit conversion, causal baseline, per-patient grouping, premature-beat
  detection, sampling-rate invariance of ratios
* `test_schema.py`: missing columns, NaN, text, inf, out-of-range RR, all errors reported together
* `test_api.py`: `/predict-batch` with a **mocked model and a small fixture CSV**, 400/422
  paths, history logging, fallback, and a **10,000-beat performance test**
* `test_predictor_and_rules.py`: risk tiers, probability alignment, rules, real `.pkl` integration
* `test_dashboard_utils.py`: payload never sends the true label, the API accepts the
  frontend's payload, all charts build

GitHub Actions (`.github/workflows/ci.yml`) runs the tests on every push, then builds both
Docker images.

### Step 8: Docker and demo
See "Run with Docker Compose" below. Demo files in `frontend/sample_data/` are whole
recordings of **unseen** (DS2) patients: record 208 (frequent PVCs + fusion), 232 (frequent
SVEB), 213 (mixed), 100 (mostly normal), and a 16,156-beat multi-patient batch.

---

## Run locally

```bash
# 0. get the data: download the Kaggle "ECG Arrhythmia Classification Dataset" and copy
#    "MIT-BIH Arrhythmia Database.csv"   -> data/raw/mitbih_features.csv
#    "INCART 2-lead Arrhythmia Database.csv" -> data/raw/incart_features.csv   (optional)
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# 1. (re)train - only needed if you change features/models or your scikit-learn version differs
python training/train_models.py
python training/make_report_figures.py

# 2. tests
pytest

# 3. backend  (terminal 1)
cd backend && uvicorn app:app --reload --port 8000      # docs: http://localhost:8000/docs

# 4. frontend (terminal 2)
cd frontend && streamlit run app.py                     # http://localhost:8501
```

## Run with Docker Compose

```bash
docker compose up --build        # frontend http://localhost:8501, API http://localhost:8000/docs
docker compose down
```

## Deploy online (Streamlit Community Cloud, free)

Without an `API_URL` setting, the dashboard runs the same FastAPI app from `backend/`
**inside** the Streamlit process. Every request still goes through FastAPI's routes,
Pydantic validation and the SQLite log, but no second server is needed, so the app can
be hosted as a single Streamlit app.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → *Deploy a public app from GitHub* → repository `ChelseaNarcisse/ecg-dashboard`,
   branch `main`, main file path **`frontend/app.py`**.
3. **Advanced settings** → Python version **3.13** (the models need scikit-learn 1.9.1).
4. **Deploy**. Packages come from `frontend/requirements.txt`.

Free apps sleep after a few days without visitors: open the link before a demo to wake it.
The prediction history is reset whenever the app restarts.

## Example API call

> Send **whole recordings in time order** (with a `record` column). Rhythm features compare
> each beat with the patient's previous beats, so a single beat sent on its own has no
> rhythm context, and an early beat can be scored as normal.

```bash
curl -s -X POST "http://localhost:8000/predict-batch?model_type=hgb" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{"sampling_rate_hz": 360, "records": [{"record": "208",
 "0_pre-RR": 240, "0_post-RR": 380, "0_pPeak": 0.02, "0_tPeak": 0.4, "0_rPeak": 1.6, "0_sPeak": -0.4,
 "0_qPeak": -0.1, "0_qrs_interval": 55, "0_pq_interval": 8, "0_qt_interval": 90, "0_st_interval": 25,
 "0_qrs_morph0": -0.1, "0_qrs_morph1": 0.2, "0_qrs_morph2": 0.9, "0_qrs_morph3": 1.3, "0_qrs_morph4": 0.6,
 "1_pre-RR": 240, "1_post-RR": 380, "1_pPeak": 0.01, "1_tPeak": 0.3, "1_rPeak": -0.2, "1_sPeak": -0.9,
 "1_qPeak": -0.5, "1_qrs_interval": 30, "1_pq_interval": 9, "1_qt_interval": 50, "1_st_interval": 15,
 "1_qrs_morph0": -0.5, "1_qrs_morph1": -0.6, "1_qrs_morph2": -0.7, "1_qrs_morph3": -0.6, "1_qrs_morph4": -0.5}]}
JSON
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `InconsistentVersionWarning` or an error loading `.pkl` | Your scikit-learn differs from 1.9.1. Re-run `python training/train_models.py` |
| Dashboard says it cannot reach the API | Start the backend first; set `API_URL` (Docker uses `http://backend:8000`) |
| 422 from `/predict-batch` | The response lists each problem (missing columns, NaN, RR out of range) |
| Large CSV upload rejected | Streamlit's upload limit is raised to 300 MB in Docker; the API maximum is 150,000 beats per batch |
| Want XGBoost like the reference repo | `pip install xgboost` and re-run training; it is added as `model_type=xgb` automatically |

## References
* Sakib, S. et al. *ECG Arrhythmia Classification Dataset* (Kaggle), features from PhysioNet databases.
* Moody, G. & Mark, R. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE EMBS Magazine*, 20(3).
* de Chazal, P., O'Dwyer, M. & Reilly, R. (2004). Automatic classification of heartbeats using ECG morphology
  and heartbeat interval features. *IEEE Trans. Biomed. Eng.*, 51(7), 1196–1206.
* ANSI/AAMI EC57: Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.
* Reference architecture: [MSDS-545/food_access_app_multipleModels-](https://github.com/MSDS-545/food_access_app_multipleModels-)
