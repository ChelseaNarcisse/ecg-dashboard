# ECG Arrhythmia Early-Warning App

Machine-learning–based early detection of life-threatening cardiac arrhythmias from
wearable-style, two-lead ECG features. **MSDS 545 – Introduction to Computational Software
Engineering, Fall 2026** · Chelsea Narcisse

A trained scikit-learn classifier is serialized to `.pkl`, served by a **FastAPI** batch
endpoint, and consumed by a **Streamlit** dashboard where a researcher uploads a CSV of
heartbeats. The dashboard shows the risk distribution, the predicted AAMI classes, a
beat-by-beat "rhythm strip", per-patient summaries, and a sortable table of flagged high-risk
beats. *(Catalog mapping: Tier 2, "Sepsis Risk Prediction Dashboard" pattern.)*

Both Kaggle datasets from the proposal are used, each with its **own** models and endpoint (no
feature union, as agreed with the professor):

| Dataset (Kaggle) | Files used | Model | Endpoint |
|---|---|---|---|
| ECG Arrhythmia Classification (Sakib et al.): 34 measured features per beat | `MIT-BIH Arrhythmia Database.csv` (train/test); `INCART`, `MIT-BIH Supraventricular`, `Sudden Cardiac Death Holter` (external validation) | 5-class AAMI beat classifier (main model) | `POST /predict-batch` |
| ECG Heartbeat Categorization (Kachuee et al.): 187-sample waveforms | `mitbih_train.csv`, `mitbih_test.csv` | 5-class AAMI beat classifier from waveforms | `POST /predict-waveform-batch?task=arrhythmia` |
| same | `ptbdb_normal.csv`, `ptbdb_abnormal.csv` | Myocardial infarction (MI) detector | `POST /predict-waveform-batch?task=mi` |

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
│   │   ├── waveform.py          ★ waveform features, validation, prediction (zip-1 data)
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
├── notebooks/
│   └── ECG_Arrhythmia_ML_Pipeline.ipynb   the whole ML workflow, step by step (EDA → features → models → evaluation)
├── training/
│   ├── train_models.py          feature-based models: train, evaluate, external validation
│   ├── train_waveform_models.py waveform models: arrhythmia + myocardial infarction
│   └── make_report_figures.py   PNG figures for the written report
├── tests/                       pytest suite (TDD)
├── reports/figures/             report figures
├── data/raw/                    put the Kaggle CSVs here (git-ignored); waveform files in data/raw/waveform/
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
   257 Hz), balanced accuracy is 0.685 with the correct sample→ms conversion and 0.582
   without it.
4. **SVEB is the hard class.** Training records 201 and 203 are atrial-fibrillation patients
   whose irregular beats are labeled N, so the model learns that "early beat" can still be
   normal. The transparent rule (narrow QRS + RR < 85% of local rhythm) catches 78% of SVEB
   beats, which suggests a hybrid model as future work.
5. **Ventricular beats are what matter most for life-threatening rhythms, and VEB/F recall
   is high (0.94 / 0.92).**

**External validation (option 1).** The served model was tested, unchanged, on three other
databases in the same 34-column format. None of their patients were used in training. Q beats are
excluded, as in training.

| Database | Sampling rate | Beats evaluated | Accuracy | Balanced acc. | Recall N / SVEB / VEB / F |
|---|---|---|---|---|---|
| St Petersburg INCART | 257 Hz | 175,723 | 0.794 | 0.685 | 0.80 / 0.78 / 0.72 / 0.44 |
| MIT-BIH Supraventricular (SVDB) | 128 Hz | 184,342 | 0.478 | 0.467 | 0.47 / 0.27 / 0.86 / 0.26 |
| Sudden Cardiac Death Holter (SCDH) | 250 Hz | 251,933 | 0.897 | 0.394 | 0.92 / 0.03 / 0.63 / 0.00 |

* **Cleaning:** SCDH has 250,832 incomplete beats (76,217 empty rows plus beats with no second
  lead). They are dropped, and SVDB loses 7 beats with a non-positive RR interval.
* **Domain shift:** the sampling rates check out (all four databases give a plausible 69–79 bpm
  median heart rate). The difference is **amplitude**: the median R peak is 1.14 in MIT-BIH but
  0.33 in SVDB and 0.45 in SCDH (different recorders and lead placement). The model transfers to
  INCART but not to these two.
* **Tried and rejected:** per-patient amplitude normalization. It cut the main inter-patient
  balanced accuracy from 0.737 to 0.506 and only slightly helped SVDB and SCDH, so it is not used.
  This is a good "limitations and future work" point.

Figures for the report: `python training/make_report_figures.py` → `reports/figures/`
(figs 1–7 feature-based model, figs 8–11 waveform models).

### Step 4b: Waveform models (options 2 and 3) (`python training/train_waveform_models.py`)
The second Kaggle dataset stores each beat as **187 raw samples** (125 Hz, scaled 0–1, zero-padded).
It gets its own feature engineering (`backend/cardio_ml/waveform.py`), 28 features computed from
the signal:

| Group | Features | Why |
|---|---|---|
| Beat length | `beat_len_ms`, `heart_rate_bpm_est` | The segment runs R peak → ~1.2 × RR, so its un-padded length is a rhythm proxy |
| Amplitude statistics | mean, std, skew, kurtosis, min/max/range, energy | Overall beat shape |
| Shape | `qrs_width_ms`, max/min slope, `t_peak_ms`, `t_peak_amp`, `st_level` | Wide QRS (ventricular), T-wave and ST changes (MI) |
| Frequency | energy in 6 FFT bands (0–62.5 Hz), zero crossings | Sharp QRS vs slow ST/T content |

| Task | Train / test beats | Model | Accuracy | Balanced acc. | Macro F1 | AUC |
|---|---|---|---|---|---|---|
| Arrhythmia (N/SVEB/VEB/F/Q) | 87,554 / 21,892 (Kaggle split) | Logistic Regression | 0.787 | 0.840 | 0.579 | 0.954 |
| | | **Gradient Boosting** | **0.975** | **0.913** | **0.888** | 0.995 |
| Myocardial infarction | 11,641 / 2,911 (80/20 stratified) | Logistic Regression | 0.865 | 0.883 | 0.846 | 0.952 |
| | | **Gradient Boosting** | **0.988** | **0.985** | **0.985** | 0.998 |

* Waveform arrhythmia recall N / SVEB / VEB / F / Q = 0.98 / 0.80 / 0.96 / 0.85 / 0.99.
  MI recall = 0.99, healthy-control recall = 0.98.
* **Ablation (gradient boosting, balanced accuracy):** arrhythmia raw 0.907, engineered only
  0.899, both **0.913**; MI raw 0.975, engineered only 0.979, both **0.985**. The 28 engineered
  features alone nearly match the 187 raw samples, with about a seventh of the inputs.
* Top waveform features: T-peak amplitude, ST level and beat length (arrhythmia); signal
  roughness, QRS slope and high-frequency energy (MI). Averaged MI beats show a raised ST
  segment (fig 11), the classic sign of infarction.
* **Caveat:** these files have no patient ids, so the split is random at the beat level and the
  same patient can appear in train and test. Like the random split in Step 4, these scores are
  optimistic. The inter-patient result of the feature-based model is the honest estimate for a
  new patient.

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
| GET | `/waveform/models` | waveform models per task, with headline metrics |
| GET | `/waveform/metrics` | full waveform evaluation (confusion matrices, ablation, mean beats) |
| POST | `/predict-waveform-batch?task=arrhythmia\|mi&model_type=hgb` | score a batch of 187-sample waveforms |

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
labels) · **Waveform models** (pick arrhythmia or MI, see test metrics, mean beat per class and
ablation, upload a headerless waveform CSV → `/predict-waveform-batch` → risk histogram, predicted
classes, uploaded beats grouped by predicted class, flagged beats, accuracy) · **History** (SQLite trend). All predictions come from the API, never from a local
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
* `test_waveform.py`: waveform features (beat length, heart-rate estimate, FFT bands), validation
  (wrong length, NaN, unscaled, flat), `/predict-waveform-batch` with a mocked model for both
  tasks, CSV parsing, and the real waveform models on the demo files

87 tests in total.

GitHub Actions (`.github/workflows/ci.yml`) runs the tests on every push, then builds both
Docker images.

### Step 8: Docker and demo
See "Run with Docker Compose" below. Demo files in `frontend/sample_data/` are whole
recordings of **unseen** (DS2) patients: record 208 (frequent PVCs + fusion), 232 (frequent
SVEB), 213 (mixed), 100 (mostly normal), and a 16,156-beat multi-patient batch.
`frontend/sample_data/waveform/` holds a 3,000-beat arrhythmia test sample and a 1,500-beat PTB
test sample, both with labels.

---

## Run locally

```bash
# 0. get the data: download the Kaggle "ECG Arrhythmia Classification Dataset" and copy
#    "MIT-BIH Arrhythmia Database.csv"   -> data/raw/mitbih_features.csv
#    "INCART 2-lead Arrhythmia Database.csv"            -> data/raw/incart_features.csv  (optional)
#    "MIT-BIH Supraventricular Arrhythmia Database.csv" -> data/raw/svdb_features.csv    (optional)
#    "Sudden Cardiac Death Holter Database.csv"         -> data/raw/scdh_features.csv    (optional)
#    and from the "ECG Heartbeat Categorization" dataset:
#    mitbih_train.csv, mitbih_test.csv, ptbdb_normal.csv, ptbdb_abnormal.csv -> data/raw/waveform/
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# 1. (re)train - only needed if you change features/models or your scikit-learn version differs
#    (or open notebooks/ECG_Arrhythmia_ML_Pipeline.ipynb and Run All: same code, step by step)
python training/train_models.py
python training/train_waveform_models.py
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
| 422 "Each beat must have exactly 187 samples" | Waveform CSVs must be headerless, with 187 signal columns plus an optional label column |
| Want XGBoost like the reference repo | `pip install xgboost` and re-run training; it is added as `model_type=xgb` automatically |

## References
* Sakib, S. et al. *ECG Arrhythmia Classification Dataset* (Kaggle), features from PhysioNet databases.
* Kachuee, M., Fazeli, S. & Sarrafzadeh, M. (2018). ECG heartbeat classification: a deep transferable
  representation. *IEEE ICHI*. (Kaggle *ECG Heartbeat Categorization Dataset*, S. Fazeli.)
* Bousseljot, R., Kreiseler, D. & Schnabel, A. (1995). Nutzung der EKG-Signaldatenbank CARDIODAT der PTB
  über das Internet. *Biomedizinische Technik*, 40(S1). (PTB Diagnostic ECG Database.)
* Moody, G. & Mark, R. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE EMBS Magazine*, 20(3).
* de Chazal, P., O'Dwyer, M. & Reilly, R. (2004). Automatic classification of heartbeats using ECG morphology
  and heartbeat interval features. *IEEE Trans. Biomed. Eng.*, 51(7), 1196–1206.
* ANSI/AAMI EC57: Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.
* Reference architecture: [MSDS-545/food_access_app_multipleModels-](https://github.com/MSDS-545/food_access_app_multipleModels-)
