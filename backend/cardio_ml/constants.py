"""Shared constants for the ECG arrhythmia project."""

# AAMI EC57 heartbeat classes, in the order the models output them.
AAMI_CLASSES = ["N", "SVEB", "VEB", "F", "Q"]

CLASS_DESCRIPTIONS = {
    "N": "Normal / bundle-branch-block beat",
    "SVEB": "Supraventricular ectopic beat (premature atrial/nodal)",
    "VEB": "Ventricular ectopic beat (PVC) - ventricular origin",
    "F": "Fusion of ventricular and normal beat",
    "Q": "Unclassifiable / paced beat",
}

# Beats of ventricular origin are the ones linked to life-threatening
# rhythms (ventricular tachycardia / fibrillation), so they get the
# strongest alert in the dashboard.
VENTRICULAR_CLASSES = {"VEB", "F"}

# The two ECG leads in the Sakib feature files are prefixed "0_" and "1_".
LEADS = ["0", "1"]

# Per-lead features exactly as they appear in the Kaggle CSV
# ("ECG Arrhythmia Classification Dataset", Sakib et al.).
PER_LEAD_FEATURES = [
    "pre-RR", "post-RR",
    "pPeak", "tPeak", "rPeak", "sPeak", "qPeak",
    "qrs_interval", "pq_interval", "qt_interval", "st_interval",
    "qrs_morph0", "qrs_morph1", "qrs_morph2", "qrs_morph3", "qrs_morph4",
]

RAW_FEATURE_COLUMNS = [f"{lead}_{feat}" for lead in LEADS for feat in PER_LEAD_FEATURES]

# Columns found to be exact duplicates of other columns during EDA
# (100% identical across all 100,689 MIT-BIH beats). They are dropped
# before modeling because they add no information.
REDUNDANT_COLUMNS = ["1_pre-RR", "1_post-RR", "0_qrs_morph0", "1_qrs_morph0"]

# Optional identifier columns that may be present in an uploaded CSV.
ID_COLUMNS = ["record"]
LABEL_COLUMN = "type"

# Sampling frequency of the source databases (Hz). Interval features are
# stored in *samples*, so they must be converted to milliseconds using the
# right rate before databases can be compared.
SAMPLING_RATES_HZ = {"MIT-BIH": 360, "INCART": 257, "MIT-BIH-SVDB": 128, "SCDH": 250}
DEFAULT_SAMPLING_RATE_HZ = 360

# Risk tiers on the "abnormal beat" probability (1 - P(N)).
RISK_TIERS = [(0.0, "low"), (0.3, "moderate"), (0.7, "high")]

# Hard limit for a single batch request (protects the API's memory).
MAX_BATCH_SIZE = 150_000
