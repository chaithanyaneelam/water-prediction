"""Shared ML/backend configuration.

Everything that could change lives here in one place:
- feature columns (order matters for models and the predict form)
- paths to data, saved models, outputs
- guideline limits (WHO/BIS/EPA typical values) -- editable, guidance only

IMPORTANT: guideline limits are used ONLY for display/guidance charts.
They must NEVER be used to create labels or features (see docs/limitations).
"""
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------- paths
DATA_DIR = os.path.join(BASE_DIR, "data", "real")
DATASET_CSV = os.path.join(DATA_DIR, "water_potability.csv")
DB_PATH = os.path.join(BASE_DIR, "waternet.db")
DB_URL_DEFAULT = "sqlite:///" + DB_PATH
SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ---------------------------------------------------------------- columns
# Exact column names in the Kaggle water_potability.csv (verified at import time
# by load_dataset.py, which maps case-insensitively and reports mismatches).
LABEL_COL = "Potability"
FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
# pH regressor inputs: all features EXCEPT ph itself (and never Potability).
PH_PREDICTOR_COLS = [c for c in FEATURE_COLS if c != "ph"]

# Physical plausibility bounds used for validation and anomaly rule checks.
PHYSICAL_RANGES = {
    "ph": (0.0, 14.0),
    "Hardness": (0.0, None),
    "Solids": (0.0, None),
    "Chloramines": (0.0, None),
    "Sulfate": (0.0, None),
    "Conductivity": (0.0, None),
    "Organic_carbon": (0.0, None),
    "Trihalomethanes": (0.0, None),
    "Turbidity": (0.0, None),
}

# Human-friendly units for UI/labels.
UNITS = {
    "ph": "pH", "Hardness": "mg/L", "Solids": "mg/L", "Chloramines": "mg/L",
    "Sulfate": "mg/L", "Conductivity": "uS/cm", "Organic_carbon": "mg/L",
    "Trihalomethanes": "ug/L", "Turbidity": "NTU",
}

# ---------------------------------------------------------------- guideline limits
# Typical WHO / BIS / EPA guidance values. EDITABLE. Guidance/display only.
# Many rows in the Kaggle dataset exceed these limits yet are labelled potable
# (e.g. Solids averages ~22,000 mg/L), so limits must never create labels.
GUIDELINE_LIMITS = {
    "ph":              {"min": 6.5, "max": 8.5, "note": "WHO: 6.5-8.5 acceptable range"},
    "Turbidity":       {"max": 5.0,  "note": "WHO < 5 NTU (ideally < 1)"},
    "Solids":          {"max": 500.0, "note": "BIS desirable TDS < 500 mg/L"},
    "Hardness":        {"max": 200.0, "note": "BIS desirable < 200 mg/L (300 permissible)"},
    "Chloramines":     {"max": 4.0,  "note": "EPA MRDL < 4 mg/L"},
    "Sulfate":         {"max": 250.0, "note": "EPA/WHO < 250 mg/L"},
    "Trihalomethanes": {"max": 80.0, "note": "EPA MCL < 80 ug/L"},
    "Organic_carbon":  {"max": 2.0,  "note": "EPA treatment goal <= 2 mg/L"},
    "Conductivity":    {"max": 400.0, "note": "Typical drinking-water guidance < 400 uS/cm"},
}

SOURCE_FOOTNOTE = (
    "Guideline limits are typical WHO / BIS / EPA values and are shown for reference only. "
    "Many rows in the training dataset exceed them yet are labelled potable, so limits never "
    "create labels or model features."
)

# ---------------------------------------------------------------- random state
RANDOM_STATE = 42
