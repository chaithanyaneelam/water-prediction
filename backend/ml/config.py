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

# ---------------------------------------------------------------- IRRIGATION
# Real dataset: Telangana groundwater quality (Kaggle,
# sadiyazubair/telangana-ground-water-classification, 1,090 samples).
IRRIGATION_XLSX = os.path.join(DATA_DIR, "irrigation_groundwater_telangana.xlsx")

# Chemistry features for the irrigation ML model (labels SAR/ussl_class/rsc_class
# are NEVER features - see tests/test_irrigation.py).
IRRIGATION_FEATURES = [
    "ph", "EC", "TDS", "CO3", "HCO3", "Cl", "F", "NO3",
    "SO4", "Na", "K", "Ca", "Mg", "TH",
]

# USSL (Richards 1954) salinity classes from EC in uS/cm.
USSL_EC_CLASSES = [
    ("C1", 0.0, 250.0, "Low salinity - suitable for most crops"),
    ("C2", 250.0, 750.0, "Medium salinity - fine with moderate leaching"),
    ("C3", 750.0, 2250.0, "High salinity - salt-tolerant crops + good drainage"),
    ("C4", 2250.0, float("inf"), "Very high salinity - generally unsuitable"),
]
# USSL sodium (alkali) classes from SAR.
USSL_SAR_CLASSES = [
    ("S1", 0.0, 10.0, "Low sodium - safe for nearly all soils"),
    ("S2", 10.0, 18.0, "Medium sodium - fine with leaching + organic matter"),
    ("S3", 18.0, 26.0, "High sodium - sodium hazard; gypsum + drainage needed"),
    ("S4", 26.0, float("inf"), "Very high sodium - generally unsuitable"),
]
# RSC classes (Richards): meq/L thresholds.
RSC_CLASSES = [
    ("P.S.", float("-inf"), 1.25, "Safe - residual sodium carbonate acceptable"),
    ("MR", 1.25, 2.5, "Marginal - watch for carbonate accumulation"),
    ("U.S.", 2.5, float("inf"), "Unsuitable - carbonate alkali hazard"),
]
IRRIGATION_USSL_MIN_CLASS_SIZE = 10  # rare USSL classes are excluded from ML (documented)
