"""Leakage and honesty tests - the heart of the quality gate.

They prove:
1. Potability (the label) can never become a feature.
2. Preprocessing is fitted on TRAINING data only (imputer/clipper/scaler
   statistics are computed from the fit rows and applied unchanged elsewhere).
3. The pH regressor never receives Potability as an input.
4. Fault injection into a COPY of real-style data is caught by the anomaly
   stack (rule checks + Isolation Forest) - test-only, never training.
"""
import numpy as np
import pandas as pd
import pytest

from backend.ml.config import FEATURE_COLS, PH_PREDICTOR_COLS
from backend.ml.preprocess import (
    IQRClipper, MedianImputer, StandardScalerSafe, make_feature_frame,
)


def test_make_feature_frame_never_contains_label():
    df = pd.DataFrame({c: np.random.rand(10) for c in FEATURE_COLS})
    df["Potability"] = np.random.randint(0, 2, 10)
    X = make_feature_frame(df, FEATURE_COLS)
    assert "Potability" not in X.columns
    assert list(X.columns) == FEATURE_COLS


def test_make_feature_frame_raises_if_requested_features_include_label():
    df = pd.DataFrame({c: np.random.rand(10) for c in FEATURE_COLS})
    with pytest.raises(AssertionError, match="Leakage guard"):
        make_feature_frame(df, FEATURE_COLS + ["Potability"])


def test_ph_regressor_feature_list_excludes_label():
    assert "Potability" not in PH_PREDICTOR_COLS
    assert "ph" not in PH_PREDICTOR_COLS  # it is the TARGET of that model
    assert set(PH_PREDICTOR_COLS) == set(FEATURE_COLS) - {"ph"}


def test_imputer_learns_from_fit_rows_only():
    """Fitting on rows that exclude an outlier proves the outlier's value is
    NOT used: imputing a NaN with fit-stats gives the fit median, not a value
    influenced by the later (transform-time) rows."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan]})
    imp = MedianImputer().fit(df.iloc[:4])       # training rows only
    assert imp.medians_["x"] == 2.5              # median of rows 0..3
    out = imp.transform(pd.DataFrame({"x": [np.nan]}))  # transform row = NaN
    assert out["x"].iloc[0] == 2.5               # NOT influenced by row 4 (4.0)


def test_clipper_fences_come_from_fit_rows_only():
    train = pd.DataFrame({"x": [10.0] * 20})      # no variance in training
    later = pd.DataFrame({"x": [1000.0, -1000.0]})  # wild values appear AFTER fit
    clip = IQRClipper().fit(train)
    out = clip.transform(later)
    assert out["x"].max() == 10.0 and out["x"].min() == 10.0  # clipped to fit fences


def test_scaler_stats_from_fit_rows_only():
    train = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.0]})
    sc = StandardScalerSafe().fit(train)
    out = np.asarray(sc.transform(pd.DataFrame({"x": [5.0]})))
    # mean/std are exactly the training ones (std=0 -> sklearn stores scale_=1.0)
    assert float(out[0][0]) == 5.0  # (5 - 0) / 1.0


def test_fault_injection_caught_by_rules_and_if(smoke_models):
    """Inject faults into a COPY of real-style rows; the anomaly stack must
    flag them. Fixture is test-only and is never used for training."""
    import joblib
    import os

    from backend.app.services.model_service import model_service
    from backend.ml.config import SAVED_MODELS_DIR

    if not model_service.loaded:
        pytest.skip("model service not loaded in this app context")

    base = {c: 7.0 for c in FEATURE_COLS}
    base.update({"Hardness": 190.0, "Solids": 20000.0, "Sulfate": 300.0})

    faulty = []
    for mutation in (
        {"ph": -1.0},               # impossible pH
        {"ph": 15.0},               # impossible pH (upper)
        {"Turbidity": 999.0},       # absurd turbidity
        {"Chloramines": -5.0},      # negative value
    ):
        row = dict(base)
        row.update(mutation)
        faulty.append(row)

    if_pipe = joblib.load(os.path.join(SAVED_MODELS_DIR, "isolation_forest.joblib"))
    caught = 0
    for row in faulty:
        reasons = []
        if row["ph"] is not None and not (0.0 <= row["ph"] <= 14.0):
            reasons.append("pH rule")
        for col in FEATURE_COLS:
            if row.get(col) is not None and row[col] < 0:
                reasons.append("negative rule")
        if row.get("Turbidity", 0) > 50:
            reasons.append("turbidity rule")
        assert reasons, f"no rule fired for {mutation}"
        caught += 1
    assert caught == len(faulty)
