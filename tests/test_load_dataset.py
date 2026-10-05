"""Tests for backend.ml.load_dataset: column inspection, sample_id, DQ report, DB import.

Uses the tiny test-only fixture CSV (see tests/fixtures/README.txt).
"""
import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from backend.ml.load_dataset import (
    data_quality_report,
    import_to_db,
    load_raw_csv,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mini_water_potability.csv")
EXPECTED_FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]


def test_load_raw_csv_inspects_columns():
    df = load_raw_csv(FIXTURE)
    assert list(df.columns) == EXPECTED_FEATURES + ["Potability"]
    assert len(df) == 12  # fixture rows


def test_load_raw_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_csv(str(tmp_path / "nope.csv"))


def test_load_raw_csv_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("ph,Hardness\n7.0,200\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        load_raw_csv(str(bad))


def test_data_quality_report_counts(tmp_path):
    df = load_raw_csv(FIXTURE)
    report = data_quality_report(df)
    assert report["n_rows"] == 12
    # Fixture deliberately has 1 missing ph (empty cell); the 0.0 and 14.0
    # rows are PRESENT but impossible -- counted under ph_extremes_0_or_14.
    assert report["missing"]["ph"] == 1
    # Fixture has pH 0 and 14 -> both flagged as outside the strict 0<ph<14 sense
    assert report["ph_impossible"] >= 2
    assert report["ph_extremes_0_or_14"] == 2
    assert report["class_balance"]["potable"] + report["class_balance"]["not_potable"] == 12


def test_import_to_db_generates_sample_id(tmp_path):
    db_path = tmp_path / "test.db"
    df = load_raw_csv(FIXTURE)
    n = import_to_db(df, f"sqlite:///{db_path}")
    assert n == 12
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT sample_id FROM water_quality_dataset ORDER BY sample_id")
        ).fetchall()
    assert [r[0] for r in rows] == list(range(1, 13))  # 1..N generated


def test_import_to_db_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    df = load_raw_csv(FIXTURE)
    import_to_db(df, f"sqlite:///{db_path}")
    n2 = import_to_db(df, f"sqlite:///{db_path}")  # re-run must not duplicate
    assert n2 == 12
