"""WATERNET demo simulator (OPTIONAL).

Replays rows from the REAL held-out test split of the trained classifier
(the same 20% split, with a fixed random_state, that the models never trained
on) through POST /api/predict. Rows are stored in the readings table flagged
`source=simulated` so demo data never mixes with manual entries.

Usage (site must be running: python run.py):
    python simulator/simulate.py            # send 10 readings, 2s apart
    python simulator/simulate.py -n 30 -i 0.5
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ml.config import FEATURE_COLS, LABEL_COL, RANDOM_STATE  # noqa: E402


def load_test_split():
    """Rebuild the exact held-out test split from the imported dataset."""
    from backend.ml.make_dataset_plots import load_dataset_from_db
    from sklearn.model_selection import train_test_split

    df = load_dataset_from_db()
    # Same split call as train_classifier.py -> identical untouched test rows.
    _, test = train_test_split(
        df, test_size=0.2, stratify=df[LABEL_COL], random_state=RANDOM_STATE
    )
    return test


def post_reading(row) -> dict:
    payload = {"source": "simulated"}
    for c in FEATURE_COLS:
        v = row[c]
        payload[c] = None if pd.isna(v) else float(v)
    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=10, help="how many readings to send")
    parser.add_argument("-i", "--interval", type=float, default=2.0,
                        help="seconds between readings")
    args = parser.parse_args()

    test_rows = load_test_split()
    # The predict endpoint requires the 8 non-ph fields (pH optional). Rows with
    # missing required values are skipped - exactly what a lab would re-measure.
    required = [c for c in FEATURE_COLS if c != "ph"]
    complete = test_rows.dropna(subset=required)
    # Rows with impossible pH are rejected by validation (correctly) - skip them.
    complete = complete[
        complete["ph"].isna() | ((complete["ph"] >= 0) & (complete["ph"] <= 14))
    ]
    skipped = len(test_rows) - len(complete)
    if skipped:
        print(f"[sim] skipping {skipped} held-out rows with missing required values")
    print(f"[sim] replaying {min(args.n, len(complete))} of {len(complete)} "
          f"complete held-out rows (source=simulated)")
    ok = 0
    for _, row in complete.head(args.n).iterrows():
        try:
            result = post_reading(row)
            print(f"[sim] #{result['reading_id']}: {result['potability_label']}"
                  f" (p={result['probability']:.2f},"
                  f" anomaly={result['anomaly']['flag']})")
            ok += 1
        except Exception as exc:
            print(f"[sim] failed: {exc}")
        time.sleep(args.interval)
    print(f"[sim] sent {ok} readings. See them on the Dashboard/History pages "
          f"(badge: simulated).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
