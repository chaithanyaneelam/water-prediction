"""WATERNET training pipeline - ONE command for all training + plots.

Usage:
    python -m backend.ml.train_all            # full run on the real dataset
    python -m backend.ml.train_all --smoke    # quick end-to-end test (fixture only)

Steps:
1. Import the real CSV into water_quality_dataset (skipped in smoke mode).
2. Dataset Explorer graphs 1-6.
3. Classifier training/comparison (graphs 7-17).
4. pH regressor vs mean baseline (graphs 18-20).
5. Isolation Forest + rules (graphs 21-23).
"""
import argparse
import sys

from backend.ml import load_dataset, make_dataset_plots


def main(smoke: bool = False) -> int:
    print("=" * 60)
    print("WATERNET TRAINING PIPELINE" + ("  [SMOKE MODE - test fixture]" if smoke else ""))
    print("=" * 60)

    if smoke:
        print("\n[1/5] smoke mode: skipping CSV import (fixture is used directly)")
    else:
        print("\n[1/5] importing real dataset...")
        load_dataset.main()

    print("\n[2/5] dataset explorer plots...")
    if smoke:
        print("      skipped in smoke mode (needs the imported DB table)")
    else:
        make_dataset_plots.main()

    print("\n[3/5] classifier training + comparison...")
    from backend.ml.train_classifier import main as train_classifier
    train_classifier(smoke=smoke)

    print("\n[4/5] pH regressor...")
    from backend.ml.train_ph_regressor import main as train_ph
    train_ph(smoke=smoke)

    print("\n[5/5] isolation forest...")
    from backend.ml.train_isolation_forest import main as train_if
    train_if(smoke=smoke)

    print("\n" + "=" * 60)
    print("DONE. Artifacts in backend/ml/outputs/ and backend/ml/saved_models/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="end-to-end test on the tiny test-only fixture")
    args = parser.parse_args()
    sys.exit(main(smoke=args.smoke))
