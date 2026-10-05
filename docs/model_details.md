# Model Details & Honest Results

> Numbers below are from the REAL run on the genuine Kaggle dataset
> (3,276 rows; imported 2026-10-05; artifacts in `outputs/models/*.json`).
> Re-running `python -m backend.ml.train_all` reproduces them (seeded).

## Training protocol (no leakage)

1. **Split first**: 20% stratified test set, `random_state=42`, never touched during tuning.
2. **Preprocessing inside pipelines**: median imputation -> IQR outlier clipping ->
   standard scaling -> (SMOTE for RF/XGB) -> model. Every transformer is fitted on
   training folds only (proven by tests in `tests/test_leakage.py`).
3. **Tuning**: GridSearchCV with inner 5-fold stratified CV (classification) / KFold
   (regression), scoring F1 (potable) / neg-MAE (pH). Optuna was deliberately not used.
4. **Imbalance**: SMOTE applied only inside CV folds via `imblearn`'s Pipeline;
   the Decision Tree uses `class_weight="balanced"`.
5. **Final evaluation**: one-shot on the untouched test set; CV numbers reported as mean ± std.

## Models

| Purpose | Algorithm | Notes |
|---|---|---|
| Potability | RandomForestClassifier | SMOTE in-fold |
| Potability | DecisionTreeClassifier | class_weight balanced |
| Potability | XGBClassifier | primary serving model |
| pH | RandomForestRegressor | inputs = 8 params (no pH, no Potability); trains only where 0 < pH < 14 |
| Anomaly | IsolationForest | contamination="auto"; no labels exist, so no accuracy is claimed |
| Baselines | DummyRegressor(mean), plain RF/DT | honest floor for comparisons |

## Metric tables

Fill after the real run (`outputs/models/classifier_metrics.json`, `ph_regressor.json`):

| Model | CV accuracy | CV F1 | CV ROC-AUC | Test acc | F1 pot. | F1 not pot. | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| RandomForest (SMOTE in-fold) | 0.673 ± 0.019 | 0.554 | 0.693 | **0.652** | 0.506 | 0.732 | **0.672** |
| DecisionTree (class_weight) | 0.570 ± 0.029 | 0.507 | 0.591 | 0.584 | 0.488 | 0.650 | 0.587 |
| XGBoost (SMOTE in-fold) | 0.636 ± 0.021 | 0.534 | 0.675 | 0.627 | 0.503 | 0.701 | 0.655 |
| Baseline RF (raw, no pipeline) | - | - | - | 0.663 | - | - | - |
| Baseline DT (raw, no pipeline) | - | - | - | 0.596 | - | - | - |

Best RF params: `n_estimators=200, max_depth=20, min_samples_leaf=1`.
Best DT params: `max_depth=None, min_samples_leaf=10` (class_weight=balanced).
Best XGB params: `n_estimators=200, max_depth=5, learning_rate=0.1`.

**Honest reading:** Random Forest wins (65.2% accuracy, AUC 0.672). Note the raw baseline
RF already scores 66.3% - on this weak-signal dataset the full pipeline does not add much,
and we report that openly rather than tuning on the test set to fake an improvement.
Recall on the potable class is the weak spot (0.457 for RF): the model is conservative
about calling water potable.

| pH model | CV MAE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| Mean baseline | 1.240 ± 0.052 | -0.002 ± 0.002 | 1.222 | 1.557 | -0.002 |
| RandomForest (tuned) | 1.234 ± 0.042 | 0.001 ± 0.022 | 1.233 | 1.570 | -0.018 |

**App decision (computed at training time): USE MEDIAN IMPUTATION** - the tuned model does
NOT clearly beat the predict-the-mean baseline (CV MAE 1.234 vs 1.240, test R² negative).
This is the honest, expected outcome: pH is essentially independent of the other eight
parameters in this dataset. Trained on the 2,784 rows with 0 < pH < 14 (492 dropped: 491
missing + 1 pH=0 extreme).

**Why scores are modest:** exactly as anticipated - the nine measured chemical parameters
correlate only weakly with the potability label (see the Dataset Explorer correlation
heatmap: all |r| < 0.1 vs Potability). The Kaggle labels come from varied real-world
sources; with these features alone, ~65-70% accuracy is the realistic ceiling. Anything
dramatically higher would signal leakage or overfitting, both of which are explicitly
tested against in this project.

## Isolation Forest (real run)

Fitted on the 2,620-row training split only; evaluated on the untouched 656-row test
split: **104/656 rows (~15.9%) flagged** as statistical outliers, with the threshold and
full score distribution shown in graph 21. Rules (pH range, negatives, extreme turbidity)
combine with the IF flag to produce the human-readable reason shown on the Predict page.

## Graph descriptions (as implemented)

| # | Graph | Type | Source |
|---|---|---|---|
| 1 | Class balance | Chart.js doughnut | outputs/dataset/class_balance.json |
| 2 | Missing values per column | bar | missing_values.json |
| 3 | Feature distributions by class | overlaid histograms (per-feature selector) | feature_distributions.json |
| 4 | Correlation heatmap | colored HTML grid with values | correlation_matrix.json |
| 5 | Boxplots + outlier counts | matplotlib PNG + bar | boxplots.png, outlier_counts.json |
| 6 | Before vs after imputation | histogram per NaN column | imputation_comparison.json |
| 7 | Metric comparison | grouped bar | models/classifier_metrics.json |
| 8 | CV score ± std | floating bar | classifier_metrics.json |
| 9 | Confusion matrices | colored HTML grids | classifier_metrics.json |
| 10 | ROC curves + diagonal | line | roc_curves.json |
| 11 | Precision-Recall curves | line | pr_curves.json |
| 12 | Feature importance + RFE ranking | horizontal bar | feature_importance.json |
| 13 | Learning curve | matplotlib PNG | learning_curve.png |
| 14 | Validation curves (n_estimators, max_depth) | line + selector | validation_curves.json |
| 15 | Threshold vs precision/recall/F1 | line (out-of-fold, train only) | threshold_curve.json |
| 16 | Calibration curve | line | calibration.json |
| 17 | Baseline vs improved | grouped bar | baseline_vs_improved.json |
| 18 | Predicted vs actual pH | scatter + y=x | ph_scatter.json |
| 19 | Residual histogram | histogram | ph_residuals.json |
| 20 | pH metrics vs baseline | bar | ph_metrics_vs_baseline.json |
| 21 | Anomaly score histogram + threshold | bar + threshold line | anomaly_score_hist.json |
| 22 | 2D PCA scatter with anomalies | scatter | anomaly_pca.json |
| 23 | Anomalies by reason | horizontal bar | anomaly_reasons.json |
| 24-27 | Dashboard charts | doughnut/line/bar/histogram | /api/stats |
| 28-30 | Predict-page charts | radar/bar | /api/predict |

## Anomaly detection design

- Isolation Forest fitted on training features only; score = negated `decision_function`
  (higher = more anomalous). `contamination="auto"` because inventing a contamination
  percentage without any labelled anomalies would be arbitrary; the score histogram and
  threshold are shown openly.
- The final flag combines the IF score with physical rule checks (pH outside 0-14,
  negative values, turbidity > 50 NTU). The "reason" lists every check that fired.
- Validation: pytest injects faults into a COPY of real-style rows (test-only) and
  asserts the stack catches them. No accuracy is reported - there is no ground truth.
