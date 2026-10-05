# Limitations (read this before trusting any number)

1. **Weak feature-label correlation.** The nine parameters barely separate potable from
   non-potable water in this dataset (overlaid histograms almost overlap; correlation with
   the label is near zero). Accuracy is therefore expected around 65-70% - honest, not a bug.
2. **Label provenance.** Kaggle potability labels aggregate diverse real-world sources;
   they are not a controlled laboratory judgement for these exact samples.
3. **Guideline limits vs labels.** Many potable-labelled rows exceed WHO/BIS/EPA limits
   (Solids averages ~22,000 mg/L vs the 500 mg/L desirable limit). Limits are display-only
   and never create labels/features; the radar and out-of-limit charts are guidance, not truth.
4. **pH prediction is weak.** pH is essentially independent of the other parameters in this
   data (R² ≈ 0). The app therefore uses median imputation unless the tuned model clearly
   beats the mean baseline - the decision is computed at training time, not hoped for.
5. **Anomaly detection has no ground truth.** Isolation Forest flags statistical outliers;
   there are no anomaly labels, so no accuracy/recall is reported anywhere. Rules add
   human-readable physical checks. Validation uses fault injection on test-only copies.
6. **Simulated deployments.** This is software-only: no sensors, no real-time streams.
   The optional simulator replays real held-out test rows so demos are reproducible.
7. **Single split variance.** One fixed split (random_state=42) is reported alongside 5-fold
   CV; with 3,276 rows the test set is ~655 rows, so single-split metrics have some noise.
8. **Class imbalance.** ~61/39 split is handled with in-fold SMOTE / class weights; recall
   on the minority (potable) class remains the hardest metric and is reported per class.
9. **XGBoost determinism.** Results are seeded, but XGBoost can vary slightly across
   CPU architectures/versions; regenerate artifacts with the included one command.

## Irrigation-specific limitations

- **Single region, one state:** the irrigation model trains on 1,090 Telangana (India)
  groundwater samples. Thresholds (USSL, RSC) are international standards, but the
  label distribution and ion ranges are regional.
- **Label conventions differ from textbook formulas in places:** the dataset's own RSC
  column disagrees with the standard formula on most rows (we could not reproduce their
  exact ion convention), though their class labels follow the standard 1.25/2.5 meq/L
  thresholds. The app's rule engine uses the standard formula and says so.
- **Hazards not modelled:** boron toxicity, specific-ion effects (chloride, nitrate on
  sensitive crops), clogging/emitter risk, soil drainage and crop water demand.
- **Rare classes excluded from ML:** USSL classes with fewer than 10 samples (36 rows)
  are documented and excluded, so the ML cross-check never predicts them; the rule
  engine still classifies them correctly.

## Deliberate exclusions (per project scope)

No alerts or notifications, no IoT/MQTT/LoRa, no Docker/Kubernetes/cloud, no Grafana,
no synthetic training data, no test-set tuning, no hard-coded chart numbers.
