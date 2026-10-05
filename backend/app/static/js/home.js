/* Home page: live stats + honest performance table (all real numbers from the API). */
"use strict";

function pct(v, digits) {
  return (v * 100).toFixed(digits === undefined ? 1 : digits) + "%";
}

async function loadHomeData() {
  // Stat tiles from the dashboard stats (readings stored so far).
  try {
    const s = await api("/api/stats");
    document.getElementById("h-readings").textContent = s.totals.readings;
  } catch (e) {
    document.getElementById("h-readings").textContent = "-";
  }

  // Performance table from the model metrics endpoints.
  try {
    const m = await api("/api/models/metrics");
    const cm = m.classifier_metrics && m.classifier_metrics.models;
    if (cm) {
      document.getElementById("perf-pot").textContent =
        pct(cm.RandomForest.test.accuracy) + " acc, AUC " + cm.RandomForest.test.roc_auc.toFixed(3);
      document.getElementById("perf-dt").textContent = pct(cm.DecisionTree.test.accuracy);
      document.getElementById("perf-xgb").textContent = pct(cm.XGBoost.test.accuracy);
    }
    if (m.ph_regressor) {
      const d = m.ph_regressor.app_decision.includes("MEDIAN") ? "not used (baseline wins)" : "in use";
      document.getElementById("perf-ph").textContent =
        `MAE ${m.ph_regressor.model.test.mae} vs ${m.ph_regressor.baseline.test.mae} - ${d}`;
    }
  } catch (e) { console.error(e); }

  try {
    const irr = await api("/api/irrigation/summary");
    if (irr.available && irr.metrics) {
      document.getElementById("perf-irr").textContent =
        pct(irr.metrics.test.accuracy) + " acc, macro-F1 " + irr.metrics.test.f1_macro.toFixed(3);
      document.getElementById("h-models").textContent = "6";
    } else {
      document.getElementById("perf-irr").textContent = "not trained yet";
      document.getElementById("h-models").textContent = "5";
    }
  } catch (e) {
    document.getElementById("perf-irr").textContent = "-";
  }
}

loadHomeData();
