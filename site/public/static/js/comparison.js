/* Model Comparison page: render graphs 7-23 from /api/models/metrics (no hard-coded numbers). */
"use strict";

/* WN_COLORS comes from common.js */

/* ---------------- tabs ---------------- */
const PANES = { "tab-clf": "pane-clf", "tab-ph": "pane-ph", "tab-if": "pane-if" };
Object.keys(PANES).forEach(tabId => {
  document.getElementById(tabId).addEventListener("click", () => {
    Object.entries(PANES).forEach(([t, p]) => {
      document.getElementById(t).classList.toggle("active", t === tabId);
      document.getElementById(p).classList.toggle("hidden", t !== tabId);
    });
  });
});

function missingArtifacts() {
  document.getElementById("cmp-hint").textContent =
    "Model artifacts not found yet. Run: python -m backend.ml.train_all";
  document.getElementById("cmp-hint").classList.remove("hidden");
}

/* ---------------- classifier tab ---------------- */
function renderClassifier(d) {
  const models = Object.keys(d.classifier_metrics.models);
  const cm = d.classifier_metrics.models;

  // G7: grouped metric bars per model.
  const metricKeys = ["accuracy", "precision_potable", "recall_potable",
    "f1_potable", "f1_not_potable", "roc_auc"];
  const metricNames = ["Accuracy", "Precision (pot.)", "Recall (pot.)",
    "F1 (pot.)", "F1 (not pot.)", "ROC-AUC"];
  barChart("g7-metrics", metricNames,
    models.map((m, i) => ({
      label: m,
      data: metricKeys.map(k => cm[m].test[k] ?? 0),
      backgroundColor: WN_COLORS[i],
    })),
    { scales: { y: { min: 0, max: 1 } } });

  // G8: CV mean with +-std floating bars.
  barChart("g8-cv", models,
    models.map((m, i) => {
      const mean = cm[m].cv_accuracy_mean, std = cm[m].cv_accuracy_std;
      return {
        label: `${m} (mean ${mean.toFixed(3)})`,
        data: [mean - std],
        backgroundColor: WN_COLORS[i] + "cc",
        borderWidth: 0,
        // draw the "rest" of the bar transparently to fake floating bars
      };
    }).concat([{
      label: "+-1 std (stacked)",
      data: models.map(m => 2 * cm[m].cv_accuracy_std),
      backgroundColor: models.map((m, i) => WN_COLORS[i] + "55"),
      borderWidth: 0,
    }]),
    { scales: { y: { min: 0, max: 1, stacked: true } },
      plugins: { legend: { display: false } } });

  // G10: ROC overlay with diagonal.
  lineChart("g10-roc", null,
    models.map((m, i) => ({
      label: `${m} (AUC ${cm[m].test.roc_auc.toFixed(3)})`,
      data: d.roc_curves.models[m].fpr.map((f, j) => ({ x: f, y: d.roc_curves.models[m].tpr[j] })),
      borderColor: WN_COLORS[i], tension: 0.2, pointRadius: 0,
    })).concat([{
      label: "Random (AUC 0.5)",
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: "#94a3b8", borderDash: [6, 6], pointRadius: 0,
    }]),
    { parsing: false,
      scales: { x: { title: { display: true, text: "False positive rate" }, min: 0, max: 1 },
                y: { title: { display: true, text: "True positive rate" }, min: 0, max: 1 } } });

  // G11: PR curves.
  lineChart("g11-pr", null,
    models.map((m, i) => ({
      label: m,
      data: d.pr_curves[m].recall.map((r, j) => ({ x: r, y: d.pr_curves[m].precision[j] })),
      borderColor: WN_COLORS[i], tension: 0.2, pointRadius: 0,
    })),
    { parsing: false,
      scales: { x: { title: { display: true, text: "Recall" }, min: 0, max: 1 },
                y: { title: { display: true, text: "Precision" }, min: 0, max: 1 } } });

  // G16: calibration curve.
  if (d.calibration && !d.calibration.skipped) {
    lineChart("g16-calib", null, [
      { label: "Random Forest",
        data: d.calibration.prob_pred.map((p, j) => ({ x: p, y: d.calibration.prob_true[j] })),
        borderColor: WN_COLORS[0], pointRadius: 4 },
      { label: "Perfectly calibrated",
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "#94a3b8", borderDash: [6, 6], pointRadius: 0 },
    ],
    { parsing: false,
      scales: { x: { title: { display: true, text: "Mean predicted probability" }, min: 0, max: 1 },
                y: { title: { display: true, text: "Fraction of positives" }, min: 0, max: 1 } } });
  } else {
    document.getElementById("g16-calib").closest(".card").querySelector(".sub").textContent =
      "Calibration data appears after a full training run.";
  }

  // G9: confusion matrices as colored HTML grids.
  document.getElementById("g9-cm").innerHTML = models.map(m => {
    const M = cm[m].confusion_matrix;
    const max = Math.max(...M.flat());
    const cell = (v) => {
      const inten = 0.15 + 0.75 * (v / max);
      const color = v > 0 ? `color-mix(in srgb, #0e7490 ${inten * 100}%, transparent)` : "transparent";
      return `<td style="background:${color}; text-align:center; font-weight:600;">${v}</td>`;
    };
    return `<div>
      <b>${m}</b>
      <table style="margin-top:0.4rem;">
        <thead><tr><th></th><th>pred: not pot.</th><th>pred: potable</th></tr></thead>
        <tbody>
          <tr><th>true: not pot.</th>${cell(M[0][0])}${cell(M[0][1])}</tr>
          <tr><th>true: potable</th>${cell(M[1][0])}${cell(M[1][1])}</tr>
        </tbody>
      </table></div>`;
  }).join("");

  // G12: feature importance RF vs XGB + RFE ranking.
  const fi = d.feature_importance;
  barChart("g12-imp", fi.RandomForest.features, [
    { label: "Random Forest", data: fi.RandomForest.values, backgroundColor: WN_COLORS[0] },
    { label: "XGBoost", data: fi.XGBoost.values, backgroundColor: WN_COLORS[1] },
  ], { indexAxis: "y" });
  barChart("g12-rfe", fi.RFE_ranking.features, [
    { label: "RFE rank (1 = selected)", data: fi.RFE_ranking.values, backgroundColor: WN_COLORS[2] },
  ], { indexAxis: "y", plugins: { legend: { display: false } } });

  // G14: validation curves (selectable parameter).
  const drawVC = (key) => {
    const vc = d.validation_curves[key];
    lineChart("g14-vc", vc.values, [
      { label: "Training F1", data: vc.train_mean, borderColor: WN_COLORS[0] },
      { label: "CV F1", data: vc.cv_mean, borderColor: WN_COLORS[1] },
    ], { scales: { y: { min: 0, max: 1 } } });
  };
  document.getElementById("g14-param").addEventListener("change", e => drawVC(e.target.value));
  if (d.validation_curves && !d.validation_curves.skipped) drawVC("n_estimators");

  // G15: threshold curve.
  lineChart("g15-thr", d.threshold_curve.thresholds, [
    { label: "Precision", data: d.threshold_curve.precision, borderColor: WN_COLORS[0] },
    { label: "Recall", data: d.threshold_curve.recall, borderColor: WN_COLORS[1] },
    { label: "F1", data: d.threshold_curve.f1, borderColor: WN_COLORS[2] },
  ], { scales: { y: { min: 0, max: 1 } } });

  // G17: baseline vs improved.
  const rows = d.baseline_vs_improved.rows;
  barChart("g17-bvi", rows.map(r => r.model + " (" + r.variant.split(" (")[0] + ")"), [
    { label: "Accuracy", data: rows.map(r => r.accuracy), backgroundColor: WN_COLORS[0] },
    { label: "F1 (potable)", data: rows.map(r => r.f1_potable ?? 0), backgroundColor: WN_COLORS[1] },
    { label: "ROC-AUC", data: rows.map(r => r.roc_auc ?? 0), backgroundColor: WN_COLORS[2] },
  ], { scales: { y: { min: 0, max: 1 } } });

  // Metrics table.
  document.querySelector("#metrics-table tbody").innerHTML = models.map(m => {
    const s = cm[m];
    return `<tr>
      <td><b>${m}</b></td>
      <td>${s.cv_accuracy_mean.toFixed(3)} ± ${s.cv_accuracy_std.toFixed(3)}</td>
      <td>${s.cv_f1_mean.toFixed(3)} ± ${s.cv_f1_std.toFixed(3)}</td>
      <td>${s.cv_roc_auc_mean.toFixed(3)} ± ${s.cv_roc_auc_std.toFixed(3)}</td>
      <td>${s.test.accuracy.toFixed(3)}</td>
      <td>${s.test.precision_potable.toFixed(3)}</td>
      <td>${s.test.recall_potable.toFixed(3)}</td>
      <td>${s.test.f1_potable.toFixed(3)}</td>
      <td>${s.test.f1_not_potable.toFixed(3)}</td>
      <td>${s.test.roc_auc.toFixed(3)}</td>
      <td class="note">${JSON.stringify(s.best_params)}</td>
    </tr>`;
  }).join("");
}

/* ---------------- pH tab ---------------- */
function renderPH(info) {
  barChart("g20-phmetrics", info.g20.labels, [
    { label: "Mean baseline", data: info.g20.baseline, backgroundColor: "#94a3b8" },
    { label: "Random Forest", data: info.g20.model, backgroundColor: WN_COLORS[0] },
  ]);
  document.getElementById("ph-decision").innerHTML =
    `<b>App decision:</b> ${info.app_decision}`;

  lineChart("g18-scatter", null, [
    { label: "Test samples",
      data: info.actual.map((a, i) => ({ x: a, y: info.predicted[i] })),
      borderColor: WN_COLORS[0], backgroundColor: WN_COLORS[0], showLine: false },
    { label: "y = x",
      data: [{ x: 0, y: 0 }, { x: 14, y: 14 }],
      borderColor: "#94a3b8", borderDash: [6, 6], pointRadius: 0 },
  ],
  { parsing: false,
    scales: { x: { title: { display: true, text: "Actual pH" }, min: 0, max: 14 },
              y: { title: { display: true, text: "Predicted pH" }, min: 0, max: 14 } } });

  barChart("g19-resid", info.hist_centers.map(c => c.toFixed(2)),
    [{ label: "Residual count", data: info.hist_counts, backgroundColor: WN_COLORS[0] }],
    { plugins: { legend: { display: false } } });

  const b = info.baseline, m = info.model;
  document.querySelector("#ph-table tbody").innerHTML = `
    <tr><td><b>Mean baseline</b></td><td>${b.cv.mae} ± ${b.cv.mae_std}</td>
      <td>${b.cv.r2} ± ${b.cv.r2_std}</td><td>${b.test.mae}</td>
      <td>${b.test.rmse}</td><td>${b.test.r2}</td></tr>
    <tr><td><b>Random Forest (tuned)</b></td><td>${m.cv.mae} ± ${m.cv.mae_std}</td>
      <td>${m.cv.r2} ± ${m.cv.r2_std}</td><td>${m.test.mae}</td>
      <td>${m.test.rmse}</td>      <td>${m.test.r2}</td></tr>`;
}

/* ---------------- anomaly tab ---------------- */
function renderIF(d) {
  const h = d.anomaly_score_hist;
  const colors = h.bin_centers.map(c => c >= h.threshold ? "#dc2626" : "#64748b");
  barChart("g21-hist", h.bin_centers.map(c => c.toFixed(3)),
    [{ label: "Rows", data: h.counts, backgroundColor: colors }],
    { plugins: {
        legend: { display: false },
        tooltip: { callbacks: { title: items => "score " + items[0].label } } } });

  barChart("g23-reasons", d.anomaly_reasons.labels,
    [{ label: "Count", data: d.anomaly_reasons.values, backgroundColor: WN_COLORS[0] }],
    { indexAxis: "y" });

  const p = d.anomaly_pca;
  lineChart("g22-pca", null, [
    { label: "Normal", data: p.normal.x.map((x, i) => ({ x, y: p.normal.y[i] })),
      backgroundColor: "#64748b", borderColor: "#64748b", showLine: false },
    { label: "Anomaly", data: p.anomaly.x.map((x, i) => ({ x, y: p.anomaly.y[i] })),
      backgroundColor: "#dc2626", borderColor: "#dc2626", showLine: false, pointRadius: 5 },
  ],
  { parsing: false,
    scales: { x: { title: { display: true, text: `PC1 (${(100 * p.explained_variance[0]).toFixed(1)}%)` } },
              y: { title: { display: true, text: `PC2 (${(100 * p.explained_variance[1]).toFixed(1)}%)` } } } });
}

/* ---------------- boot ---------------- */
(async () => {
  try {
    const d = await api("/api/models/metrics");
    const needed = ["classifier_metrics", "roc_curves", "pr_curves", "feature_importance",
      "threshold_curve", "baseline_vs_improved", "ph_regressor", "ph_scatter",
      "anomaly_score_hist", "anomaly_pca", "anomaly_reasons"];
    if (needed.some(k => !d.available[k])) { missingArtifacts(); return; }
    renderClassifier(d);
    if (d.ph_regressor && d.ph_scatter) {
      renderPH({
        app_decision: d.ph_regressor.app_decision,
        baseline: d.ph_regressor.baseline,   // {cv, test, note}
        model: d.ph_regressor.model,         // {best_params, cv, test}
        g20: d.ph_metrics_vs_baseline,       // {labels, baseline[], model[]}
        actual: d.ph_scatter.actual,
        predicted: d.ph_scatter.predicted,
        hist_centers: d.ph_residuals.hist_centers,
        hist_counts: d.ph_residuals.hist_counts,
      });
    }
    renderIF(d);
  } catch (err) {
    missingArtifacts();
    console.error(err);
  }
})();
