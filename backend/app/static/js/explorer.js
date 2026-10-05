/* Dataset Explorer: graphs 1-6 + summary stats from /api/dataset/summary. */
"use strict";

(async () => {
  let d;
  try {
    d = await api("/api/dataset/summary");
  } catch (err) {
    console.error(err);
    return;
  }
  if (!d.available) {
    const h = document.getElementById("exp-hint");
    h.textContent = d.hint || "Dataset artifacts not found. Run: python -m backend.ml.train_all";
    h.classList.remove("hidden");
  }

  // G1: class balance doughnut.
  if (d.class_balance) {
    doughnut("e1-class", d.class_balance.labels, d.class_balance.values,
      ["#b91c1c", "#15803d"]);
  }

  // G2: missing values bar.
  if (d.missing_values) {
    barChart("e2-missing", d.missing_values.labels,
      [{ label: "Missing rows", data: d.missing_values.values, backgroundColor: "#f59e0b" }],
      { plugins: { legend: { display: false } } });
  }

  // G3: overlaid distributions, one feature at a time.
  if (d.feature_distributions) {
    const feats = Object.keys(d.feature_distributions);
    const sel3 = document.getElementById("e3-feature");
    sel3.innerHTML = feats.map(f => `<option>${f}</option>`).join("");
    const draw3 = () => {
      const f = d.feature_distributions[sel3.value];
      lineChart("e3-dist", f.bin_centers.map(c => c.toFixed(1)), [
        { label: "Potable (1)", data: f.potable, borderColor: "#0e7490",
          backgroundColor: "rgba(14,116,144,0.25)", fill: true },
        { label: "Not potable (0)", data: f.not_potable, borderColor: "#f59e0b",
          backgroundColor: "rgba(245,158,11,0.25)", fill: true },
      ], { scales: { x: { title: { display: true, text: `${sel3.value} (${f.unit})` } },
                      y: { title: { display: true, text: "Count" } } } });
    };
    sel3.addEventListener("change", draw3);
    draw3();
  }

  // G4: correlation heatmap as a colored HTML grid with values.
  if (d.correlation_matrix) {
    const cmz = d.correlation_matrix;
    const colorFor = v => {
      const a = Math.min(Math.abs(v), 1);
      return v >= 0 ? `rgba(14,116,144,${a})` : `rgba(220,38,38,${a})`;
    };
    const head = `<tr><th></th>${cmz.features.map(f => `<th>${f}</th>`).join("")}</tr>`;
    const body = cmz.matrix.map((row, i) =>
      `<tr><th>${cmz.features[i]}</th>${row.map(v =>
        `<td style="background:${colorFor(v)}; color:${Math.abs(v) > 0.55 ? "#fff" : "inherit"}; text-align:center; font-size:0.75rem;">${v.toFixed(2)}</td>`
      ).join("")}</tr>`).join("");
    document.getElementById("e4-corr").innerHTML = head + body;
  }

  // G5b: outlier counts.
  if (d.outlier_counts) {
    barChart("e5-outliers", d.outlier_counts.labels,
      [{ label: "Outliers (1.5*IQR)", data: d.outlier_counts.values, backgroundColor: "#7c3aed" }],
      { indexAxis: "y", plugins: { legend: { display: false } } });
  }

  // G6: before/after imputation.
  if (d.imputation_comparison && Object.keys(d.imputation_comparison).length) {
    const feats = Object.keys(d.imputation_comparison);
    const sel6 = document.getElementById("e6-feature");
    sel6.innerHTML = feats.map(f => `<option>${f}</option>`).join("");
    const draw6 = () => {
      const f = d.imputation_comparison[sel6.value];
      barChart("e6-imp", f.bin_centers.map(c => c.toFixed(1)), [
        { label: `Before (without ${f.n_imputed} NaN rows)`, data: f.before,
          backgroundColor: "#94a3b8" },
        { label: `After (median ${f.median_used} filled in)`, data: f.after,
          backgroundColor: "#0e7490" },
      ], { scales: { x: { title: { display: true, text: `${sel6.value} (${f.unit})` } } } });
    };
    sel6.addEventListener("change", draw6);
    draw6();
  } else {
    document.getElementById("e6-feature").closest(".card").querySelector(".sub").textContent =
      "No missing values in the dataset - imputation chart not applicable.";
  }

  // Summary stats table.
  document.querySelector("#e-stats tbody").innerHTML = (d.summary_stats || []).map(r => `
    <tr><td><b>${r.feature}</b></td><td>${r.unit}</td><td>${r.count}</td>
    <td>${r.missing}</td><td>${r.mean}</td><td>${r.std}</td><td>${r.min}</td>
    <td>${r["25%"]}</td><td>${r["50%"]}</td><td>${r["75%"]}</td><td>${r.max}</td></tr>`).join("");
})();
