/* Irrigation page: rule-engine verdict + ML cross-check + USSL diagram. */
"use strict";

const USSL_ZONES = {
  ec: [{code: "C1", to: 250}, {code: "C2", to: 750}, {code: "C3", to: 2250}, {code: "C4", to: Infinity}],
  sar: [{code: "S1", to: 10}, {code: "S2", to: 18}, {code: "S3", to: 26}, {code: "S4", to: Infinity}],
};
const CLASS_COLORS = {
  C1S1: "#15803d", C2S1: "#0e7490", C3S1: "#f59e0b", C4S1: "#dc2626",
  C4S2: "#7c3aed", C3S2: "#b45309",
};

async function loadIrrigationSummary() {
  try {
    const d = await api("/api/irrigation/summary");
    if (!d.available) {
      const h = document.getElementById("irr-hint");
      h.textContent = d.hint || "Run the irrigation training first.";
      h.classList.remove("hidden");
      return;
    }
    // Class distribution bar.
    const dist = d.class_distribution;
    barChart("ch-irr-dist", dist.labels,
      [{label: "Samples", data: dist.values,
        backgroundColor: dist.labels.map(c => CLASS_COLORS[c] || "#64748b")}],
      { plugins: { legend: { display: false } } });

    // Feature importance.
    barChart("ch-irr-imp", d.feature_importance.features,
      [{label: "Importance", data: d.feature_importance.values, backgroundColor: "#0e7490"}],
      { indexAxis: "y", plugins: { legend: { display: false } } });

    // Confusion matrix as colored grid.
    const cm = d.confusion_matrix;
    const max = Math.max(...cm.matrix.flat());
    document.getElementById("irr-cm").innerHTML =
      `<table><thead><tr><th>true \\ pred</th>${cm.labels.map(l => `<th>${l}</th>`).join("")}</tr></thead>` +
      cm.matrix.map((row, i) =>
        `<tr><th>${cm.labels[i]}</th>` + row.map(v => {
          const inten = v > 0 ? 15 + 75 * v / max : 0;
          return `<td style="background:rgba(14,116,144,${inten / 100});text-align:center;">${v}</td>`;
        }).join("") + "</tr>").join("") + "</table>";

    // USSL diagram (drawn once; the sample point is added on prediction).
    window._usslData = d.ussl_scatter;
    drawUssl(null);
    document.getElementById("irr-charts").classList.remove("hidden");
  } catch (err) { console.error(err); }
}

function drawUssl(sample) {
  const s = window._usslData;
  if (!s) return;
  const datasets = Object.entries(s.series).map(([cls, pts]) => ({
    label: cls, data: pts.ec.map((ec, i) => ({x: ec, y: pts.sar[i]})),
    backgroundColor: (CLASS_COLORS[cls] || "#64748b") + "aa", pointRadius: 3,
    showLine: false,
  }));
  if (sample) {
    datasets.push({
      label: "Your sample",
      data: [{x: sample.ec, y: sample.sar}],
      backgroundColor: "#000000", borderColor: "#000",
      pointRadius: 9, pointStyle: "rectRot", showLine: false,
    });
  }
  lineChart("ch-ussl", null, datasets, {
    parsing: false,
    plugins: { legend: { position: "bottom" } },
    scales: {
      x: { type: "logarithmic", title: { display: true, text: "EC (µS/cm) - salinity hazard" } },
      y: { title: { display: true, text: "SAR - sodium hazard" }, beginAtZero: true },
    },
  });
}

document.getElementById("btn-irrigation").addEventListener("click", async () => {
  const errBox = document.getElementById("irr-errors");
  errBox.classList.add("hidden");
  const fields = ["ph", "EC", "TDS", "CO3", "HCO3", "Cl", "F", "NO3",
    "SO4", "Na", "K", "Ca", "Mg", "TH"];
  const payload = {};
  fields.forEach(f => {
    const el = document.getElementById("i-" + f);
    if (el && el.value.trim() !== "") payload[f] = el.value;
  });
  try {
    const r = await api("/api/irrigation/predict", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    renderIrrigation(r);
  } catch (err) {
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
  }
});

function renderIrrigation(r) {
  const area = document.getElementById("irr-result");
  area.className = r.suitable ? "hint-box" : "error-box";
  area.innerHTML = `
    <div class="result-hero">
      <span class="big ${r.suitable ? "potable" : "not-potable"}">${r.ussl_class}</span>
      <div>
        <div><b>Verdict:</b> ${r.verdict}</div>
        <div><b>SAR:</b> ${r.sar} &nbsp; <b>RSC:</b> ${r.rsc} meq/L (${r.rsc_class})</div>
        <div class="note">ML cross-check: ${r.ml_cross_check.class}
          (${r.ml_cross_check.probability !== null
            ? (100 * r.ml_cross_check.probability).toFixed(1) + "% confidence" : "n/a"})</div>
      </div>
    </div>
    <ul style="margin:0.6rem 0 0 1.2rem;">${r.notes.map(n => `<li>${n}</li>`).join("")}</ul>`;
  document.getElementById("irr-detail").classList.remove("hidden");
  const ec = parseFloat(document.getElementById("i-EC").value);
  drawUssl({ec: ec, sar: r.sar});

  // Delivery confirmation: email for email users, SMS for phone users.
  area.insertAdjacentHTML("beforeend", wnNotifyLine(r.notification));
}

loadIrrigationSummary();
