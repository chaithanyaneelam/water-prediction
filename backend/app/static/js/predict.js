/* Predict page: POST /api/predict, then draw graphs 28-30 + treatment text. */
"use strict";

const WN_LIMITS = { // display only; matches backend/ml/config.py
  ph: [6.5, 8.5], Turbidity: [0, 5], Solids: [0, 500], Hardness: [0, 200],
  Chloramines: [0, 4], Sulfate: [0, 250], Trihalomethanes: [0, 80],
  Organic_carbon: [0, 2], Conductivity: [0, 400],
};

document.getElementById("btn-predict").addEventListener("click", async () => {
  const fields = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"];
  const payload = {};
  fields.forEach(f => {
    const el = document.getElementById("f-" + f);
    if (el.value.trim() !== "") payload[f] = el.value;
  });

  const errBox = document.getElementById("predict-errors");
  errBox.classList.add("hidden");
  try {
    const res = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderResult(res);
  } catch (err) {
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
  }
});

/* Manual (re-)send of the last prediction's report to the user's email/SMS. */
let _lastReadingId = null;

function updateSendStatus(html) {
  const row = document.getElementById("send-report-row");
  if (!row) return;
  const old = document.getElementById("send-report-status");
  if (old) old.remove();
  const holder = document.createElement("div");
  holder.id = "send-report-status";
  holder.innerHTML = html || "";
  row.after(holder);
}

async function sendReport() {
  if (_lastReadingId == null) return;
  const btn = document.getElementById("btn-send-report");
  btn.disabled = true;
  btn.textContent = "Sending…";
  try {
    const resp = await api(`/api/predict/${_lastReadingId}/send`, { method: "POST" });
    updateSendStatus(wnNotifyLine(resp.notification));
  } catch (err) {
    updateSendStatus(`<p class="note">Could not send: ${wnEsc(err.message)}</p>`);
  } finally {
    btn.disabled = false;
    btn.textContent = "📤 Send report";
  }
}

function renderResult(r) {
  document.getElementById("result-area").className = "";
  document.getElementById("result-area").innerHTML = `
    <div class="result-hero">
      <span class="big ${r.potability === 1 ? "potable" : "not-potable"}">
        ${r.potability_label.toUpperCase()}
      </span>
      <div>
        <div><b>Probability potable:</b> ${(100 * r.probability).toFixed(1)}%</div>
        <div><b>pH used:</b> ${r.ph ?? "-"}
          ${r.ph_filled_by === "model" ? "(predicted by model)"
            : r.ph_filled_by === "median" ? "(median imputed)" : ""}</div>
        <div><b>Anomaly:</b> ${r.anomaly.flag
          ? `<span class="badge warn">flagged (${r.anomaly.score})</span>`
          : `<span class="badge good">no</span>`}
          <span class="note">reason: ${r.anomaly.reason}</span></div>
      </div>
    </div>`;

  document.getElementById("result-charts").classList.remove("hidden");

  // Graph 29: probability bar rendered as a horizontal half-gauge.
  wnDestroy("ch-gauge");
  const ctxG = document.getElementById("ch-gauge");
  ctxG._chart = new Chart(ctxG, {
    type: "bar",
    data: {
      labels: ["Potable probability"],
      datasets: [{
        data: [r.probability * 100],
        backgroundColor: r.probability >= 0.5 ? "#15803d" : "#b91c1c",
        borderWidth: 0,
      }],
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { min: 0, max: 100 },
        y: { display: false },
      },
    },
  });

  // Graph 28: radar of reading vs guideline limits (normalized value/limit).
  const params = ["Turbidity", "Solids", "Hardness", "Chloramines",
    "Sulfate", "Trihalomethanes", "Organic_carbon", "Conductivity"];
  const readingVals = params.map(p => {
    const lim = WN_LIMITS[p][1];
    return Math.min(+document.getElementById("f-" + p).value / lim, 5);
  });
  const limitVals = params.map(() => 1);
  wnDestroy("ch-radar");
  const ctxR = document.getElementById("ch-radar");
  ctxR._chart = new Chart(ctxR, {
    type: "radar",
    data: {
      labels: params,
      datasets: [
        { label: "This reading", data: readingVals, fill: true,
          backgroundColor: "rgba(14,116,144,0.2)", borderColor: "#0e7490" },
        { label: "Guideline limit", data: limitVals, fill: false,
          borderColor: "#dc2626", borderDash: [4, 4], pointRadius: 0 },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } } },
  });

  // Graph 30: count of parameters outside limits.
  const hits = r.guideline_hits || [];
  const labels = hits.map(h => h.parameter);
  wnDestroy("ch-others");
  const ctxO = document.getElementById("ch-others");
  ctxO._chart = new Chart(ctxO, {
    type: "bar",
    data: {
      labels: labels.length ? labels : ["none"],
      datasets: [{
        label: "Times over limit",
        data: labels.length ? hits.map(() => 1) : [0],
        backgroundColor: labels.length ? "#dc2626" : "#94a3b8",
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { precision: 0 } } },
    },
  });

  // Treatment text (rule-based, display only).
  const tips = r.treatment || [];
  document.getElementById("treatment-list").innerHTML = tips.length
    ? `<div class="hint-box"><b>Suggested treatment:</b><ul style="margin:0.4rem 0 0 1.2rem;">
       ${tips.map(t => `<li>${t}</li>`).join("")}</ul></div>`
    : `<p class="note">No treatment suggestions - all values within guideline limits.</p>`;

  // Manual send button + delivery confirmation (email for email users,
  // SMS for phone users). The report is auto-sent after the prediction;
  // the button re-sends it on demand.
  _lastReadingId = r.reading_id;
  const tl = document.getElementById("treatment-list");
  tl.insertAdjacentHTML("beforeend", `<div id="send-report-row" style="margin-top:0.75rem;">
    <button class="btn secondary" id="btn-send-report" type="button">📤 Send report</button>
    <span class="note">sends this report to your registered email or phone</span>
  </div>`);
  document.getElementById("btn-send-report").addEventListener("click", sendReport);
  updateSendStatus(wnNotifyLine(r.notification));
}
