/* History page: /api/readings with filters + pagination; chart 25 via /api/stats. */
"use strict";

let PAGE = 1;

function buildQuery() {
  const p = new URLSearchParams();
  p.set("page", PAGE);
  p.set("per_page", 15);
  const v = id => document.getElementById(id).value;
  if (v("q-search")) p.set("search", v("q-search"));
  if (v("q-potable")) p.set("potable", v("q-potable"));
  if (v("q-anomaly")) p.set("anomaly", v("q-anomaly"));
  if (v("q-from")) p.set("date_from", v("q-from"));
  return p.toString();
}

async function loadHistory() {
  try {
    const d = await api("/api/readings?" + buildQuery());
    const tb = document.querySelector("#hist-table tbody");
    tb.innerHTML = d.items.map(r => `
      <tr>
        <td>${r.id}</td>
        <td>${(r.created_at || "").replace("T", " ").slice(0, 16)}</td>
        <td><span class="badge muted">${r.source}</span></td>
        <td>${r.ph ?? "-"}</td>
        <td>${r.Hardness ?? "-"}</td><td>${r.Solids ?? "-"}</td>
        <td>${r.Chloramines ?? "-"}</td><td>${r.Sulfate ?? "-"}</td>
        <td>${r.Conductivity ?? "-"}</td><td>${r.Organic_carbon ?? "-"}</td>
        <td>${r.Trihalomethanes ?? "-"}</td><td>${r.Turbidity ?? "-"}</td>
        <td>${r.potability === 1
          ? '<span class="badge good">potable</span>'
          : '<span class="badge bad">not potable</span>'}</td>
        <td>${(100 * r.probability).toFixed(1)}%</td>
        <td>${r.is_anomaly ? '<span class="badge warn">flagged</span>' : "-"}</td>
      </tr>`).join("");
    document.getElementById("hist-meta").textContent =
      `Page ${d.page} - showing ${d.items.length} of ${d.total} readings`;
    document.getElementById("btn-prev").disabled = PAGE <= 1;
    document.getElementById("btn-next").disabled = PAGE * d.per_page >= d.total;
  } catch (err) {
    console.error(err);
  }
}

async function loadTimeChart() {
  try {
    const d = await api("/api/stats");
    lineChart("ch-hist-time", d.over_time.days, [
      { label: "Readings", data: d.over_time.readings, tension: 0.3 },
      { label: "Anomalies", data: d.over_time.anomalies, tension: 0.3,
        borderColor: "#dc2626", backgroundColor: "rgba(220,38,38,0.15)" },
    ], { scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } });
  } catch (err) { console.error(err); }
}

document.getElementById("btn-apply").addEventListener("click", () => { PAGE = 1; loadHistory(); });
document.getElementById("btn-reset").addEventListener("click", () => {
  ["q-search", "q-potable", "q-anomaly", "q-from"].forEach(id =>
    document.getElementById(id).value = "");
  PAGE = 1; loadHistory();
});
document.getElementById("btn-prev").addEventListener("click", () => { PAGE--; loadHistory(); });
document.getElementById("btn-next").addEventListener("click", () => { PAGE++; loadHistory(); });

loadHistory();
loadTimeChart();
