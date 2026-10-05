/* Dashboard page: fetch /api/stats and draw graphs 24-27 + recent table.
   Simple polling keeps the page fresh without websockets. */
"use strict";

async function loadStats() {
  try {
    const d = await api("/api/stats");
    document.getElementById("stat-readings").textContent = d.totals.readings;
    document.getElementById("stat-potable").textContent = d.totals.potable_pct + "%";
    document.getElementById("stat-anomaly").textContent = d.totals.anomaly_pct + "%";
    document.getElementById("stat-notpotable").textContent =
      (100 - d.totals.potable_pct).toFixed(1) + "%";

    // Graph 24
    doughnut("ch-class", d.class_balance.labels, d.class_balance.values,
             ["#15803d", "#b91c1c"]);

    // Graph 25
    lineChart("ch-time", d.over_time.days, [
      { label: "Readings", data: d.over_time.readings, tension: 0.3 },
      { label: "Anomalies", data: d.over_time.anomalies, tension: 0.3,
        borderColor: "#dc2626", backgroundColor: "rgba(220,38,38,0.15)" },
    ], { scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } });

    // Graph 26: averages vs guideline limits (dashed limit bars)
    const datasets = [
      { label: "Average of readings", data: d.param_comparison.averages, backgroundColor: "#0e7490" },
      { label: "Guideline limit", data: d.param_comparison.limits,
        backgroundColor: "rgba(220,38,38,0.45)" },
    ];
    barChart("ch-params", d.param_comparison.labels, datasets, {
      scales: { y: { type: "logarithmic", beginAtZero: true } },
    });

    // Graph 27: pH histogram with 6.5-8.5 band
    const bins = d.ph_distribution.bin_centers.map((c, i) => ({
      lo: d.ph_distribution.bin_edges[i], hi: d.ph_distribution.bin_edges[i + 1],
    }));
    const colors = bins.map(b => (b.lo >= 6.5 && b.hi <= 8.5) ? "#15803d" : "#94a3b8");
    barChart("ch-ph", d.ph_distribution.bin_centers.map(c => c.toFixed(1)), [
      { label: "Readings", data: d.ph_distribution.counts, backgroundColor: colors },
    ], { plugins: { legend: { display: false } } });

    // Recent table
    const tb = document.querySelector("#recent-table tbody");
    tb.innerHTML = d.recent.map(r => `
      <tr>
        <td>${r.id}</td>
        <td>${(r.created_at || "").replace("T", " ").slice(0, 16)}</td>
        <td><span class="badge muted">${r.source}</span></td>
        <td>${r.ph ?? "-"}</td>
        <td>${r.potability === 1
          ? '<span class="badge good">potable</span>'
          : '<span class="badge bad">not potable</span>'}</td>
        <td>${(100 * r.probability).toFixed(1)}%</td>
        <td>${r.is_anomaly ? '<span class="badge warn">anomaly</span>' : "-"}</td>
      </tr>`).join("");
  } catch (err) {
    console.error("stats failed:", err);
  }
}

loadStats();
setInterval(loadStats, 10000); // simple polling (no websockets/alerts)
