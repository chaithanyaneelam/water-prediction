/* Bulk upload: POST the CSV to /api/predict/bulk, render summary + table + download. */
"use strict";

let BULK_RESULTS = [];

document.getElementById("btn-upload").addEventListener("click", async () => {
  const errBox = document.getElementById("bulk-errors");
  errBox.classList.add("hidden");
  const fileInput = document.getElementById("csv-file");
  const text = document.getElementById("csv-text").value.trim();

  try {
    let res;
    if (fileInput.files.length) {
      const fd = new FormData();
      fd.append("file", fileInput.files[0]);
      res = await api("/api/predict/bulk", { method: "POST", body: fd });
    } else if (text) {
      res = await api("/api/predict/bulk", {
        method: "POST",
        headers: { "Content-Type": "text/csv" },
        body: text,
      });
    } else {
      throw new Error("Choose a CSV file or paste CSV text first.");
    }
    BULK_RESULTS = res.results || [];
    renderBulk(res);
  } catch (err) {
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
  }
});

function renderBulk(r) {
  document.getElementById("bulk-results").classList.remove("hidden");
  document.getElementById("bulk-total").textContent = r.total_rows;
  document.getElementById("bulk-ok").textContent = r.predicted;
  document.getElementById("bulk-fail").textContent = r.failed_rows;

  doughnut("ch-bulk-class", ["Potable", "Not potable"],
    [r.potable, r.not_potable], ["#15803d", "#b91c1c"]);
  doughnut("ch-bulk-anom", ["Normal", "Anomaly"],
    [r.predicted - r.anomalies, r.anomalies], ["#64748b", "#f59e0b"]);

  const tb = document.querySelector("#bulk-table tbody");
  tb.innerHTML = r.results.map(x => `
    <tr>
      <td>${x.row}</td>
      <td>${x.ph ?? "-"}</td>
      <td>${x.Hardness ?? "-"}</td><td>${x.Solids ?? "-"}</td>
      <td>${x.Chloramines ?? "-"}</td><td>${x.Sulfate ?? "-"}</td>
      <td>${x.Conductivity ?? "-"}</td><td>${x.Organic_carbon ?? "-"}</td>
      <td>${x.Trihalomethanes ?? "-"}</td><td>${x.Turbidity ?? "-"}</td>
      <td>${x.potability === 1
        ? '<span class="badge good">potable</span>'
        : '<span class="badge bad">not potable</span>'}</td>
      <td>${(100 * x.probability).toFixed(1)}%</td>
      <td>${x.anomaly.flag ? '<span class="badge warn">flagged</span>' : "-"}</td>
    </tr>`).join("") +
    (r.errors || []).map(e =>
      `<tr><td colspan="13"><span class="badge bad">row ${e.row} failed:</span>
       ${e.errors.join(", ")}</td></tr>`).join("");
}

/* Client-side CSV download of exactly what the API returned. */
document.getElementById("btn-download").addEventListener("click", () => {
  if (!BULK_RESULTS.length) return;
  const cols = ["row", "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
    "potability", "potability_label", "probability", "ph_filled_by"];
  const lines = [cols.join(",")];
  BULK_RESULTS.forEach(x => {
    lines.push(cols.map(c => (x[c] === null || x[c] === undefined) ? "" : x[c]).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "waternet_bulk_results.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});
