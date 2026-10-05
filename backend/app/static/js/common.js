/* WATERNET shared frontend helpers (no frameworks). */
"use strict";

/* ---------------- theme (light/dark) ---------------- */
function waternetApplyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("waternet-theme", theme); } catch (e) { /* private mode */ }
  // Recolor existing charts for the new theme.
  if (window.Chart) {
    const grid = getComputedStyle(document.documentElement).getPropertyValue("--border").trim();
    const text = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
    Chart.defaults.color = text;
    Chart.defaults.borderColor = grid;
    Object.values(Chart.instances).forEach(ch => ch.update());
  }
}

function waternetInitTheme() {
  let theme = "light";
  try { theme = localStorage.getItem("waternet-theme") || "light"; } catch (e) {}
  waternetApplyTheme(theme);
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.textContent = theme === "dark" ? "Light mode" : "Dark mode";
    btn.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      waternetApplyTheme(next);
      btn.textContent = next === "dark" ? "Light mode" : "Dark mode";
    });
  }
}

/* ---------------- Chart.js defaults ---------------- */
function waternetChartDefaults() {
  if (!window.Chart) return;
  const grid = getComputedStyle(document.documentElement).getPropertyValue("--border").trim();
  const text = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
  Chart.defaults.color = text;
  Chart.defaults.borderColor = grid;
  Chart.defaults.font.family = "Segoe UI, system-ui, sans-serif";
}

const WN_COLORS = ["#0e7490", "#f59e0b", "#7c3aed", "#16a34a", "#dc2626", "#64748b"];

function wnDestroy(id) {
  const el = document.getElementById(id);
  if (el && el._chart) { el._chart.destroy(); el._chart = null; }
}

/* ---------------- API helper ---------------- */
async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) {
    const msg = (body && (body.error || (body.errors || []).join(", "))) || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

/* ---------------- page chrome ---------------- */
function waternetInitNav() {
  const links = document.querySelectorAll(".navbar nav a");
  const here = location.pathname.replace(/\/$/, "") || "/";
  links.forEach(a => {
    const href = a.getAttribute("href").replace(/\/$/, "") || "/";
    if (href === here) a.classList.add("active");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  waternetInitTheme();
  waternetChartDefaults();
  waternetInitNav();
});

/* Small shared helpers for charts built on the page scripts. */
function doughnut(canvasId, labels, values, colors) {
  wnDestroy(canvasId);
  const ctx = document.getElementById(canvasId);
  ctx._chart = new Chart(ctx, {
    type: "doughnut",
    data: { labels, datasets: [{ data: values, backgroundColor: colors || WN_COLORS }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } },
  });
}

function barChart(canvasId, labels, datasets, opts) {
  wnDestroy(canvasId);
  opts = opts || {};
  const ctx = document.getElementById(canvasId);
  ctx._chart = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: Object.assign({
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } },
      scales: { y: { beginAtZero: true } },
    }, opts),
  });
}

function lineChart(canvasId, labels, datasets, opts) {
  wnDestroy(canvasId);
  opts = opts || {};
  const ctx = document.getElementById(canvasId);
  ctx._chart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: Object.assign({
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } },
    }, opts),
  });
}
