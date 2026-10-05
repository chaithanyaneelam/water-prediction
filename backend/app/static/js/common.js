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
  if (body === null) {
    // A 200 without JSON would crash callers later - surface it clearly here.
    throw new Error("Server returned an unexpected (non-JSON) response. Please retry.");
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

/* Nav auth area: shows the logged-in user (or a Login link) on every page. */
async function waternetInitAuthArea() {
  const area = document.getElementById("auth-area");
  if (!area) return;
  try {
    const me = await fetch("/api/auth/me").then(r => r.json());
    if (me.user) {
      area.innerHTML =
        `<span class="badge muted">${me.user.email || "account"}</span>` +
        `<button class="btn secondary" id="nav-logout">Logout</button>`;
      document.getElementById("nav-logout").addEventListener("click", async () => {
        await fetch("/api/auth/logout", { method: "POST" });
        location.href = "/";
      });
    } else {
      area.innerHTML = `<a class="btn secondary" href="/login">Login / Register</a>`;
    }
  } catch (e) { /* offline: leave empty */ }
}

document.addEventListener("DOMContentLoaded", () => {
  waternetInitTheme();
  waternetChartDefaults();
  waternetInitNav();
  waternetInitAuthArea();
});

/* Small shared helpers for charts built on the page scripts. */
function wnEsc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

/* Delivery confirmation line shown after a prediction button is clicked.
   `n` is the `notification` summary attached by the backend to the result. */
function wnNotifyLine(n) {
  if (!n) return "";
  const dest = wnEsc(n.destination);
  const via = n.channel === "sms" ? "SMS" : "email";
  const icon = n.channel === "sms" ? "📱" : "📧";
  if (n.status === "sent") {
    return `<p class="note" style="margin-top:0.6rem;">${icon} Result sent by ${via} to <b>${dest}</b>.</p>`;
  }
  if (n.status === "failed") {
    return `<p class="note" style="margin-top:0.6rem;">${icon} Could not send by ${via} to
      <b>${dest}</b> - ${wnEsc(n.error || "unknown error")}. The message is stored and can be retried.</p>`;
  }
  return `<p class="note" style="margin-top:0.6rem;">${icon} Result for <b>${dest}</b> kept in the
    outbox (delivery service not configured).</p>`;
}

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
