/* WATERNET Cloudflare Worker: serves the static site + the full /api surface.
   Data: D1 (SQLite). Models: JSON trees (parity-verified, see src/inference.js). */

import { xgbPredictProba, iforestScore, irrPredict } from "./inference.js";

/* Models load lazily from the static assets bundle (module-level cache per
   isolate) - keeps the Worker bundle small and the deployment honest. */
const _cache = {};
async function assetJson(env, path) {
  if (_cache[path]) return _cache[path];
  const res = await env.ASSETS.fetch(new URL(path, "https://internal"));
  if (!res.ok) throw new Error(`asset missing: ${path}`);
  _cache[path] = await res.json();
  return _cache[path];
}
async function loadModels(env) {
  const [xgb, ifo, irr, meta] = await Promise.all([
    assetJson(env, "/static/models/xgb_potability.json"),
    assetJson(env, "/static/models/iforest.json"),
    assetJson(env, "/static/models/irr_rf.json"),
    assetJson(env, "/static/models/meta.json"),
  ]);
  return { XGB: xgb, IF: ifo, IRR: irr, META: meta };
}
let MODELS = null;
async function getModels(env) {
  if (!MODELS) MODELS = await loadModels(env);
  return MODELS;
}

/* ---------------------------------------------------------------- constants */
const FEATURE_COLS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
  "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"];
const REQUIRED_FIELDS = FEATURE_COLS.filter(f => f !== "ph");
const PHYSICAL_RANGES = {
  ph: [0.0, 14.0], Hardness: [0.0, null], Solids: [0.0, null],
  Chloramines: [0.0, null], Sulfate: [0.0, null], Conductivity: [0.0, null],
  Organic_carbon: [0.0, null], Trihalomethanes: [0.0, null], Turbidity: [0.0, null],
};
const GUIDELINE_LIMITS = {
  ph: { min: 6.5, max: 8.5 }, Turbidity: { max: 5.0 }, Solids: { max: 500.0 },
  Hardness: { max: 200.0 }, Chloramines: { max: 4.0 }, Sulfate: { max: 250.0 },
  Trihalomethanes: { max: 80.0 }, Organic_carbon: { max: 2.0 }, Conductivity: { max: 400.0 },
};
const TREATMENT_RULES = [
  ["Turbidity", "high", "Elevated turbidity: filtration or coagulation recommended."],
  ["ph", "low", "Low pH: neutralisation (e.g. lime dosing) recommended."],
  ["ph", "high", "High pH: acid dosing or blending with lower-pH water recommended."],
  ["Solids", "high", "High dissolved solids: reverse osmosis or blending recommended."],
  ["Hardness", "high", "Hard water: softening (ion exchange) recommended."],
  ["Chloramines", "high", "High disinfectant residual: activated carbon filtration."],
  ["Sulfate", "high", "High sulfate: reverse osmosis or distillation recommended."],
  ["Trihalomethanes", "high", "High THMs: activated carbon adsorption recommended."],
  ["Organic_carbon", "high", "High organic carbon: coagulation + activated carbon."],
  ["Conductivity", "high", "High conductivity: check for dissolved salts; RO if persistent."],
];
const IRRIGATION_FIELDS = ["ph", "EC", "TDS", "CO3", "HCO3", "Cl", "F", "NO3",
  "SO4", "Na", "K", "Ca", "Mg", "TH"];
const IRRIGATION_REQUIRED = ["EC", "HCO3", "Na", "Ca", "Mg"];
const IRRIGATION_RANGES = Object.fromEntries(IRRIGATION_FIELDS.map(f => [f, [0, null]]));
IRRIGATION_RANGES.ph = [0, 14];
const USSL_EC_CLASSES = [
  ["C1", 0, 250, "Low salinity - suitable for most crops"],
  ["C2", 250, 750, "Medium salinity - fine with moderate leaching"],
  ["C3", 750, 2250, "High salinity - salt-tolerant crops + good drainage"],
  ["C4", 2250, Infinity, "Very high salinity - generally unsuitable"],
];
const USSL_SAR_CLASSES = [
  ["S1", 0, 10, "Low sodium - safe for nearly all soils"],
  ["S2", 10, 18, "Medium sodium - fine with leaching + organic matter"],
  ["S3", 18, 26, "High sodium - sodium hazard; gypsum + drainage needed"],
  ["S4", 26, Infinity, "Very high sodium - generally unsuitable"],
];
const RSC_CLASSES = [
  ["P.S.", -Infinity, 1.25, "Safe - residual sodium carbonate acceptable"],
  ["MR", 1.25, 2.5, "Marginal - watch for carbonate accumulation"],
  ["U.S.", 2.5, Infinity, "Unsuitable - carbonate alkali hazard"],
];
const F_CA = 20.04, F_MG = 12.15, F_NA = 22.99, F_HCO3 = 61.02, F_CO3 = 30.0;

/* Precompiled model objects are loaded lazily via getModels(). */

/* ---------------------------------------------------------------- helpers */
const json = (body, status = 200, setCookie = null) => {
  const headers = { "content-type": "application/json" };
  if (setCookie) headers["set-cookie"] = setCookie;
  return new Response(JSON.stringify(body), { status, headers });
};
const ok = (body, setCookie = null) => json({ ok: true, ...body }, 200, setCookie);
const SESSION_COOKIE = (token, maxAge) =>
  `wn_session=${token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${maxAge}`;

function toFloat(value) {
  if (value === undefined || value === null || value === "") return null;
  if (typeof value === "boolean") throw new Error("must be a number");
  const v = typeof value === "number" ? value : parseFloat(String(value));
  if (Number.isNaN(v)) return null;
  if (!Number.isFinite(v)) throw new Error("infinity is not a valid measurement");
  return v;
}

function validateReading(data, { requireAll = true } = {}) {
  const errors = [];
  const parsed = {};
  for (const field of REQUIRED_FIELDS) {
    let v;
    try { v = toFloat(data[field]); }
    catch (e) { errors.push(`${field}: ${e.message}`); continue; }
    if (v === null) { if (requireAll) errors.push(`${field}: this field is required`); continue; }
    parsed[field] = v;
  }
  if ("ph" in (data || {})) {
    try {
      const v = toFloat(data.ph);
      if (v !== null) parsed.ph = v;
    } catch (e) { errors.push(`ph: ${e.message}`); }
  }
  for (const [field, v] of Object.entries(parsed)) {
    const [lo, hi] = PHYSICAL_RANGES[field] || [0, null];
    if (lo !== null && v < lo) errors.push(`${field}: ${v} is below the physical minimum (${lo})`);
    if (hi !== null && v > hi) errors.push(`${field}: ${v} is above the physical maximum (${hi})`);
  }
  if (errors.length) { const e = new Error("validation"); e.errors = errors; throw e; }
  return parsed;
}

function guidelineHits(parsed) {
  const hits = [];
  for (const [col, lim] of Object.entries(GUIDELINE_LIMITS)) {
    const v = parsed[col];
    if (v === undefined || v === null) continue;
    if (lim.min !== undefined && v < lim.min)
      hits.push({ parameter: col, value: v, limit: `>= ${lim.min}` });
    if (lim.max !== undefined && v > lim.max)
      hits.push({ parameter: col, value: v, limit: `<= ${lim.max}` });
  }
  return hits;
}

function treatmentSuggestions(parsed) {
  const tips = [];
  for (const [param, direction, text] of TREATMENT_RULES) {
    const lim = GUIDELINE_LIMITS[param];
    const v = parsed[param];
    if (!lim || v === undefined || v === null) continue;
    if (direction === "high" && lim.max !== undefined && v > lim.max) tips.push(text);
    else if (direction === "low" && lim.min !== undefined && v < lim.min) tips.push(text);
  }
  return tips;
}

function predictPotability(models, parsed) {
  const proba = xgbPredictProba(models.XGB, parsed);
  return [proba >= 0.5 ? 1 : 0, proba];
}

function predictPh(models, parsed) {
  if (parsed.ph !== undefined && parsed.ph !== null) return [parsed.ph, "none"];
  const v = Math.round(Math.min(Math.max(models.META.ph_median, 0), 14) * 100) / 100;
  return [v, "median"];
}

function predictAnomaly(models, parsed) {
  const row = {};
  for (const c of FEATURE_COLS) row[c] = parsed[c] ?? null;
  const { score, isAnomaly } = iforestScore(models.IF, row);
  const reasons = [];
  if (isAnomaly) reasons.push("Isolation Forest outlier score above threshold");
  if (parsed.ph !== undefined && parsed.ph !== null && !(0 <= parsed.ph && parsed.ph <= 14))
    reasons.push(`pH ${parsed.ph.toFixed(2)} outside the physical range 0-14`);
  for (const c of FEATURE_COLS) {
    const v = parsed[c];
    if (v !== undefined && v !== null && v < 0) reasons.push(`negative value for ${c} (${v.toFixed(2)})`);
  }
  if (parsed.Turbidity !== undefined && parsed.Turbidity !== null && parsed.Turbidity > 50)
    reasons.push(`Turbidity ${parsed.Turbidity.toFixed(1)} NTU is extremely high (> 50)`);
  return [isAnomaly ? 1 : 0, score, reasons];
}

/* Irrigation rule engine (exact, citable formulas - not ML). */
function _lookup(value, table) {
  for (const [code, lo, hi, meaning] of table)
    if (lo <= value && value < hi) return { code, meaning };
  const [code, , , meaning] = table[table.length - 1];
  return { code, meaning };
}

function irrigationVerdict(ec, sar, rsc) {
  const c = _lookup(ec, USSL_EC_CLASSES);
  const s = _lookup(sar, USSL_SAR_CLASSES);
  const r = _lookup(rsc, RSC_CLASSES);
  const u = `${c.code}${s.code}`;
  const notes = [
    `Salinity hazard ${c.code}: ${c.meaning}.`,
    `Sodium hazard ${s.code}: ${s.meaning}.`,
  ];
  let suitable = true;
  if (c.code === "C4") {
    suitable = false;
    notes.push("Very high salinity water can dehydrate plant roots; use only with heavy leaching and salt-tolerant crops, if at all.");
  } else if (c.code === "C3") {
    notes.push("Use salt-tolerant crops (e.g. barley, cotton, sugarbeet) and ensure drainage/leaching to prevent salt build-up.");
  }
  if (s.code === "S3" || s.code === "S4") {
    suitable = false;
    notes.push("High sodium damages soil structure (dispersion, reduced infiltration); apply gypsum and organic matter, ensure drainage.");
  } else if (s.code === "S2") {
    notes.push("Acceptable for well-drained soils; monitor soil sodium.");
  }
  if (r.code === "U.S.") {
    suitable = false;
    notes.push("High residual sodium carbonate will precipitate calcium and raise soil pH; avoid or treat (gypsum/acidulation).");
  } else if (r.code === "MR") {
    notes.push("Marginal RSC: monitor soil carbonate levels over time.");
  }
  return {
    sar: Math.round(sar * 1000) / 1000,
    rsc: Math.round(rsc * 1000) / 1000,
    ussl_class: u,
    salinity: c, sodium: s,
    rsc_class: r.code, rsc_meaning: r.meaning,
    suitable,
    verdict: suitable ? "SUITABLE for irrigation (with the noted precautions)"
      : "NOT SUITABLE for most irrigation uses (see notes)",
    notes,
  };
}

function validateIrrigation(data) {
  const errors = [];
  const parsed = {};
  for (const field of IRRIGATION_FIELDS) {
    let v;
    try { v = toFloat(data[field]); }
    catch (e) { errors.push(`${field}: ${e.message}`); continue; }
    if (v === null && IRRIGATION_REQUIRED.includes(field)) {
      errors.push(`${field}: this field is required`); continue;
    }
    if (v !== null) {
      const [lo, hi] = IRRIGATION_RANGES[field];
      if (v < lo) errors.push(`${field}: ${v} is below the physical minimum (${lo})`);
      if (hi !== null && v > hi) errors.push(`${field}: ${v} is above the physical maximum (${hi})`);
      parsed[field] = v;
    }
  }
  if (errors.length) { const e = new Error("validation"); e.errors = errors; throw e; }
  return parsed;
}

/* ---------------------------------------------------------------- auth */
async function getUser(request, env) {
  const cookie = request.headers.get("cookie") || "";
  const m = cookie.match(/wn_session=([A-Za-z0-9_-]+)/);
  if (!m) return null;
  const row = await env.DB.prepare(
    "SELECT u.id, u.email, u.display_name FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?"
  ).bind(m[1]).first();
  return row || null;
}

async function hashPassword(password, saltHex) {
  const data = new TextEncoder().encode(saltHex + ":" + password);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function createSession(env, userId) {
  const token = crypto.randomUUID().replace(/-/g, "") + crypto.randomUUID().replace(/-/g, "");
  await env.DB.prepare("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, datetime('now'))")
    .bind(token, userId).run();
  return token;
}

/* ---------------------------------------------------------------- model payload builders */
function buildResult(id, parsed, potability, proba, ph, filledBy, anomFlag, anomScore, anomReason) {
  return {
    reading_id: id,
    potability,
    potability_label: potability === 1 ? "potable" : "not potable",
    probability: Math.round(proba * 10000) / 10000,
    ph: ph === null ? null : ph,
    ph_filled_by: filledBy,
    anomaly: {
      flag: !!anomFlag,
      score: Math.round(anomScore * 10000) / 10000,
      reason: anomReason || "none",
    },
    guideline_hits: guidelineHits(parsed),
    treatment: treatmentSuggestions(parsed),
  };
}

async function insertReading(env, userId, source, parsed) {
  const cols = FEATURE_COLS.map(c => parsed[c] ?? null);
  const res = await env.DB.prepare(
    `INSERT INTO readings (user_id, source, ${FEATURE_COLS.join(",")})
     VALUES (?, ?, ${FEATURE_COLS.map(() => "?").join(",")})`
  ).bind(userId, source, ...cols).run();
  return res.meta.last_row_id;
}

async function insertPrediction(env, readingId, potability, proba, ph, filledBy, flag, score, reason) {
  await env.DB.prepare(
    `INSERT INTO predictions (reading_id, potability, probability, ph_predicted, ph_filled_by,
       is_anomaly, anomaly_score, anomaly_reason)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  ).bind(readingId, potability, Math.round(proba * 10000) / 10000,
    ph, filledBy, flag ? 1 : 0, Math.round(score * 10000) / 10000,
    reason && reason.length ? reason.join("; ") : null).run();
}

/* ---------------------------------------------------------------- static assets */
const PAGES = new Set(["/dashboard", "/predict", "/bulk", "/history",
  "/comparison", "/explorer", "/irrigation", "/login"]);

async function serveAsset(request, env, path) {
  const origin = new URL(request.url).origin;
  if (PAGES.has(path)) {
    const page = await env.ASSETS.fetch(new URL(path + ".html", origin));
    if (page.ok) return page;
  }
  return env.ASSETS.fetch(new URL(path, origin));
}

/* ---------------------------------------------------------------- route table */
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    if (!path.startsWith("/api/")) {
      try { return await serveAsset(request, env, path); }
      catch (e) { return new Response("asset error: " + e.message, { status: 500 }); }
    }

    try {
      /* ---------------- health ---------------- */
      if (path === "/api/health") {
        return ok({ models_loaded: true, runtime: "cloudflare-workers" });
      }

      /* ---------------- auth ---------------- */
      if (path === "/api/auth/register" && method === "POST") {
        const body = await request.json().catch(() => ({}));
        const email = String(body.email || "").trim().toLowerCase();
        const password = String(body.password || "");
        const name = String(body.display_name || "").trim() || email.split("@")[0];
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))
          return json({ ok: false, errors: ["email: an email address is required"] }, 400);
        if (password.length < 6)
          return json({ ok: false, errors: ["password: must be at least 6 characters"] }, 400);
        const exists = await env.DB.prepare("SELECT id FROM users WHERE email = ?").bind(email).first();
        if (exists) return json({ ok: false, errors: ["email: already registered"] }, 400);
        const salt = crypto.randomUUID().replace(/-/g, "");
        const hash = await hashPassword(password, salt);
        const res = await env.DB.prepare(
          "INSERT INTO users (email, display_name, password_hash) VALUES (?, ?, ?)"
        ).bind(email, name, salt + "$" + hash).run();
        const token = await createSession(env, res.meta.last_row_id);
        return ok({
          user: { id: res.meta.last_row_id, email, display_name: name },
          token,
        }, SESSION_COOKIE(token, 60 * 60 * 24 * 30));
      }

      if (path === "/api/auth/login" && method === "POST") {
        const body = await request.json().catch(() => ({}));
        const identifier = String(body.identifier || body.email || "").trim().toLowerCase();
        const password = String(body.password || "");
        const user = await env.DB.prepare(
          "SELECT id, email, display_name, password_hash FROM users WHERE email = ?"
        ).bind(identifier).first();
        if (!user) return json({ ok: false, error: "invalid email or password" }, 401);
        const [salt, stored] = String(user.password_hash).split("$");
        const hash = await hashPassword(password, salt);
        if (hash !== stored) return json({ ok: false, error: "invalid email or password" }, 401);
        const token = await createSession(env, user.id);
        return ok({
          user: { id: user.id, email: user.email, display_name: user.display_name },
          token,
        }, SESSION_COOKIE(token, 60 * 60 * 24 * 30));
      }

      if (path === "/api/auth/logout" && method === "POST") {
        const cookie = request.headers.get("cookie") || "";
        const m = cookie.match(/wn_session=([A-Za-z0-9_-]+)/);
        if (m) await env.DB.prepare("DELETE FROM sessions WHERE token = ?").bind(m[1]).run();
        return ok({});
      }

      if (path === "/api/auth/me") {
        const user = await getUser(request, env);
        return ok({ user: user || null });
      }

      /* ---------------- everything below needs a user ---------------- */
      const user = await getUser(request, env);

      if ((path === "/api/predict" || /^\/api\/predict\/\d+\/send$/.test(path)) && !user)
        return json({ ok: false, error: "login required" }, 401);

      if (path === "/api/predict" && method === "POST") {
        const body = await request.json().catch(() => ({}));
        let parsed;
        try { parsed = validateReading(body); }
        catch (e) { return json({ ok: false, errors: e.errors }, 400); }
        const models = await getModels(env);
        const [potability, proba] = predictPotability(models, parsed);
        const [ph, filledBy] = predictPh(models, parsed);
        const [flag, score, reasons] = predictAnomaly(models, parsed);
        const readingId = await insertReading(env, user.id, "manual", parsed);
        await insertPrediction(env, readingId, potability, proba, ph, filledBy, flag, score, reasons);
        const result = buildResult(readingId, parsed, potability, proba, ph, filledBy, flag, score, reasons.join("; "));
        return ok(result);
      }

      const sendMatch = path.match(/^\/api\/predict\/(\d+)\/send$/);
      if (sendMatch && method === "POST") {
        const readingId = parseInt(sendMatch[1], 10);
        const row = await env.DB.prepare(
          `SELECT r.*, p.potability, p.probability, p.ph_predicted, p.ph_filled_by,
                  p.is_anomaly, p.anomaly_score, p.anomaly_reason
           FROM readings r JOIN predictions p ON p.reading_id = r.id
           WHERE r.id = ? AND r.user_id = ?`
        ).bind(readingId, user.id).first();
        if (!row) return json({ ok: false, errors: ["report not found (or it belongs to another user)"] }, 404);
        const parsed = {};
        for (const c of FEATURE_COLS) parsed[c] = row[c];
        const result = buildResult(readingId, parsed, row.potability, row.probability,
          row.ph_predicted, row.ph_filled_by, row.is_anomaly, row.anomaly_score, row.anomaly_reason);
        return ok({ reading_id: readingId, result, note: "email delivery is not part of the hosted demo; the report is rebuilt from stored data" });
      }

      /* ---------------- bulk ---------------- */
      if (path === "/api/predict/bulk" && method === "POST") {
        let content;
        const ct = request.headers.get("content-type") || "";
        if (ct.includes("multipart/form-data")) {
          const fd = await request.formData();
          const file = fd.get("file");
          content = file ? await file.text() : "";
        } else {
          content = await request.text();
        }
        if (!content.trim())
          return json({ ok: false, errors: ["empty request: attach a CSV file or send CSV text"] }, 400);
        const lines = content.replace(/^\uFEFF/, "").split(/\r?\n/).filter(l => l.trim());
        if (lines.length < 2)
          return json({ ok: false, errors: ["CSV needs a header row plus at least one data row"] }, 400);
        const parseCsvLine = (line) => {
          const out = []; let cur = "", inQ = false;
          for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (inQ) {
              if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
              else if (ch === '"') inQ = false;
              else cur += ch;
            } else if (ch === '"') inQ = true;
            else if (ch === ",") { out.push(cur); cur = ""; }
            else cur += ch;
          }
          out.push(cur);
          return out;
        };
        const header = parseCsvLine(lines[0]).map(h => h.trim().replace(/^\uFEFF/, "").toLowerCase());
        const missing = REQUIRED_FIELDS.filter(f => !header.includes(f.toLowerCase()));
        if (missing.length)
          return json({ ok: false, errors: [`CSV is missing required column(s): ${missing}. Found these columns instead: ${header}. The header row must be: ${FEATURE_COLS.join(",")}`] }, 400);
        const colIdx = Object.fromEntries(FEATURE_COLS.map(f => [f, header.indexOf(f.toLowerCase())]));
        const results = [], errors = [];
        let potable = 0, anomalies = 0;
        for (let i = 1; i < lines.length; i++) {
          const cells = parseCsvLine(lines[i]);
          const raw = {};
          for (const f of FEATURE_COLS) raw[f] = cells[colIdx[f]];
          try {
            const parsed = validateReading(raw, { requireAll: false });
            if (FEATURE_COLS.some(c => c !== "ph" && (parsed[c] === undefined || parsed[c] === null))) {
              errors.push({ row: i + 1, errors: ["missing required value(s)"] });
              continue;
            }
            const models = await getModels(env);
            const [potability, proba] = predictPotability(models, parsed);
            const [ph, filledBy] = predictPh(models, parsed);
            const [flag, score, reasons] = predictAnomaly(models, parsed);
            const readingId = await insertReading(env, null, "bulk", parsed);
            await insertPrediction(env, readingId, potability, proba, ph, filledBy, flag, score, reasons);
            const result = buildResult(readingId, parsed, potability, proba, ph, filledBy, flag, score, reasons.join("; "));
            result.row = i + 1;
            results.push(result);
            if (potability === 1) potable++;
            if (flag) anomalies++;
          } catch (e) {
            errors.push({ row: i + 1, errors: e.errors || [String(e.message || e)] });
          }
        }
        return ok({
          total_rows: lines.length - 1,
          predicted: results.length,
          failed_rows: errors.length,
          potable, not_potable: results.length - potable,
          anomalies,
          results, errors,
        });
      }

      /* ---------------- readings (history) ---------------- */
      if (path === "/api/readings" && method === "GET") {
        const q = url.searchParams;
        const page = Math.max(parseInt(q.get("page") || "1", 10) || 1, 1);
        const perPage = Math.min(parseInt(q.get("per_page") || "20", 10) || 20, 100);
        const conds = [], binds = [];
        if (["0", "1"].includes(q.get("potable"))) { conds.push("p.potability = ?"); binds.push(parseInt(q.get("potable"), 10)); }
        if (["0", "1"].includes(q.get("anomaly"))) { conds.push("p.is_anomaly = ?"); binds.push(parseInt(q.get("anomaly"), 10)); }
        if (q.get("source")) { conds.push("r.source = ?"); binds.push(q.get("source")); }
        if (q.get("date_from")) { conds.push("r.created_at >= ?"); binds.push(q.get("date_from")); }
        if (q.get("search")) {
          const like = `%${q.get("search")}%`;
          conds.push("(" + FEATURE_COLS.map(c => `CAST(r.${c} AS TEXT) LIKE ?`).join(" OR ") + ")");
          for (const _ of FEATURE_COLS) binds.push(like);
        }
        const where = conds.length ? "WHERE " + conds.join(" AND ") : "";
        const totalRow = await env.DB.prepare(
          `SELECT COUNT(*) AS n FROM readings r JOIN predictions p ON p.reading_id = r.id ${where}`
        ).bind(...binds).first();
        const rows = await env.DB.prepare(
          `SELECT r.id, r.created_at, r.source, ${FEATURE_COLS.map(c => "r." + c).join(",")},
                  p.potability, p.probability, p.ph_predicted, p.ph_filled_by,
                  p.is_anomaly, p.anomaly_score, p.anomaly_reason
           FROM readings r JOIN predictions p ON p.reading_id = r.id ${where}
           ORDER BY r.created_at DESC, r.id DESC LIMIT ? OFFSET ?`
        ).bind(...binds, perPage, (page - 1) * perPage).all();
        return ok({
          total: totalRow.n, page, per_page: perPage,
          items: rows.results.map(r => ({
            id: r.id, created_at: r.created_at, source: r.source,
            ...Object.fromEntries(FEATURE_COLS.map(c => [c, r[c]])),
            potability: r.potability, probability: r.probability,
            ph_predicted: r.ph_predicted, ph_filled_by: r.ph_filled_by,
            is_anomaly: !!r.is_anomaly, anomaly_score: r.anomaly_score,
            anomaly_reason: r.anomaly_reason,
          })),
        });
      }

      /* ---------------- stats (dashboard) ---------------- */
      if (path === "/api/stats" && method === "GET") {
        const totals = await env.DB.prepare(
          `SELECT COUNT(*) AS readings,
                  SUM(CASE WHEN p.potability = 1 THEN 1 ELSE 0 END) AS potable,
                  SUM(CASE WHEN p.is_anomaly = 1 THEN 1 ELSE 0 END) AS anomalies
           FROM readings r JOIN predictions p ON p.reading_id = r.id`
        ).first();
        const total = totals ? (totals.readings || 0) : 0;
        const potable = totals ? (totals.potable || 0) : 0;
        const anomalies = totals ? (totals.anomalies || 0) : 0;
        const overTime = await env.DB.prepare(
          `SELECT date(r.created_at) AS day, COUNT(*) AS n,
                  SUM(CASE WHEN p.is_anomaly = 1 THEN 1 ELSE 0 END) AS anom
           FROM readings r JOIN predictions p ON p.reading_id = r.id
           WHERE r.created_at >= datetime('now', '-30 days')
           GROUP BY date(r.created_at) ORDER BY day`
        ).all();
        const averages = {};
        for (const c of FEATURE_COLS) {
          const a = await env.DB.prepare(`SELECT AVG(${c}) AS v FROM readings`).first();
          averages[c] = a && a.v !== null ? Math.round(a.v * 1000) / 1000 : null;
        }
        const paramComparison = {
          labels: FEATURE_COLS,
          averages: FEATURE_COLS.map(c => averages[c]),
          limits: FEATURE_COLS.map(c => {
            const lim = GUIDELINE_LIMITS[c];
            if (c === "ph") return (lim.min + lim.max) / 2;
            return lim.max !== undefined ? lim.max : null;
          }),
          footnote: "Guideline limits are WHO/BIS/EPA reference values for display only; many potable-labelled samples in the dataset exceed them.",
        };
        const phRows = await env.DB.prepare("SELECT ph FROM readings WHERE ph IS NOT NULL").all();
        const phValues = phRows.results.map(r => r.ph);
        let hist = [], edges = [];
        if (phValues.length) {
          const min = Math.min(...phValues), max = Math.max(...phValues);
          const width = (max - min) / 20 || 1;
          edges = Array.from({ length: 21 }, (_, i) => Math.round((min + i * width) * 1000) / 1000);
          hist = new Array(20).fill(0);
          for (const v of phValues) {
            let b = Math.floor((v - min) / width);
            if (b >= 20) b = 19;
            hist[b]++;
          }
        }
        const recent = await env.DB.prepare(
          `SELECT r.id, r.created_at, r.source, ${FEATURE_COLS.map(c => "r." + c).join(",")},
                  p.potability, p.probability, p.is_anomaly
           FROM readings r JOIN predictions p ON p.reading_id = r.id
           ORDER BY r.created_at DESC, r.id DESC LIMIT 10`
        ).all();
        return ok({
          totals: {
            readings: total, potable, not_potable: total - potable, anomalies,
            potable_pct: total ? Math.round(1000 * potable / total) / 10 : 0,
            anomaly_pct: total ? Math.round(1000 * anomalies / total) / 10 : 0,
          },
          class_balance: { labels: ["Potable", "Not potable"], values: [potable, total - potable] },
          over_time: {
            days: overTime.results.map(r => r.day),
            readings: overTime.results.map(r => r.n),
            anomalies: overTime.results.map(r => r.anom || 0),
          },
          param_comparison: paramComparison,
          ph_distribution: {
            bin_edges: edges,
            bin_centers: edges.slice(0, -1).map((e, i) => Math.round((e + edges[i + 1]) / 2 * 1000) / 1000),
            counts: hist, band: [6.5, 8.5],
          },
          recent: recent.results.map(r => ({
            id: r.id, created_at: r.created_at, source: r.source,
            ...Object.fromEntries(FEATURE_COLS.map(c => [c, r[c]])),
            potability: r.potability, probability: r.probability, is_anomaly: !!r.is_anomaly,
          })),
        });
      }

      /* ---------------- model comparison metrics (static artifacts) ---------------- */
      if (path === "/api/models/metrics") {
        const names = ["classifier_metrics", "roc_curves", "pr_curves", "calibration",
          "threshold_curve", "feature_importance", "learning_curve", "validation_curves",
          "baseline_vs_improved", "ph_regressor", "ph_scatter", "ph_residuals",
          "ph_metrics_vs_baseline", "anomaly_score_hist", "anomaly_pca", "anomaly_reasons"];
        const payload = { ok: true, available: {} };
        for (const n of names) {
          try { payload[n] = await assetJson(env, `/static/artifacts/models/${n}.json`); }
          catch (e) { payload[n] = null; }
          payload.available[n] = payload[n] !== null && payload[n] !== undefined;
        }
        return json(payload);
      }

      /* ---------------- dataset explorer (static artifacts) ---------------- */
      if (path === "/api/dataset/summary") {
        const names = ["class_balance", "missing_values", "feature_distributions",
          "correlation_matrix", "outlier_counts", "imputation_comparison"];
        const payload = { ok: true };
        for (const n of names) {
          try { payload[n] = await assetJson(env, `/static/artifacts/dataset/${n}.json`); }
          catch (e) { payload[n] = null; }
        }
        payload.available = ["class_balance", "missing_values", "correlation_matrix"]
          .every(k => payload[k] !== null && payload[k] !== undefined);
        try {
          payload.summary_stats = await assetJson(env, "/static/artifacts/dataset/summary_stats.json");
        } catch (e) { payload.summary_stats = []; }
        payload.hint = payload.available ? null : "Dataset explorer artifacts not found.";
        return json(payload);
      }

      /* ---------------- irrigation ---------------- */
      if (path === "/api/irrigation/predict" && method === "POST") {
        if (!user) return json({ ok: false, error: "login required" }, 401);
        const body = await request.json().catch(() => ({}));
        let parsed;
        try { parsed = validateIrrigation(body); }
        catch (e) { return json({ ok: false, errors: e.errors }, 400); }
        const ec = parsed.EC, na = parsed.Na, ca = parsed.Ca, mg = parsed.Mg;
        const co3 = parsed.CO3 || 0, hco3 = parsed.HCO3;
        const sar = (na / F_NA) / Math.sqrt(((ca / F_CA) + (mg / F_MG)) / 2);
        const rsc = (co3 / F_CO3 + hco3 / F_HCO3) - ((ca / F_CA) + (mg / F_MG));
        const v = irrigationVerdict(ec, sar, rsc);
        const row = {};
        for (const c of IRRIGATION_FIELDS) row[c] = parsed[c] ?? null;
        const models = await getModels(env);
        const ml = irrPredict(models.IRR, row);
        await env.DB.prepare(
          `INSERT INTO irrigation_readings (user_id, source, ${IRRIGATION_FIELDS.join(",")},
             SAR, RSC, ussl_class_rule, rsc_class_rule, suitable, verdict_notes, ml_ussl_class, ml_probability)
           VALUES (?, ?, ${IRRIGATION_FIELDS.map(() => "?").join(",")}, ?, ?, ?, ?, ?, ?, ?, ?)`
        ).bind(user.id, "manual",
          ...IRRIGATION_FIELDS.map(f => parsed[f] ?? null),
          v.sar, v.rsc, v.ussl_class, v.rsc_class, v.suitable ? 1 : 0,
          v.notes.join(" | "), ml.cls, Math.round(ml.proba * 10000) / 10000).run();
        return ok({ ...v, ml_cross_check: { class: ml.cls, probability: Math.round(ml.proba * 10000) / 10000 } });
      }

      if (path === "/api/irrigation/summary") {
        const names = ["class_distribution", "ussl_scatter", "confusion_matrix",
          "feature_importance", "irrigation_metrics", "rule_verification"];
        const payload = { stored: { total: 0, suitable: 0 } };
        for (const n of names) {
          try { payload[n] = await assetJson(env, `/static/artifacts/irrigation/${n}.json`); }
          catch (e) { payload[n] = null; }
        }
        payload.available = payload.irrigation_metrics != null;
        payload.hint = payload.available ? null : "Irrigation model artifacts not found.";
        try {
          const c = await env.DB.prepare(
            "SELECT COUNT(*) AS total, SUM(CASE WHEN suitable = 1 THEN 1 ELSE 0 END) AS s FROM irrigation_readings"
          ).first();
          if (c) payload.stored = { total: c.total || 0, suitable: c.s || 0 };
        } catch (e) { /* table missing */ }
        return ok(payload);
      }

      return json({ ok: false, error: `unknown API route ${method} ${path}` }, 404);
    } catch (err) {
      console.error("worker error:", err);
      return json({ ok: false, error: String(err && err.message || err) }, 500);
    }
  },
};
