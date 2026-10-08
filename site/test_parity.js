/* Node-side parity test: run src/inference.js (the EXACT code the Worker runs)
   against the exported model JSONs on 120 real Kaggle rows, and compare with
   the real sklearn/xgboost/isolation-forest predictions recorded by Python
   (site/expected_rows.json, generated from the trained .joblib pipelines). */
const fs = require("fs");
const path = require("path");
const { xgbPredictProba, iforestScore } = require("./src/inference.js");

const MODELS = path.join(__dirname, "public", "static", "models");
const xgb = JSON.parse(fs.readFileSync(path.join(MODELS, "xgb_potability.json"), "utf8"));
const ifo = JSON.parse(fs.readFileSync(path.join(MODELS, "iforest.json"), "utf8"));
const meta = JSON.parse(fs.readFileSync(path.join(MODELS, "meta.json"), "utf8"));

const expected = JSON.parse(fs.readFileSync(path.join(__dirname, "expected_rows.json"), "utf8"));
console.log(`rows: ${expected.length}`);

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log(`PASS ${name} ${detail || ""}`);
  else { console.error(`FAIL ${name} ${detail || ""}`); failures++; }
}

/* ---- xgboost: JS vs real sklearn probas ---- */
let maxDiff = 0, labelDiff = 0;
for (const r of expected) {
  const js = xgbPredictProba(xgb, r);
  const sk = r._proba;
  maxDiff = Math.max(maxDiff, Math.abs(js - sk));
  if ((js >= 0.5 ? 1 : 0) !== (sk >= 0.5 ? 1 : 0)) labelDiff++;
}
check("xgboost max |proba diff| < 1e-5", maxDiff < 1e-5, `max=${maxDiff.toExponential(2)}`);
check("xgboost label diffs == 0", labelDiff === 0, `diffs=${labelDiff}`);

/* ---- isolation forest: JS vs real sklearn scores/flags ---- */
let maxSDiff = 0, flagDiff = 0;
for (const r of expected) {
  const row = {};
  for (const [k, v] of Object.entries(r)) if (!k.startsWith("_")) row[k] = v;
  const { score, isAnomaly } = iforestScore(ifo, row);
  maxSDiff = Math.max(maxSDiff, Math.abs(score - r._score));
  if ((isAnomaly ? 1 : 0) !== r._anom) flagDiff++;
}
check("iforest max |score diff| < 1e-9", maxSDiff < 1e-9, `max=${maxSDiff.toExponential(2)}`);
check("iforest flag mismatches == 0", flagDiff === 0, `diffs=${flagDiff}`);

/* ---- pH median imputation path ---- */
const ph = Math.round(Math.min(Math.max(meta.ph_median, 0), 14) * 100) / 100;
check("ph median imputation", ph === 7.0 && meta.ph_beats_baseline === false, `ph=${ph}`);

/* ---- sanity: a plausible sample predicts sensibly ---- */
const sample = {
  ph: 7.0, Hardness: 195.97, Solids: 20507, Chloramines: 7.3, Sulfate: 368.5,
  Conductivity: 564.3, Organic_carbon: 10.4, Trihalomethanes: 86.99, Turbidity: 2.96,
};
const sp = xgbPredictProba(xgb, sample);
const sf = iforestScore(ifo, sample);
console.log(`sample: potability_proba=${sp.toFixed(4)}, anomaly_score=${sf.score.toFixed(4)}, flag=${sf.isAnomaly}`);
check("sample proba in (0,1)", sp > 0 && sp < 1);
check("sample score in (-1,1)", sf.score > -1 && sf.score < 1);

process.exit(failures ? 1 : 0);
