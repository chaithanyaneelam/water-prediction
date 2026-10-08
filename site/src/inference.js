/* WATERNET inference engine (Cloudflare Workers, no dependencies).
   Mirrors backend/ml/export_models_json.py + portable_parity.py, which verify
   against the real sklearn/xgboost predictions in Python before deployment:
   - xgboost: max |proba diff| 1.6e-7, 0 label diffs on 120 real rows
   - iforest: max |score diff| 1.1e-16, 0 flag mismatches
   - irrigation RF: 0 class mismatches on 80 real rows */

const eulerGamma = 0.5772156649015329;

function averagePathLength(n) {
  if (n <= 1) return 0.0;
  if (n === 2) return 1.0;
  return 2.0 * (Math.log(n - 1.0) + eulerGamma) - 2.0 * (n - 1.0) / n;
}

function transformRow(rowMap, features, tf) {
  let v = features.map(f => {
    const x = rowMap[f];
    return x === undefined || x === null ? NaN : x;
  });
  if (tf.impute) v = v.map((x, i) => (Number.isNaN(x) ? tf.impute[i] : x));
  if (tf.clip_lower) v = v.map((x, i) => Math.min(Math.max(x, tf.clip_lower[i]), tf.clip_upper[i]));
  if (tf.mean) v = v.map((x, i) => (x - tf.mean[i]) / tf.scale[i]);
  return v;
}

/* -------- xgboost (dumped trees, node/children/leaf nodes) -------- */
function xgbPredictProba(exportObj, rowMap) {
  const v = transformRow(rowMap, exportObj.features, exportObj.transform);
  const idx = new Map(exportObj.features.map((n, i) => [n, i]));
  let total = 0;
  for (const tree of exportObj.trees) {
    let node = tree;
    while (!("leaf" in node)) {
      let x = v[idx.get(node.split)];
      if (Number.isNaN(x)) {
        node = node.children.find(c => c.nodeid === node.missing);
      } else if (Math.fround(x) < Math.fround(node.split_condition)) {
        node = node.children.find(c => c.nodeid === node.yes);
      } else {
        node = node.children.find(c => c.nodeid === node.no);
      }
    }
    total += node.leaf;
  }
  return 1 / (1 + Math.exp(-total));
}

/* -------- isolation forest (flat arrays per tree) -------- */
function iforestScore(exportObj, rowMap) {
  const v = transformRow(rowMap, exportObj.features, exportObj.transform);
  const f32 = Math.fround;
  let depthSum = 0;
  for (const t of exportObj.trees) {
    let node = 0, depth = 0;
    while (t.f[node] !== -1) {
      const x = f32(v[t.f[node]]);
      node = x <= f32(t.t[node]) ? t.l[node] : t.r[node];
      depth++;
    }
    depthSum += depth + averagePathLength(t.n[node]);
  }
  const score = -(2 ** (-depthSum / exportObj.n_trees / exportObj.c_n));
  return { score, isAnomaly: score - exportObj.offset < 0 };
}

/* -------- irrigation random forest (flat arrays, class votes) -------- */
function irrPredict(exportObj, rowMap) {
  const v = transformRow(rowMap, exportObj.features, exportObj.transform);
  const f32 = Math.fround;
  const agg = new Array(exportObj.classes.length).fill(0);
  for (const t of exportObj.trees) {
    let node = 0;
    while (t.f[node] !== -1) {
      const x = f32(v[t.f[node]]);
      node = x <= f32(t.t[node]) ? t.l[node] : t.r[node];
    }
    const leaf = t.v[node];
    const s = leaf.reduce((a, b) => a + b, 0);
    for (let i = 0; i < leaf.length; i++) agg[i] += leaf[i] / s;
  }
  let best = 0;
  for (let i = 1; i < agg.length; i++) if (agg[i] > agg[best]) best = i;
  return { cls: exportObj.classes[best], proba: agg[best] };
}

module.exports = { transformRow, xgbPredictProba, iforestScore, irrPredict, averagePathLength };
