"""Export trained models to JSON for the WATERNET Cloudflare Worker site.

Writes to site/public/static/models/:
- meta.json            pH median + app decision
- xgb_potability.json  XGBoost pipeline (impute -> clip -> scale -> trees)
- iforest.json         Isolation Forest pipeline (impute -> clip -> scale -> trees)
- irr_rf.json          Irrigation RF pipeline (impute -> scale -> trees)

VERIFIES parity: re-implements in Python the exact arithmetic the Worker JS
will use (including the preprocessing transforms) and compares against the
real sklearn/xgboost predictions on real dataset rows. A mismatch aborts.
"""
import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ml.config import (  # noqa: E402
    DATASET_CSV, FEATURE_COLS, IRRIGATION_FEATURES,
)

SAVED = os.path.join(os.path.dirname(__file__), "saved_models")
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "site", "public", "static", "models"))
os.makedirs(OUT, exist_ok=True)

EULER_GAMMA = 0.5772156649015329


def f32(x: float) -> float:
    """float32 rounding, exactly like Math.fround in the Worker JS."""
    return float(np.float32(x))


def average_path_length(n: float) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + EULER_GAMMA) - 2.0 * (n - 1.0) / n


def export_transforms(pipe, cols):
    """Serialize impute/clip/scale parameters in `cols` order."""
    out = {}
    steps = getattr(pipe, "named_steps", {})
    imp = steps.get("impute") or steps.get("imputer")
    if imp is not None and hasattr(imp, "medians_"):
        out["impute"] = [float(imp.medians_[c]) for c in cols]
    clip = steps.get("clip")
    if clip is not None and hasattr(clip, "lower_"):
        out["clip_lower"] = [float(v) for v in clip.lower_]
        out["clip_upper"] = [float(v) for v in clip.upper_]
    sc = steps.get("scale")
    if sc is not None and hasattr(sc, "mean_"):
        out["mean"] = [float(v) for v in sc.mean_]
        out["scale"] = [float(v) for v in sc.scale_]
    return out


def transformed(row_map, cols, tf):
    """Apply impute -> clip -> scale exactly as the JS will."""
    v = [row_map.get(c) for c in cols]
    if "impute" in tf:
        v = [tf["impute"][i] if x is None else x for i, x in enumerate(v)]
    if "clip_lower" in tf:
        v = [min(max(x, tf["clip_lower"][i]), tf["clip_upper"][i])
             for i, x in enumerate(v)]
    if "mean" in tf:
        v = [(x - tf["mean"][i]) / tf["scale"][i] for i, x in enumerate(v)]
    return v


xgb_pipe = joblib.load(os.path.join(SAVED, "xgboost_potability.joblib"))
if_pipe = joblib.load(os.path.join(SAVED, "isolation_forest.joblib"))
irr_pipe = joblib.load(os.path.join(SAVED, "rf_irrigation.joblib"))
ph_pipe = joblib.load(os.path.join(SAVED, "rf_ph_regressor.joblib"))
ph_info_path = os.path.join(os.path.dirname(__file__), "outputs", "models", "ph_regressor.json")
ph_beats = False
if os.path.exists(ph_info_path):
    with open(ph_info_path, encoding="utf-8") as f:
        ph_beats = bool(json.load(f).get("beats_baseline", False))

ph_median = 7.0
imputer = getattr(ph_pipe, "named_steps", {}).get("impute")
if imputer is not None and hasattr(imputer, "medians_"):
    ph_median = float(imputer.medians_.get("ph", 7.0))

xgb_model = xgb_pipe.named_steps["model"]
if_model = if_pipe.named_steps["model"]
irr_model = irr_pipe.named_steps["model"]

# ----------------------------------------------------------------- xgboost dump
booster = xgb_model.get_booster()
if not booster.feature_names:
    booster.feature_names = FEATURE_COLS
dump = booster.get_dump(dump_format="json")
trees = [json.loads(d) for d in dump]
with open(os.path.join(OUT, "xgb_potability.json"), "w", encoding="utf-8") as f:
    json.dump({"features": FEATURE_COLS, "transform": export_transforms(xgb_pipe, FEATURE_COLS),
               "trees": trees}, f, separators=(",", ":"))

# ----------------------------------------------------------------- iforest
if_trees = []
for est in if_model.estimators_:
    t = est.tree_
    if_trees.append({
        "f": [-1 if v <= -2 else int(v) for v in t.feature.tolist()],
        "t": [float(v) for v in t.threshold],
        "l": t.children_left.tolist(),
        "r": t.children_right.tolist(),
        "n": t.n_node_samples.tolist(),
    })
with open(os.path.join(OUT, "iforest.json"), "w", encoding="utf-8") as f:
    json.dump({
        "features": FEATURE_COLS,
        "transform": export_transforms(if_pipe, FEATURE_COLS),
        "n_trees": len(if_trees),
        "c_n": average_path_length(int(if_model.max_samples_)),
        "offset": float(if_model.offset_),
        "trees": if_trees,
    }, f, separators=(",", ":"))

# ----------------------------------------------------------------- irrigation RF
irr_trees = []
for est in irr_model.estimators_:
    t = est.tree_
    irr_trees.append({
        "f": [-1 if v <= -2 else int(v) for v in t.feature.tolist()],
        "t": [float(v) for v in t.threshold],
        "l": t.children_left.tolist(),
        "r": t.children_right.tolist(),
        "v": t.value[:, 0, :].tolist(),
    })
with open(os.path.join(OUT, "irr_rf.json"), "w", encoding="utf-8") as f:
    json.dump({
        "features": IRRIGATION_FEATURES,
        "transform": export_transforms(irr_pipe, IRRIGATION_FEATURES),
        "classes": [str(c) for c in irr_model.classes_],
        "trees": irr_trees,
    }, f, separators=(",", ":"))

with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
    json.dump({"ph_median": ph_median, "ph_beats_baseline": ph_beats}, f)

for name in ("xgb_potability.json", "iforest.json", "irr_rf.json", "meta.json"):
    p = os.path.join(OUT, name)
    print(f"[export] {name}: {os.path.getsize(p) / 1024:.0f} KB")

# ----------------------------------------------------------------- PARITY CHECK
if os.environ.get("WN_SKIP_PARITY"):
    print("[export] WN_SKIP_PARITY set - exports written, parity skipped")
    sys.exit(0)


def xgb_predict_js(export, row_map):
    tf = export["transform"]
    v = transformed(row_map, export["features"], tf)
    idx = {name: i for i, name in enumerate(export["features"])}
    total = 0.0
    for tree in export["trees"]:
        node = tree
        while "leaf" not in node:
            x = v[idx[node["split"]]]
            if x is None or x != x:  # NaN -> missing branch (before f32)
                node = next(c for c in node["children"] if c["nodeid"] == node["missing"])
                continue
            nxt = node["yes"] if f32(x) < f32(node["split_condition"]) else node["no"]
            node = next(c for c in node["children"] if c["nodeid"] == nxt)
        total += float(node["leaf"])
    return 1.0 / (1.0 + math.exp(-total))


def iforest_predict_js(export, row_map):
    v = transformed(row_map, export["features"], export["transform"])
    depths = 0.0
    for t in export["trees"]:
        node, depth = 0, 0
        while t["f"][node] != -1:
            x = v[t["f"][node]]
            node = t["l"][node] if f32(x) <= f32(t["t"][node]) else t["r"][node]
            depth += 1
        depths += depth + average_path_length(t["n"][node])
    score = -(2.0 ** (-depths / export["n_trees"] / export["c_n"]))
    return score, score - export["offset"] < 0


def irr_predict_js(export, row_map):
    v = transformed(row_map, export["features"], export["transform"])
    vals = np.zeros(len(export["classes"]))
    for t in export["trees"]:
        node = 0
        while t["f"][node] != -1:
            node = t["l"][node] if f32(v[t["f"][node]]) <= f32(t["t"][node]) else t["r"][node]
        leaf = np.array(t["v"][node], dtype=float)
        vals += leaf / leaf.sum()
    return vals / len(export["trees"])


df = pd.read_csv(DATASET_CSV)
df = df.dropna(subset=[c for c in FEATURE_COLS if c != "ph"]).head(120)
rows = [{c: (None if pd.isna(df.iloc[i][c]) else float(df.iloc[i][c]))
         for c in FEATURE_COLS} for i in range(len(df))]

xgb_export = json.load(open(os.path.join(OUT, "xgb_potability.json"), encoding="utf-8"))
X_real = pd.DataFrame(rows, columns=FEATURE_COLS)
proba_real = xgb_pipe.predict_proba(X_real)[:, 1]
# Raw float rows as plain Python lists -> fast traversal, no pandas in the loop.
raw_rows = [tuple(None if r[c] is None else float(r[c]) for c in xgb_export["features"])
            for r in rows]
feat_idx = {name: i for i, name in enumerate(xgb_export["features"])}
diffs = [abs(xgb_predict_js(xgb_export, raw_rows[i]) - float(proba_real[i]))
         for i in range(len(raw_rows))]
print(f"[verify] xgboost max |proba diff| on {len(rows)} rows: {max(diffs):.2e}")
assert max(diffs) < 1e-5, "xgboost parity FAILED"

ife = json.load(open(os.path.join(OUT, "iforest.json"), encoding="utf-8"))
flags_real = if_pipe.predict(pd.DataFrame(rows, columns=FEATURE_COLS))
scores_real = -if_pipe.score_samples(pd.DataFrame(rows, columns=FEATURE_COLS))
max_sd, flag_bad = 0.0, 0
for i, row in enumerate(rows):
    s, flag = iforest_predict_js(ife, row)
    max_sd = max(max_sd, abs(s - float(scores_real[i])))
    if bool(flag) != bool(flags_real[i] == -1):
        flag_bad += 1
print(f"[verify] iforest max |score diff|: {max_sd:.2e}, flag mismatches: {flag_bad}/{len(rows)}")
assert max_sd < 1e-9 and flag_bad == 0, "iforest parity FAILED"

from backend.ml.load_irrigation_dataset import load_irrigation_dataframe  # noqa: E402
idf = load_irrigation_dataframe().head(80)
irows = [{c: (None if pd.isna(idf.iloc[i][c]) else float(idf.iloc[i][c]))
          for c in IRRIGATION_FEATURES} for i in range(len(idf))]
irre = json.load(open(os.path.join(OUT, "irr_rf.json"), encoding="utf-8"))
proba_real = irr_pipe.predict_proba(pd.DataFrame(irows, columns=IRRIGATION_FEATURES))
mism = sum(1 for i, row in enumerate(irows)
           if irre["classes"][int(np.argmax(irr_predict_js(irre, row)))]
           != str(irr_model.classes_[int(np.argmax(proba_real[i]))]))
print(f"[verify] irrigation RF class mismatches on {len(irows)} rows: {mism}")
assert mism == 0, "irrigation RF parity FAILED"

print("[verify] ALL PARITY CHECKS PASSED")
