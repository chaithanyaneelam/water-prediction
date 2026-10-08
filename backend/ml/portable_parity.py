"""Fast, standalone parity verification for site/public/static/models/*.json.

Reimplements the EXACT inference arithmetic the Worker JS uses (impute -> clip
-> scale, float32 comparisons like Math.fround) and compares against the real
sklearn/xgboost predictions on real dataset rows. Vectorized with numpy so it
runs in seconds instead of the pandas-loop exporter.

Also emits backend/ml/outputs/portable_parity.json recording the max diffs, so
the deployment can honestly claim verified parity.
"""
import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ml.config import DATASET_CSV, FEATURE_COLS, IRRIGATION_FEATURES  # noqa: E402

SAVED = os.path.join(os.path.dirname(__file__), "saved_models")
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "site", "public", "static", "models"))

EULER_GAMMA = 0.5772156649015329


def average_path_length(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + EULER_GAMMA) - 2.0 * (n - 1.0) / n


def f32(a):
    return np.asarray(a, dtype=np.float32)


def apply_transform(V, cols, tf):
    """V: (n, d) float array with NaN for missing; returns transformed array."""
    V = V.copy()
    if "impute" in tf:
        med = np.array(tf["impute"], float)
        nanmask = np.isnan(V)
        V[nanmask] = np.broadcast_to(med, V.shape)[nanmask]
    if "clip_lower" in tf:
        lo = np.array(tf["clip_lower"], float)
        hi = np.array(tf["clip_upper"], float)
        V = np.clip(V, lo, hi)
    if "mean" in tf:
        V = (V - np.array(tf["mean"], float)) / np.array(tf["scale"], float)
    return V


def xgb_proba_js(export, V):
    """V already transformed (n, d). Vectorized over rows."""
    idx = {name: i for i, name in enumerate(export["features"])}
    X32 = f32(V)
    probs = np.zeros(len(V), float)
    feats_orig = export["features"]
    for tree in export["trees"]:
        # Flatten the dumped tree into arrays for fast traversal.
        nodeid, children, leaf_val, split_col, split_cond, is_leaf = \
            _flatten_xgb(tree, idx)
        col = np.zeros(len(V), np.int64)
        cur = np.zeros(len(V), np.int64)
        active = np.ones(len(V), bool)
        d = X32[:, :]
        # Traverse per tree iteratively (max depth ~8).
        for _ in range(64):
            if not active.any():
                break
            leaf_mask = is_leaf[cur] & active
            if leaf_mask.any():
                probs[leaf_mask] += leaf_val[cur[leaf_mask]]
                active &= ~leaf_mask
            if not active.any():
                break
            c = split_col[cur[active]]
            x = X32[active, c]
            go_yes = x < f32(split_cond[cur[active]])
            nxt = np.where(go_yes, children[cur[active], 0], children[cur[active], 1])
            cur_arr = cur.copy()
            cur_arr[active] = nxt
            cur = cur_arr
        # Any rows still not landed at a leaf after 64 iters: malformed tree.
    return 1.0 / (1.0 + np.exp(-probs))


def _flatten_xgb(tree, idx):
    nodes = {}

    def walk(n):
        nodes[n["nodeid"]] = n
        for c in n.get("children", []):
            walk(c)
    walk(tree)
    max_id = max(nodes) + 1
    children = np.full((max_id, 2), -1, np.int64)
    leaf_val = np.zeros(max_id, float)
    split_col = np.zeros(max_id, np.int64)
    split_cond = np.zeros(max_id, float)
    is_leaf = np.zeros(max_id, bool)
    for nid, n in nodes.items():
        if "leaf" in n:
            is_leaf[nid] = True
            leaf_val[nid] = float(n["leaf"])
        else:
            split_col[nid] = idx[n["split"]]
            split_cond[nid] = float(n["split_condition"])
            for c in n["children"]:
                children[nid, 0 if c["nodeid"] == n["yes"] else 1] = c["nodeid"]
    return None, children, leaf_val, split_col, split_cond, is_leaf


def iforest_scores_js(export, V):
    """V already transformed. Returns (score_samples, is_anomaly)."""
    scores = np.zeros(len(V), float)
    for t in export["trees"]:
        f = np.array(t["f"], np.int64)
        thr = f32(np.array(t["t"], float))
        left = np.array(t["l"], np.int64)
        right = np.array(t["r"], np.int64)
        nsamp = np.array(t["n"], float)
        cur = np.zeros(len(V), np.int64)
        depth = np.zeros(len(V), float)
        alive = np.ones(len(V), bool)
        while True:
            move = alive & (f[cur] != -1)
            if not move.any():
                break
            x = f32(V[np.arange(len(V)), f[cur]])[move]
            go_left = x <= thr[cur[move]]
            nxt = np.where(go_left, left[cur[move]], right[cur[move]])
            # Leaves have children -1; clamp so cur never goes negative
            # (numpy would wrap -1 to the last tree node).
            nxt = np.maximum(nxt, 0)
            newcur = cur.copy()
            newcur[move] = nxt
            cur = newcur
            depth[move] += 1
            alive = alive & (f[cur] != -1)
        leaf_n = nsamp[cur]
        scores += depth + np.array([average_path_length(n) for n in leaf_n])
    pts = -(2.0 ** (-scores / export["n_trees"] / export["c_n"]))
    return pts, pts - export["offset"] < 0


def irr_classes_js(export, V):
    agg = np.zeros((len(V), len(export["classes"])), float)
    for t in export["trees"]:
        thr = f32(np.array(t["t"], float))
        left = np.array(t["l"], np.int64)
        right = np.array(t["r"], np.int64)
        val = np.array(t["v"], float)
        cur = np.zeros(len(V), np.int64)
        alive = np.ones(len(V), bool)
        while True:
            move = alive & (np.array(t["f"])[cur] != -1)
            if not move.any():
                break
            x = f32(V[np.arange(len(V)), np.array(t["f"])[cur]])[move]
            go_left = x <= thr[cur[move]]
            nxt = np.where(go_left, left[cur[move]], right[cur[move]])
            # Clamp: leaves carry children -1; never let cur go negative.
            nxt = np.maximum(nxt, 0)
            newcur = cur.copy()
            newcur[move] = nxt
            cur = newcur
            alive = alive & (np.array(t["f"])[cur] != -1)
        leaf = val[cur]
        agg += leaf / leaf.sum(axis=1, keepdims=True)
    return agg / len(export["trees"])


if __name__ == "__main__":
    xgb_pipe = joblib.load(os.path.join(SAVED, "xgboost_potability.joblib"))
    if_pipe = joblib.load(os.path.join(SAVED, "isolation_forest.joblib"))
    irr_pipe = joblib.load(os.path.join(SAVED, "rf_irrigation.joblib"))
    xgb_export = json.load(open(os.path.join(OUT, "xgb_potability.json"), encoding="utf-8"))
    ife = json.load(open(os.path.join(OUT, "iforest.json"), encoding="utf-8"))
    irre = json.load(open(os.path.join(OUT, "irr_rf.json"), encoding="utf-8"))

    # ---------------- potability + anomaly on real Kaggle rows
    df = pd.read_csv(DATASET_CSV)
    df = df.dropna(subset=[c for c in FEATURE_COLS if c != "ph"]).head(120)
    raw = df[list(FEATURE_COLS)].to_numpy(float)
    miss = np.isnan(raw[:, FEATURE_COLS.index("ph")])

    Xp = apply_transform(raw, FEATURE_COLS, xgb_export["transform"])
    p_js = xgb_proba_js(xgb_export, Xp)
    p_sk = xgb_pipe.predict_proba(pd.DataFrame(raw, columns=FEATURE_COLS))[:, 1]
    xgb_diff = float(np.max(np.abs(p_js - p_sk)))
    xgb_flag_diff = int(np.sum((p_js >= 0.5) != (p_sk >= 0.5)))

    Xi = apply_transform(raw, FEATURE_COLS, ife["transform"])
    js_pts, js_flag = iforest_scores_js(ife, Xi)
    sk_pts = if_pipe.score_samples(pd.DataFrame(raw, columns=FEATURE_COLS))
    """sklearn score_samples == -2^(-E/(c*n_trees)) exactly, which is what
    iforest_scores_js returns - no sign flip needed."""
    sk_flag = if_pipe.predict(pd.DataFrame(raw, columns=FEATURE_COLS))
    if_diff = float(np.max(np.abs(js_pts - sk_pts)))
    if_flag_diff = int(np.sum(js_flag != (sk_flag == -1)))

    print(f"[verify] xgboost: max |proba diff| = {xgb_diff:.2e}, "
          f"label diffs = {xgb_flag_diff}/{len(raw)}")
    print(f"[verify] iforest: max |score diff| = {if_diff:.2e}, "
          f"flag mismatches = {if_flag_diff}/{len(raw)}")

    # ---------------- irrigation RF on real Telangana rows
    from backend.ml.load_irrigation_dataset import load_raw
    idf = load_raw().head(80)
    iraw = idf[list(IRRIGATION_FEATURES)].to_numpy(float)
    iraw = np.where(np.isnan(iraw), np.nan, iraw)
    Xi2 = apply_transform(iraw, IRRIGATION_FEATURES, irre["transform"])
    js_p = irr_classes_js(irre, Xi2)
    sk_p = irr_pipe.predict_proba(pd.DataFrame(iraw, columns=IRRIGATION_FEATURES))
    js_lab = np.array(irre["classes"])[js_p.argmax(1)]
    sk_lab = np.array([str(c) for c in irr_pipe.named_steps["model"].classes_])[sk_p.argmax(1)]
    irr_mism = int(np.sum(js_lab != sk_lab))
    print(f"[verify] irrigation RF: class mismatches = {irr_mism}/{len(iraw)}")

    ok = xgb_diff < 1e-5 and xgb_flag_diff == 0 and if_diff < 1e-9 \
        and if_flag_diff == 0 and irr_mism == 0
    result = {
        "n_rows_potability": int(len(raw)),
        "n_rows_irrigation": int(len(iraw)),
        "xgboost_max_proba_diff": xgb_diff,
        "xgboost_label_diffs": xgb_flag_diff,
        "iforest_max_score_diff": if_diff,
        "iforest_flag_mismatches": if_flag_diff,
        "irrigation_class_mismatches": irr_mism,
        "passed": bool(ok),
    }
    out_path = os.path.join(os.path.dirname(__file__), "outputs", "portable_parity.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("[verify] wrote", out_path)
    if not ok:
        raise SystemExit("PARITY FAILED")
    print("[verify] ALL PARITY CHECKS PASSED")
