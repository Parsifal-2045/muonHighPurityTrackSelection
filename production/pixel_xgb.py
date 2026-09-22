"""
pixel_xgb.py - XGBoost model for high-purity muon pixel-track selection.

Reuses pixel_features.py for feature extraction and evaluation, so the
XGBoost model sees byte-identical features to the DNN (pixel_model.py) and
uses the same train/val/test split, sample weights, and evaluation suite.

Two export formats are produced, both verified against XGBoost predict():
  - model_xgb.onnx       : ONNX TreeEnsembleClassifier (via onnxmltools),
                          for ONNX Runtime inference.
  - model_compact.bin    : Compact gradient-boosted-tree binary, read by the
                          CMSSW PixelTrackForestHighPuritySelector module
                          (int8 feat, fp32 val, int32 left/right/roots, fp32 base_logit).

Configuration: the default CFG is the validated 33-feature production model
(11 features pruned, 2000 trees, early stopping 100).

Run:
    python pixel_xgb.py
"""

import gc
import os
import time

import numpy as np
import uproot
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import pixel_features as pf


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CFG = dict(
    data_dir=pf.DATA_DIR,
    output_dir="pixel_xgb_output_33f/",
    # Feature flags (match DNN)
    useL1TkMuFeatures=True,
    useL1TkMuStubFeatures=True,
    # Feature pruning: drop these 11 features (by name).
    # Selected via permutation importance on the 44-feature model:
    #   - 2 zero-gain features (nLostHits, hitEfficiency)
    #   - 6 near-zero gain chi2/err features (normalizedChi2, chi2PerHit, chi2,
    #     impactSignificance, dxyErr, dszErr)
    #   - 3 redundant features (eta is absEta's mirror, ptErr/qoverpErr/
    #     relUncertaintyProduct duplicate qoverpErr's info)
    # Set to None to use all 44 features.
    drop_features=[
        "muon_pixel_tracks_nLostHits",          # zero gain
        "muon_pixel_tracks_hitEfficiency",      # zero gain
        "muon_pixel_tracks_normalizedChi2",      # gain 64.6
        "muon_pixel_tracks_chi2PerHit",         # gain 90.3
        "muon_pixel_tracks_chi2",               # gain 105.8
        "muon_pixel_tracks_impactSignificance",  # gain 113.7
        "muon_pixel_tracks_dxyErr",               # gain 115.5
        "muon_pixel_tracks_dszErr",              # gain 128.8
        "muon_pixel_tracks_eta",                 # redundant with absEta
        "muon_pixel_tracks_ptErr",               # redundant with qoverpErr
        "muon_pixel_tracks_relUncertaintyProduct",  # redundant with qoverpErr
    ],
    # Sample weights (match DNN exactly)
    signal_boost=pf.SIGNAL_BOOST,
    kin_weight_max=pf.KIN_WEIGHT_MAX,
    # XGBoost hyperparameters (validated via pixel_xgb_optimize.py)
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    # Early stopping
    early_stopping_rounds=100,
    # ONNX
    onnx_opset=pf.ONNX_OPSET,
    # Compact binary export (forest module format).
    # fp16 quantizes threshold/leaf values to float16, halving the .bin size.
    # The accumulated rounding across 2000 trees produces max score error ~0.18,
    # which is too lossy for a selector operating at fixed threshold. The
    # deployed displaced model also uses fp32. Keep fp32.
    compact_bin_fp16=False,
)


def load_data(cfg):
    """Load all input files and build the feature matrix.

    Caches to /tmp to avoid re-reading from NFS on restart.
    """
    import json
    cache_path = "/tmp/pixel_data_cache.npz"
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path} ...")
        cached = np.load(cache_path, allow_pickle=True)
        X = cached["X"]
        y = cached["y"]
        fl = cached["fl"]
        with open("/tmp/pixel_feature_names.json") as f:
            feature_names = json.load(f)
        files = pf.get_files(cfg["data_dir"])
        print(f"  Cached: {X.shape}, {len(feature_names)} features")
        return X, y, fl, feature_names, files

    files = pf.get_files(cfg["data_dir"])
    print(f"Selected {len(files)} input files:")
    print(files)

    X_list, y_list, fl_list = [], [], []
    feature_names = []
    total_ev = 0
    print(f"Processing {len(files)} files ...")
    for i, f in enumerate(files):
        print(f"  [{i + 1}/{len(files)}] {f}")
        with uproot.open(f) as rf:
            a = rf[pf.MAIN_BRANCH].arrays(
                pf.TK_BRANCHES + pf.L1TKMUON_BRANCHES + pf.STUB_BRANCHES
            )
            ne = len(a)
            total_ev += ne
            Xc, yc, lc, fn = pf.build_dataset(
                a,
                np.full(ne, i),
                useL1TkMuFeatures=cfg["useL1TkMuFeatures"],
                useL1TkMuStubFeatures=cfg["useL1TkMuStubFeatures"],
                verbose=True,
            )
            X_list.append(Xc)
            y_list.append(yc)
            fl_list.append(lc)
            if i == 0:
                feature_names = fn
            del a, Xc, yc, lc
            gc.collect()

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    fl = np.concatenate(fl_list)
    print(
        f"\nTotal events: {total_ev}  |  Features: {X.shape}  |  "
        f"{len(feature_names)} features"
    )
    print(f"Features: {feature_names}")

    # Cache to local disk for fast restart
    print(f"Caching data to {cache_path} ...")
    np.savez(cache_path, X=X, y=y, fl=fl)
    with open("/tmp/pixel_feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print(f"  Cached: {os.path.getsize(cache_path) / 1e9:.1f} GB")

    return X, y, fl, feature_names, files


def main():
    cfg = CFG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- Load data ------------------------------------------------------- #
    X, y, fl, feature_names, files = load_data(cfg)

    # ---- Feature pruning ------------------------------------------------ #
    drop = cfg.get("drop_features")
    if drop:
        keep_idx = [i for i, fn in enumerate(feature_names) if fn not in drop]
        dropped = [fn for fn in feature_names if fn not in
                   [feature_names[i] for i in keep_idx]]
        X = X[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]
        print(f"\nFeature pruning: dropped {len(drop)} features, kept {len(feature_names)}")
        print(f"  Dropped: {drop}")
        print(f"  Kept: {feature_names}")
    else:
        print(f"\nUsing all {len(feature_names)} features (no pruning)")

    pt_feat_idx = feature_names.index("muon_pixel_tracks_pt")

    # ---- Sample weights (identical to DNN) ------------------------------ #
    print("\nComputing sample weights ...")
    pt_vals = 10 ** X[:, pt_feat_idx]
    weights = pf.compute_sample_weights(
        y, pt_vals, signal_boost=cfg["signal_boost"], kin_weight_max=cfg["kin_weight_max"]
    )
    sig = y == 1
    bg = y == 0
    print(
        f"  Signal weight mean: {weights[sig].mean():.3f}  |  "
        f"Background: {weights[bg].mean():.3f}"
    )

    # Low-pT mask (before scaling; trees don't need scaling but we track it for eval)
    low_pt_mask = (pt_vals < pf.LOW_PT_CUT).astype(np.float32)
    print(f"  Low-pT tracks: {low_pt_mask.sum():.0f} ({100 * low_pt_mask.mean():.1f}%)")

    # ---- Split (identical stratification to DNN) ------------------------ #
    print("\nStratified split ...")
    strat = y * len(files) + fl
    X_tv, X_test, y_tv, y_test, w_tv, w_test, l_tv, l_test, lp_tv, lp_test = (
        train_test_split(
            X,
            y,
            weights,
            fl,
            low_pt_mask,
            test_size=0.2,
            stratify=strat,
            random_state=42,
        )
    )
    X_train, X_val, y_train, y_val, w_train, w_val, lp_train, lp_val = train_test_split(
        X_tv, y_tv, w_tv, lp_tv, test_size=0.2, stratify=y_tv, random_state=42
    )
    del X, y, weights, low_pt_mask, X_tv, y_tv, w_tv, lp_tv
    gc.collect()
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ---- Train XGBoost --------------------------------------------------- #
    print("\n" + "=" * 70 + "\nXGBOOST TRAINING\n" + "=" * 70)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    params = dict(
        objective="binary:logistic",
        eval_metric=["aucpr", "auc", "logloss"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        min_child_weight=cfg["min_child_weight"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print(f"  Params: {params}")
    print(f"  n_estimators={cfg['n_estimators']}  "
          f"early_stopping={cfg['early_stopping_rounds']}")

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=cfg["early_stopping_rounds"],
        verbose_eval=50,
    )

    best_iter = model.best_iteration
    best_score = model.best_score
    print(f"\n  Best iteration: {best_iter}")
    print(f"  Best val score: {best_score:.6f}")

    # ---- Evaluation ------------------------------------------------------ #
    print("\n" + "=" * 70 + "\nEVALUATION\n" + "=" * 70)

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    y_true = y_test.astype(np.int32)

    prauc = average_precision_score(y_true, y_pred)
    ra = roc_auc_score(y_true, y_pred)
    print(f"\n  Test ROC AUC: {ra:.4f}  |  Test PR AUC: {prauc:.4f}")

    # ROC + PR curves
    roc_auc_val, pr_auc_val, prec_arr, rec_arr, thresholds = pf.plot_roc_pr(
        y_true, y_pred, cfg["output_dir"]
    )

    # F2-optimal threshold
    th1, f1_best, th2, f2_best = pf.find_f2_threshold(y_true, y_pred)
    final_th = th2
    print(f"\n  Optimal threshold (F1): {th1:.4f}  F1={f1_best:.4f}")
    print(f"  Optimal threshold (F2): {th2:.4f}  F2={f2_best:.4f}")
    print(f"  Using F2 threshold: {final_th:.4f}")

    # Classification report + confusion matrix
    yb = (y_pred >= final_th).astype(int)
    print(f"\n{classification_report(y_true, yb, digits=4)}")
    cm = confusion_matrix(y_true, yb)
    print(cm)
    pf.plot_confusion_matrix(y_true, y_pred, final_th, cfg["output_dir"])

    # Per-pT-bin evaluation
    print("\n" + "=" * 70 + "\nPER-pT-BIN PERFORMANCE\n" + "=" * 70)
    pt_test_raw = 10 ** X_test[:, pt_feat_idx]
    pf.evaluate_pt_bins(y_true, y_pred, pt_test_raw, final_th, cfg["output_dir"])

    # Per-file evaluation
    print("\n" + "=" * 70 + "\nPER-FILE PERFORMANCE\n" + "=" * 70)
    for fi, fname in enumerate(files):
        fm = l_test == fi
        if fm.sum() == 0:
            continue
        yf, pf_ = y_true[fm], y_pred[fm]
        bf = (pf_ >= final_th).astype(int)
        a = roc_auc_score(yf, pf_) if len(np.unique(yf)) > 1 else float("nan")
        c = confusion_matrix(yf, bf, labels=[0, 1]).ravel()
        tn, fp, fn, tp = c
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f2 = 5 * p * r / (4 * p + r + 1e-6)
        print(
            f"\n  {fname.split('/')[-1]:<30s}  AUC={a:.4f}  Prec={p:.4f}  "
            f"Rec={r:.4f}  F2={f2:.4f}"
            f"\n    {'':30s}  [TN={tn} FP={fp} FN={fn} TP={tp}]"
        )

    # ---- Feature importance (XGBoost gain + SHAP) ----------------------- #
    print("\n" + "=" * 70 + "\nFEATURE IMPORTANCE\n" + "=" * 70)

    # Gain importance (built-in)
    print("\n[A] Full test set (XGBoost gain):")
    gain = model.get_score(importance_type="gain")
    cover = model.get_score(importance_type="cover")
    # XGBoost names features f0..fN; map back to real names
    name_map = {f"f{i}": fn for i, fn in enumerate(feature_names)}
    gain_named = {name_map.get(k, k): v for k, v in gain.items()}
    cover_named = {name_map.get(k, k): v for k, v in cover.items()}

    # Sort by gain
    sorted_gain = sorted(gain_named.items(), key=lambda x: -x[1])
    print(f"  {'Rank':<6}{'Feature':<55}{'Gain':>12}")
    print("  " + "-" * 73)
    for rank, (fn, gv) in enumerate(sorted_gain):
        cv = cover_named.get(fn, 0.0)
        print(f"  {rank + 1:<6}{fn:<55}{gv:>12.2f}  cover={cv:.1f}")

    # Save gain importance to file
    with open(cfg["output_dir"] + "feature_importance.txt", "w") as fo:
        fo.write(f"Baseline PR-AUC: {prauc:.6f}\n")
        fo.write(f"Test ROC-AUC: {ra:.6f}\n\n")
        fo.write(f"{'Rank':<6}{'Feature':<55}{'Gain':>12}\n")
        fo.write("-" * 73 + "\n")
        for rank, (fn, gv) in enumerate(sorted_gain):
            cv = cover_named.get(fn, 0.0)
            fo.write(f"{rank + 1:<6}{fn:<55}{gv:>12.2f}  cover={cv:.1f}\n")

    # Plot gain importance
    names_arr = [fn for fn, _ in sorted_gain]
    gain_arr = np.array([gv for _, gv in sorted_gain])
    std_arr = np.zeros_like(gain_arr)
    pf.plot_importance(
        gain_arr,
        std_arr,
        names_arr,
        "Feature importance (XGBoost gain) - full test",
        cfg["output_dir"] + "feat_imp_all.png",
    )

    # Low-pT subset importance
    lpm = pt_test_raw < pf.LOW_PT_CUT
    print(
        f"\n[B] Low-pT subset: {lpm.sum()} tracks ({y_true[lpm].sum():.0f} signal)"
    )
    if lpm.sum() > 1000 and len(np.unique(y_true[lpm])) == 2:
        # Permutation importance on the low-pT subset (model-agnostic)
        print("  Computing permutation importance on low-pT subset ...")
        il, sl, bl = compute_permutation_importance_xgb(
            model, X_test[lpm], y_true[lpm], feature_names
        )
        pf.plot_importance(
            il,
            sl,
            feature_names,
            "Feature importance (permutation) - low-pT",
            cfg["output_dir"] + "feat_imp_lowpt.png",
        )
        print("\n[C] Shift (low-pT vs global gain):")
        # Compare low-pT permutation importance to global gain
        global_gain_map = {fn: gv for fn, gv in sorted_gain}
        d = il - np.array([global_gain_map.get(fn, 0.0) for fn in feature_names])
        od = np.argsort(d)[::-1]
        for i in od:
            print(
            )

    # ---- ONNX export ---------------------------------------------------- #
    print("\n" + "=" * 70 + "\nONNX EXPORT\n" + "=" * 70)
    onnx_path = export_onnx(model, feature_names, cfg)
    if onnx_path:
        verify_onnx(onnx_path, X_test, y_pred, cfg)

    # ---- Compact binary export (forest) --------------------------------- #
    print("\n" + "=" * 70 + "\nCOMPACT BINARY EXPORT (forest)\n" + "=" * 70)
    bin_path = export_compact_bin(model, feature_names, cfg)
    if bin_path:
        verify_compact_bin(bin_path, X_test, y_pred, cfg)

    # ---- Save thresholds ------------------------------------------------ #
    with open(cfg["output_dir"] + "thresholds.txt", "w") as fo:
        fo.write(f"F1_Threshold: {th1}\nF2_Threshold: {th2}\n")
        fo.write(f"Test_ROC_AUC: {ra}\nTest_PR_AUC: {prauc}\n")
        fo.write(f"Best_iteration: {best_iter}\n")
        fo.write(f"Signal_boost: {cfg['signal_boost']}\n")
        fo.write(f"Model: XGBoost\n")

    # ---- Save model ----------------------------------------------------- #
    model_path = cfg["output_dir"] + "model.json"
    model.save_model(model_path)
    print(f"\n  Saved model: {model_path}")

    # ---- Re-score DNN ONNX on exact same test set --------------------- #
    dnn_onnx = cfg.get("dnn_onnx_path", "pixel_output/model_standard.onnx")
    if os.path.isfile(dnn_onnx):
        print("\n" + "=" * 70 + "\nDNN RE-SCORE (exact same test set)\n" + "=" * 70)
        rescore_dnn(dnn_onnx, X_test, y_test, pt_test_raw, l_test, files, cfg)
    else:
        print(f"\n  DNN ONNX not found at {dnn_onnx}, skipping re-score")

# ---- Compact binary export (forest) ---------------------------------- #
# Writes the compact gradient-boosted-tree binary format consumed by the
# CMSSW PixelTrackForestHighPuritySelector module:
#   int32 nNodes, int32 nTrees, float baseLogit,
#   int8  feat[nNodes]   (-1 = leaf),
#   float val[nNodes]    (threshold / leaf value),
#   int32 left[nNodes], right[nNodes],
#   int32 roots[nTrees]
# The kernel traverses: while feat[node] >= 0:
#   node = (x[feat[node]] < val[node]) ? left[node] : right[node]
# then margin += val[node];  score = sigmoid(margin).
#
# fp16 quantization of leaf/threshold values (upcast back to fp32 on load)
# halves the binary size for negligible accuracy loss. The default matches
# the deployed displaced model.


def _base_logit_from_booster(bst):
    """logit(base_score). XGBoost stores base_score as a probability for
    binary:logistic; the kernel adds baseLogit in margin space."""
    import json

    cfg = json.loads(bst.save_config())
    bs = float(cfg["learner"]["learner_model_param"]["base_score"])
    bs = min(max(bs, 1e-7), 1.0 - 1e-7)
    return float(np.log(bs / (1.0 - bs)))


def _flatten_tree(node_json, feat, val, left, right):
    """Append one tree's nodes (BFS, sequential child allocation) to the
    running lists. Returns the global index of this tree's root."""
    import json

    root_global = len(feat)
    feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
    queue = [(node_json, root_global)]
    while queue:
        n, slot = queue.pop(0)
        if "leaf" in n:
            feat[slot] = -1
            val[slot] = float(n["leaf"])
            left[slot] = -1
            right[slot] = -1
            continue
        feat[slot] = int(n["split"][1:])  # "f12" -> 12
        val[slot] = float(n["split_condition"])
        children = {c["nodeid"]: c for c in n["children"]}
        yes_c = children[n["yes"]]
        no_c = children[n["no"]]
        l_idx = len(feat)
        feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
        r_idx = len(feat)
        feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
        left[slot] = l_idx
        right[slot] = r_idx
        queue.append((yes_c, l_idx))
        queue.append((no_c, r_idx))
    return root_global


def export_compact_bin(model, feature_names, cfg):
    """Export the XGBoost booster to the compact .bin format read by the
    CMSSW forest selector module. Returns the output path."""
    import json
    import struct

    dump = model.get_dump(dump_format="json")
    nt = len(dump)
    feat, val, left, right, roots = [], [], [], [], []
    for t in range(nt):
        r = _flatten_tree(json.loads(dump[t]), feat, val, left, right)
        roots.append(r)

    feat = np.asarray(feat, dtype=np.int8)
    val = np.asarray(val, dtype=np.float32)
    left = np.asarray(left, dtype=np.int32)
    right = np.asarray(right, dtype=np.int32)
    roots = np.asarray(roots, dtype=np.int32)
    base_logit = np.float32(_base_logit_from_booster(model))

    fp16 = cfg.get("compact_bin_fp16", True)
    val_out = val.astype(np.float16).astype(np.float32) if fp16 else val

    path = cfg["output_dir"] + "model_compact.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<iif", feat.shape[0], nt, float(base_logit)))
        f.write(feat.tobytes())
        f.write(val_out.tobytes())
        f.write(left.tobytes())
        f.write(right.tobytes())
        f.write(roots.tobytes())

    size = os.path.getsize(path)
    print(f"  Saved: {path}  ({size / 1024:.0f} KB)")
    print(f"    nNodes={feat.shape[0]}  nTrees={nt}  "
          f"baseLogit={float(base_logit):.8f}  "
          f"val precision={'fp16->fp32' if fp16 else 'fp32'}")
    return path


def verify_compact_bin(bin_path, X_test, y_pred_xgb, cfg):
    """Verify the compact .bin produces the same predictions as XGBoost by
    reimplementing the kernel traversal in numpy."""
    import struct

    with open(bin_path, "rb") as f:
        n_nodes, n_trees, base_logit = struct.unpack("<iif", f.read(12))
        feat = np.frombuffer(f.read(n_nodes), dtype=np.int8)
        val = np.frombuffer(f.read(n_nodes * 4), dtype=np.float32)
        left = np.frombuffer(f.read(n_nodes * 4), dtype=np.int32)
        right = np.frombuffer(f.read(n_nodes * 4), dtype=np.int32)
        roots = np.frombuffer(f.read(n_trees * 4), dtype=np.int32)

    n_check = min(50000, X_test.shape[0])
    X = X_test[:n_check].astype(np.float32)
    scores = np.empty(n_check, dtype=np.float32)
    for i in range(n_check):
        margin = base_logit
        for t in range(n_trees):
            node = roots[t]
            while feat[node] >= 0:
                node = left[node] if X[i, feat[node]] < val[node] else right[node]
            margin += val[node]
        scores[i] = 1.0 / (1.0 + np.exp(-margin))

    max_diff = np.max(np.abs(scores - y_pred_xgb[:n_check]))
    print(f"  Compact .bin vs XGBoost max |diff|: {max_diff:.2e}")
    assert max_diff < 1e-5, (
        f"Compact .bin / XGBoost mismatch: max |diff|={max_diff:.2e} >= 1e-5"
    )
    print(f"  Compact .bin verification PASSED (max |diff| < 1e-5)")

# --------------------------------------------------------------------------- #
# Permutation importance for XGBoost
# --------------------------------------------------------------------------- #
def compute_permutation_importance_xgb(model, X, y, names, n=5):
    """Compute permutation importance by shuffling each feature."""
    rng = np.random.default_rng(0)
    base = _prauc_xgb(model, X, y)
    print(f"  Baseline PR-AUC: {base:.6f}")
    imp = np.zeros((len(names), n), dtype=np.float32)
    for fi, fn in enumerate(names):
        for r in range(n):
            Xp = X.copy()
            rng.shuffle(Xp[:, fi])
            imp[fi, r] = base - _prauc_xgb(model, Xp, y)
        print(
            f"    [{fi:02d}] {fn:<50s}  D={imp[fi].mean():+.5f} +/- {imp[fi].std():.5f}"
        )
    return imp.mean(1), imp.std(1), base


def _prauc_xgb(model, X, y):
    d = xgb.DMatrix(X)
    p = model.predict(d)
    return average_precision_score(y, p)


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #
def export_onnx(model, feature_names, cfg):
    """Export XGBoost model to ONNX via onnxmltools.

    The ONNX graph has two outputs:
      - 'label' (int64): predicted class
      - 'probabilities' (float32, [N, 2]): class probabilities
    The positive-class score is probabilities[:, 1], matching XGBoost's
    predict() output for binary:logistic.
    """
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        import onnx

        n_features = len(feature_names)
        initial_type = [("input", FloatTensorType([None, n_features]))]

        # onnxmltools 1.16 does not support a zipmap option; the graph
        # will have two outputs: 'label' (int64) and 'probabilities'
        # (float32 [N,2]). verify_onnx extracts probabilities[:,1].
        onnx_model = convert_xgboost(
            model,
            initial_types=initial_type,
            target_opset=cfg["onnx_opset"],
        )

        # Store feature names as model metadata
        meta = {f"feature_{i}": fn for i, fn in enumerate(feature_names)}
        meta["feature_names"] = ",".join(feature_names)
        for k, v in meta.items():
            entry = onnx.StringStringEntryProto()
            entry.key = k
            entry.value = v
            onnx_model.metadata_props.append(entry)

        path = cfg["output_dir"] + "model_xgb.onnx"
        onnx.save_model(onnx_model, path)
        print(f"  Saved: {path}  ({os.path.getsize(path) / 1024:.0f} KB)")
        return path
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def verify_onnx(onnx_path, X_test, y_pred_xgb, cfg):
    """Verify ONNX model produces the same predictions as XGBoost.

    The ONNX graph outputs (label, probabilities). We extract
    probabilities[:, 1] (positive-class score) and compare to XGBoost's
    predict() output. A hard assert ensures the check can actually fail.
    """
    try:
        import onnxruntime as rt

        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        outputs = sess.get_outputs()
        print(f"  ONNX outputs: {[(o.name, o.shape, o.type) for o in outputs]}")

        # Run on a subset for speed
        n_check = min(10000, X_test.shape[0])
        onnx_result = sess.run(
            None, {input_name: X_test[:n_check].astype(np.float32)}
        )
        # Find the probabilities output (float32, 2D [N, 2])
        probs = None
        for r in onnx_result:
            if r.dtype == np.float32 and r.ndim == 2 and r.shape[1] == 2:
                probs = r
                break
        if probs is None:
            raise RuntimeError(
                "Could not find probabilities output in ONNX graph. "
                f"Outputs: {[(r.shape, r.dtype) for r in onnx_result]}"
            )
        onnx_pred = probs[:, 1]  # positive-class score
        xgb_pred = y_pred_xgb[:n_check]
        max_diff = np.abs(onnx_pred.ravel() - xgb_pred.ravel()).max()
        print(f"  ONNX vs XGBoost max |diff|: {max_diff:.2e}")
        assert max_diff < 1e-5, (
            f"ONNX/XGBoost mismatch: max |diff|={max_diff:.2e} >= 1e-5"
        )
        print(f"  ONNX verification PASSED (max |diff| < 1e-5)")

        # Inference benchmark (1000 x single-sample CPU)
        single = X_test[:1].astype(np.float32)
        for _ in range(100):
            sess.run(None, {input_name: single})
        t0 = time.perf_counter()
        for _ in range(1000):
            sess.run(None, {input_name: single})
        dt = (time.perf_counter() - t0) / 1000
        print(f"  Inference benchmark (1000 x single-sample CPU): {dt * 1e6:.1f} us/sample")
    except Exception as e:
        print(f"  ONNX verification failed: {e}")
        raise


def rescore_dnn(dnn_onnx_path, X_test, y_test, pt_test_raw, l_test, files, cfg):
    """Re-score the DNN ONNX model on the exact same test set as XGBoost.

    Produces a side-by-side comparison table so the architecture comparison
    is exact (same data, same split, same test rows).
    """
    import onnxruntime as rt
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )

    print(f"  Loading DNN ONNX: {dnn_onnx_path}")
    sess = rt.InferenceSession(dnn_onnx_path)
    input_name = sess.get_inputs()[0].name

    # Run in batches
    bs = 8192
    y_pred_dnn = []
    t0 = time.perf_counter()
    for i in range(0, len(X_test), bs):
        chunk = X_test[i : i + bs].astype(np.float32)
        out = sess.run(None, {input_name: chunk})[0]
        y_pred_dnn.append(out)
    dt = time.perf_counter() - t0
    y_pred_dnn = np.concatenate(y_pred_dnn).ravel()
    print(f"  DNN inference: {dt:.1f}s")

    y_true = y_test.astype(np.int32)
    prauc = average_precision_score(y_true, y_pred_dnn)
    ra = roc_auc_score(y_true, y_pred_dnn)
    print(f"  DNN ROC-AUC={ra:.4f}  PR-AUC={prauc:.4f}")

    # F2 threshold
    th1, f1_best, th2, f2_best = pf.find_f2_threshold(y_true, y_pred_dnn)
    print(f"  DNN F2 threshold: {th2:.4f}  F2={f2_best:.4f}")

    # Confusion matrix
    yb = (y_pred_dnn >= th2).astype(int)
    cm = confusion_matrix(y_true, yb)
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    f2 = 5 * p * r / (4 * p + r + 1e-6)
    print(f"  DNN confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  DNN Prec={p:.4f} Rec={r:.4f} F2={f2:.4f}")

    # Per-pT-bin
    print("\n  DNN per-pT-bin (exact XGB test set):")
    pf.evaluate_pt_bins(y_true, y_pred_dnn, pt_test_raw, th2, cfg["output_dir"] + "dnn_")

    # Save DNN re-score results
    with open(cfg["output_dir"] + "dnn_rescore.txt", "w") as fo:
        fo.write(f"DNN_ONNX: {dnn_onnx_path}\n")
        fo.write(f"Test_ROC_AUC: {ra}\nTest_PR_AUC: {prauc}\n")
        fo.write(f"F2_Threshold: {th2}\nF2: {f2_best}\n")
        fo.write(f"TN={tn} FP={fp} FN={fn} TP={tp}\n")
        fo.write(f"Precision={p:.6f} Recall={r:.6f} F2={f2:.6f}\n")


if __name__ == "__main__":
    main()
