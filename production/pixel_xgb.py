"""
pixel_xgb.py - XGBoost model for high-purity muon pixel-track selection.

Reuses pixel_features.py for feature extraction and evaluation, so the
XGBoost model sees byte-identical features to the DNN (pixel_model.py) and
uses the same sample weights and evaluation suite.

v2 pipeline (HP-forest scheme):
  - event-level "evt10" split: ev % 10 -> train 60 / val 10 / test 30, so
    correlated tracks can never leak between splits;
  - the forest is fitted to a large budget with early stopping disabled and
    then pruned to the smallest tree prefix whose validation F2 (at the
    validation-chosen per-pT-bin working point) is within prune_tol of the
    best prefix;
  - working points are F2 set points per pT bin (edges
    CFG["pt_threshold_edges"], first bin pT < 2 GeV), derived on validation
    and frozen before the single test-split evaluation.

Two export formats are produced, both verified against XGBoost predict():
  - model_xgb.onnx       : ONNX TreeEnsembleClassifier (via onnxmltools),
                          for ONNX Runtime inference.
  - model_compact.bin    : Compact gradient-boosted-tree binary, read by the
                          CMSSW MuonIOTracksForestSelector module
                          (int8 feat, fp32 val, int32 left/right/roots, fp32 base_logit).

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

import pixel_features as pf


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CFG = dict(
    data_dir=pf.DATA_DIR,
    output_dir="io/pixel/",
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
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    # Tree budget + data-driven pruning (HP-forest scheme): fit the full
    # budget with early stopping disabled, then keep the smallest tree prefix
    # whose validation metric at the validation-chosen working point is within
    # prune_tol of the best prefix (the 2000-tree production model saturated
    # its budget: best_iteration=1999).
    n_estimators=5000,
    early_stopping_rounds=0,  # 0 = disabled; size is set by pruning
    prune_tol=0.001,          # val-F2 tolerance for the size selection
    prune_grid_step=50,       # prefix grid spacing [trees]
    prune_min_trees=100,      # smallest prefix considered
    # Device: "cuda" uses the GPU hist implementation (falls back to CPU).
    device="cuda",
    # Event-level split ("evt10"): ev % 10 -> train 60% / val 10% / test 30%;
    # every track of an event stays in the same split. Thresholds, pruning and
    # any other choice are derived on the validation split only; the test
    # split is reported once with frozen working points.
    split_mode="evt10",
    split_seed_offset=0,
    # pT-binned working points [GeV]: one F2 set point per bin (first set
    # point separates the pT < 2 GeV regime).
    pt_threshold_edges=[0.0, 2.0, 5.0, 10.0, 50.0, 200.0],
    min_bin_signal=100,  # sparser bins fall back to the global F2 threshold
    # Previous production forest, re-scored on the same evt10 split with
    # validation-derived working points for an honest comparison.
    ref_model_path="io/archive/pixel_xgb_output_33f/model.json",
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

    Returns (X, y, fl, ev, feature_names, files) where ev is a per-track event
    id (local to its input file) used for the event-level split.

    Caches to /tmp to avoid re-reading from NFS on restart. The cache carries
    an input-file manifest and is rebuilt when the manifest or the cache
    format (ev ids) does not match.
    """
    import json
    cache_path = "/tmp/pixel_data_cache.npz"
    names_path = "/tmp/pixel_feature_names.json"
    files = pf.get_files(cfg["data_dir"])
    if os.path.exists(cache_path):
        with open(names_path) as f:
            feature_names = json.load(f)
        cached = np.load(cache_path, allow_pickle=True)
        manifest_ok = (
            "ev" in cached and "files" in cached
            and [str(x) for x in cached["files"]] == files
        )
        if manifest_ok:
            print(f"Loading cached data from {cache_path} ...")
            X, y, fl, ev = cached["X"], cached["y"], cached["fl"], cached["ev"]
            print(f"  Cached: {X.shape}, {len(feature_names)} features")
            return X, y, fl, ev, feature_names, files
        print(f"Cache at {cache_path} is stale (format or inputs); rebuilding.")

    print(f"Selected {len(files)} input files:")
    print(files)

    X_list, y_list, fl_list, ev_list = [], [], [], []
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
            Xc, yc, lc, ec, fn = pf.build_dataset(
                a,
                np.full(ne, i),
                useL1TkMuFeatures=cfg["useL1TkMuFeatures"],
                useL1TkMuStubFeatures=cfg["useL1TkMuStubFeatures"],
                verbose=True,
                event_ids=np.arange(ne, dtype=np.int64),
            )
            X_list.append(Xc)
            y_list.append(yc)
            fl_list.append(lc)
            ev_list.append(ec)
            if i == 0:
                feature_names = fn
            del a, Xc, yc, lc, ec
            gc.collect()

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    fl = np.concatenate(fl_list)
    ev = np.concatenate(ev_list)
    print(
        f"\nTotal events: {total_ev}  |  Features: {X.shape}  |  "
        f"{len(feature_names)} features"
    )
    print(f"Features: {feature_names}")

    # Cache to local disk for fast restart
    print(f"Caching data to {cache_path} ...")
    np.savez(cache_path, X=X, y=y, fl=fl, ev=ev, files=np.array(files))
    with open(names_path, "w") as f:
        json.dump(feature_names, f)
    print(f"  Cached: {os.path.getsize(cache_path) / 1e9:.1f} GB")

    return X, y, fl, ev, feature_names, files


def main():
    cfg = CFG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    _log_fh = pf.tee_log(cfg["output_dir"])  # full log stored in the model dir

    # ---- Load data ------------------------------------------------------- #
    X, y, fl, ev, feature_names, files = load_data(cfg)

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

    # ---- Split (event-level "evt10": train 60 / val 10 / test 30) -------- #
    # Every track of an event lands in the same split, so correlated tracks
    # cannot leak between train and test. Working points and the forest size
    # are derived on the validation split only; the test split is evaluated
    # once, with frozen choices.
    print(f"\nEvent-level split ({cfg['split_mode']}) ...")
    assert cfg["split_mode"] == "evt10", f"unknown split_mode {cfg['split_mode']}"
    tr_m, va_m, te_m = pf.evt10_split(ev, seed_offset=cfg["split_seed_offset"])
    X_train, y_train, w_train = X[tr_m], y[tr_m], weights[tr_m]
    X_val, y_val, w_val, lp_val = X[va_m], y[va_m], weights[va_m], low_pt_mask[va_m]
    X_test, y_test, w_test, l_test, lp_test = (
        X[te_m], y[te_m], weights[te_m], fl[te_m], low_pt_mask[te_m],
    )
    del X, y, weights, low_pt_mask
    gc.collect()
    for name, (m_, yy) in {
        "train": (tr_m, y_train), "val": (va_m, y_val), "test": (te_m, y_test),
    }.items():
        print(
            f"  {name:5s}: {len(yy):>9d} tracks ({yy.sum():>9d} signal, "
            f"{100 * yy.mean():.1f}%)  in {len(np.unique(ev[m_]))} events"
        )
    del tr_m, va_m, te_m
    gc.collect()

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
        random_state=42,
    )
    esr = cfg["early_stopping_rounds"] or None
    try:
        device = cfg.get("device", "cpu")
        p = dict(params, device=device, n_jobs=-1) if device == "cpu" \
            else dict(params, device=device)
        print(f"  Params: {p}")
        print(f"  n_estimators={cfg['n_estimators']} (budget)  "
              f"early_stopping={cfg['early_stopping_rounds']}  "
              f"prune_tol={cfg['prune_tol']}")
        evals_result = {}
        model = xgb.train(
            p, dtrain,
            num_boost_round=cfg["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=esr,
            verbose_eval=100,
        )
    except Exception as e:
        if cfg.get("device", "cpu") == "cpu":
            raise
        print(f"  !! device={cfg['device']} failed ({e}); falling back to CPU")
        p = dict(params, device="cpu", n_jobs=-1)
        evals_result = {}
        model = xgb.train(
            p, dtrain,
            num_boost_round=cfg["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=esr,
            verbose_eval=100,
        )

    n_fitted = len(model.get_dump())
    print(f"\n  Fitted boosting rounds: {n_fitted} (budget {cfg['n_estimators']})")

    # ---- Data-driven pruning (size chosen on validation) ----------------- #
    # Prefix curve: validation F2 at the validation-chosen per-pT-bin working
    # point, for every prefix of the forest on a prune_grid_step grid. Tree
    # margins are accumulated in prune_grid_step-sized blocks, so the grid
    # costs one full pass over the forest, not one prediction per grid point.
    # The smallest prefix within prune_tol of the best prefix is kept; the
    # booster is then sliced to it.
    print("\n" + "=" * 70 + "\nTREE PRUNING (validation)\n" + "=" * 70)
    pt_val_raw = 10 ** X_val[:, pt_feat_idx]
    base_margin = _base_logit_from_booster(model)
    step = cfg["prune_grid_step"]
    # grid values must be block-aligned (multiples of step) so the block-wise
    # margin accumulation visits each of them; plus the full forest length.
    g0 = step * max(1, int(np.ceil(cfg["prune_min_trees"] / step)))
    grid = sorted(set(range(g0, n_fitted, step)) | {n_fitted})
    margins_at_grid = {}
    t0 = time.perf_counter()
    dval_margin = xgb.DMatrix(X_val)
    run = np.full(len(y_val), base_margin, dtype=np.float64)
    for t0_tree in range(0, n_fitted, step):
        t1_tree = min(t0_tree + step, n_fitted)
        run += model.predict(
            dval_margin, iteration_range=(t0_tree, t1_tree), output_margin=True
        ) - base_margin
        if t1_tree in grid:
            margins_at_grid[t1_tree] = run.copy()
    assert set(margins_at_grid) == set(grid)
    del dval_margin
    print(f"  Prefix margins accumulated in {time.perf_counter() - t0:.0f}s "
          f"({len(grid)} grid points)")

    def _val_f2_binned(scores):
        wp = pf.find_f2_threshold_binned(
            y_val, scores, pt_val_raw,
            cfg["pt_threshold_edges"], cfg["min_bin_signal"],
        )
        dec = pf.apply_binned_thresholds(
            scores, pt_val_raw, cfg["pt_threshold_edges"],
            [b[2] for b in wp["bins"]],
        )
        tn, fp, fn, tp = confusion_matrix(y_val, dec, labels=[0, 1]).ravel()
        _, _, _, _, f2 = pf.calculate_metrics((tp, fp, fn, tn))
        return f2

    size_curve = []
    best_f2, best_n = -1.0, n_fitted
    for n in grid:
        s = 1.0 / (1.0 + np.exp(-margins_at_grid[n]))
        f2 = _val_f2_binned(s)
        size_curve.append((n, f2))
        if f2 > best_f2:
            best_f2, best_n = f2, n
    print("  Size curve (val F2 @ per-bin WP): "
          + " ".join(f"{n}:{f2:.5f}" for n, f2 in size_curve))
    chosen_n = min(n for n, f2 in size_curve if f2 >= best_f2 - cfg["prune_tol"])
    print(f"  Best prefix: {best_n} (val F2 {best_f2:.5f}); "
          f"tolerance {cfg['prune_tol']} -> keeping {chosen_n} trees")

    # Slice the booster to the chosen prefix.
    model = model[0:chosen_n]
    assert len(model.get_dump()) == chosen_n
    # Sanity: the sliced booster's predictions must reproduce the recorded
    # prefix scores (catches any base-score bookkeeping mistake).
    s_val = model.predict(dval)
    d_max = np.abs(
        s_val - 1.0 / (1.0 + np.exp(-margins_at_grid[chosen_n]))
    ).max()
    margins_at_grid.clear()
    gc.collect()
    assert d_max < 1e-5, f"sliced booster deviates from prefix margins: {d_max:.2e}"
    print(f"  Sliced booster: {chosen_n} trees (prefix-consistency {d_max:.2e})")

    # ---- Working points (validation only, frozen for test) --------------- #
    # Every decision taken after this block uses thresholds derived on the
    # validation split of the pruned model; the test split is evaluated
    # exactly once with these frozen working points.
    print("\n" + "=" * 70 + "\nWORKING POINTS (validation)\n" + "=" * 70)
    wp = pf.find_f2_threshold_binned(
        y_val, s_val, pt_val_raw, cfg["pt_threshold_edges"], cfg["min_bin_signal"]
    )
    th1, th2 = wp["global_f1"], wp["global_f2"]
    bin_edges = cfg["pt_threshold_edges"]
    bin_thrs = [b[2] for b in wp["bins"]]
    print(f"  Global F1 threshold: {th1:.6f}   Global F2 threshold: {th2:.6f}")
    print(f"  Per-pT-bin F2 set points (min_bin_signal={cfg['min_bin_signal']}):")
    for lo, hi, bthr, n_sig, n_rows, fb in wp["bins"]:
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.0f}"
        tag = "  (fallback->global)" if fb else ""
        print(f"    pT [{lo:6.1f}, {hi_s:>4s}) GeV: thr={bthr:.6f}  "
              f"n_sig={n_sig}  n={n_rows}{tag}")

    # ---- Evaluation (test split, frozen working points) ------------------ #
    print("\n" + "=" * 70 + "\nEVALUATION (test, frozen WPs)\n" + "=" * 70)

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    y_true = y_test.astype(np.int32)
    pt_test_raw = 10 ** X_test[:, pt_feat_idx]

    prauc = average_precision_score(y_true, y_pred)
    ra = roc_auc_score(y_true, y_pred)
    print(f"\n  Test ROC AUC: {ra:.4f}  |  Test PR AUC: {prauc:.4f}")

    # ROC + PR curves
    roc_auc_val, pr_auc_val, prec_arr, rec_arr, thresholds = pf.plot_roc_pr(
        y_true, y_pred, cfg["output_dir"]
    )

    # Decisions with the frozen working points: global F2 and per-pT-bin F2.
    dec_global = y_pred >= th2
    dec_bins = pf.apply_binned_thresholds(y_pred, pt_test_raw, bin_edges, bin_thrs)

    tn, fp, fn, tp = confusion_matrix(y_true, dec_global, labels=[0, 1]).ravel()
    p_g, r_g, a_g, f1_g, f2_g = pf.calculate_metrics((tp, fp, fn, tn))
    tn, fp, fn, tp = confusion_matrix(y_true, dec_bins, labels=[0, 1]).ravel()
    p_b, r_b, a_b, f1_b, f2_b = pf.calculate_metrics((tp, fp, fn, tn))
    print(f"\n  Test @ global F2 WP ({th2:.6f}): P={p_g:.4f} R={r_g:.4f} F2={f2_g:.4f}")
    print(f"  Test @ per-bin F2 WPs:           P={p_b:.4f} R={r_b:.4f} F2={f2_b:.4f}")

    # Classification report + confusion matrices. confusion_matrix.png holds
    # the deployment configuration (per-pT-bin set points); the global-WP
    # matrix is kept alongside for reference.
    print(f"\n{classification_report(y_true, dec_bins.astype(int), digits=4)}")
    cm_bins, _ = pf.plot_confusion_matrix(
        y_true, None, 0, cfg["output_dir"], decisions=dec_bins,
        title_suffix="(per-pT-bin F2 working points)",
    )
    print(cm_bins)
    pf.plot_confusion_matrix(
        y_true, y_pred, th2, cfg["output_dir"],
        title_suffix="(global F2 working point)",
        filename="confusion_matrix_globalthr.png",
    )

    # Per-pT-bin evaluation (decisions use the per-bin set points)
    print("\n" + "=" * 70 + "\nPER-pT-BIN PERFORMANCE\n" + "=" * 70)
    pf.evaluate_pt_bins(y_true, y_pred, pt_test_raw, th2, cfg["output_dir"],
                        decisions=dec_bins)

    # Per-file evaluation
    print("\n" + "=" * 70 + "\nPER-FILE PERFORMANCE\n" + "=" * 70)
    for fi, fname in enumerate(files):
        fm = l_test == fi
        if fm.sum() == 0:
            continue
        yf, pf_ = y_true[fm], y_pred[fm]
        bf = dec_bins[fm].astype(int)
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
    f2_at_chosen = dict(size_curve)[chosen_n]
    with open(cfg["output_dir"] + "thresholds.txt", "w") as fo:
        fo.write(f"F1_Threshold: {th1}\nF2_Threshold: {th2}\n")
        fo.write(f"Test_ROC_AUC: {ra}\nTest_PR_AUC: {prauc}\n")
        fo.write(f"Test_F2_global_WP: {f2_g}\nTest_F2_perbin_WP: {f2_b}\n")
        fo.write(f"Split: evt10 (event-level, train 60 / val 10 / test 30)\n")
        fo.write(f"Tree_budget: {cfg['n_estimators']}\n")
        fo.write(f"Chosen_trees: {chosen_n} (prune_tol={cfg['prune_tol']}, "
                 f"val F2 {f2_at_chosen:.6f}, best prefix {best_n} at {best_f2:.6f})\n")
        fo.write(f"Signal_boost: {cfg['signal_boost']}\n")
        fo.write("pT-bin F2 set points (validation-derived):\n")
        for lo, hi, bthr, n_sig, n_rows, fb in wp["bins"]:
            hi_s = "inf" if not np.isfinite(hi) else f"{hi:.1f}"
            tag = "  [fallback->global]" if fb else ""
            fo.write(f"  pT [{lo:.1f}, {hi_s}): thr={bthr:.6f}  "
                     f"n_val_sig={n_sig}{tag}\n")
        fo.write(f"Model: XGBoost\n")

    import json as _json

    with open(cfg["output_dir"] + "thresholds.json", "w") as fo:
        _json.dump(
            {
                "split": "evt10",
                "tree_budget": cfg["n_estimators"],
                "chosen_trees": chosen_n,
                "prune_tol": cfg["prune_tol"],
                "val_f2_at_chosen": f2_at_chosen,
                "global_f1_threshold": th1,
                "global_f2_threshold": th2,
                "pt_bin_edges": bin_edges,
                "pt_bin_f2_thresholds": bin_thrs,
                "test": {
                    "roc_auc": ra,
                    "pr_auc": prauc,
                    "f2_global_wp": f2_g,
                    "f2_perbin_wp": f2_b,
                    "precision_perbin_wp": p_b,
                    "recall_perbin_wp": r_b,
                },
                "model": "XGBoost",
            },
            fo,
            indent=2,
        )
    print(f"  Saved: {cfg['output_dir']}thresholds.txt / thresholds.json")

    # ---- Save model ----------------------------------------------------- #
    model_path = cfg["output_dir"] + "model.json"
    model.save_model(model_path)
    print(f"\n  Saved model: {model_path}")
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



    # ---- Re-score previous production forest on the same split ----------- #
    ref_path = cfg.get("ref_model_path")
    if ref_path and os.path.isfile(ref_path):
        print("\n" + "=" * 70 +
              "\nREFERENCE FOREST RE-SCORE (same evt10 split, val-derived WPs)\n"
              + "=" * 70)
        pf.rescore_reference_forest(
            ref_path, X_val, y_val, pt_val_raw, X_test, y_test, pt_test_raw, cfg
        )
    else:
        print(f"\n  Reference forest not found ({ref_path}), skipping re-score")

    # ---- Re-score DNN ONNX on exact same test set --------------------- #
    dnn_onnx = cfg.get("dnn_onnx_path", "io/archive/pixel_dnn_output/model_standard.onnx")
    if os.path.isfile(dnn_onnx):
        print("\n" + "=" * 70 + "\nDNN RE-SCORE (exact same test set)\n" + "=" * 70)
        rescore_dnn(dnn_onnx, X_val, y_val, pt_val_raw,
                    X_test, y_test, pt_test_raw, l_test, files, cfg)
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


def rescore_dnn(dnn_onnx_path, X_val, y_val, pt_val_raw,
                X_test, y_test, pt_test_raw, l_test, files, cfg):
    """Re-score the DNN ONNX model on the exact same test set as XGBoost.

    Working points are derived on the validation split (frozen numbers, like
    the forest's). Skipped with a warning when the ONNX input width does not
    match the current feature count."""
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
    n_in = sess.get_inputs()[0].shape[-1]
    if n_in != X_test.shape[1]:
        print(f"  DNN expects {n_in} features but X has {X_test.shape[1]}; "
              "skipping re-score (stale model for the pruned feature set).")
        return

    # Run in batches
    def _score(Xm):
        bs = 8192
        out = []
        for i in range(0, len(Xm), bs):
            chunk = Xm[i : i + bs].astype(np.float32)
            out.append(sess.run(None, {input_name: chunk})[0])
        return np.concatenate(out).ravel()

    import time as _time

    t0 = _time.perf_counter()
    y_pred_dnn = _score(X_test)
    dt = _time.perf_counter() - t0
    print(f"  DNN inference: {dt:.1f}s")

    s_val_dnn = _score(X_val)
    th1, _, th2, _ = pf.find_f2_threshold(y_val, s_val_dnn)
    print(f"  DNN F2 threshold (validation): {th2:.6f}")

    y_true = y_test.astype(np.int32)
    prauc = average_precision_score(y_true, y_pred_dnn)
    ra = roc_auc_score(y_true, y_pred_dnn)
    print(f"  DNN ROC-AUC={ra:.4f}  PR-AUC={prauc:.4f}")

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
        fo.write(f"F2_Threshold: {th2}\n")
        fo.write(f"TN={tn} FP={fp} FN={fn} TP={tp}\n")
        fo.write(f"Precision={p:.6f} Recall={r:.6f} F2={f2:.6f}\n")


if __name__ == "__main__":
    main()
