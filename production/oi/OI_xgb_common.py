"""
OI_xgb_common.py - shared runner for the OI (outside-in) high-purity XGBoost
forests (pixel-chain and seeds-chain variants).

Reuses the v2 pipeline machinery from pixel_xgb.py (identical training setup)
and the C++-faithful 26-feature extraction from OI_features.py:
identical XGBoost hyperparameters, identical 5000-tree budget with
validation-based pruning, identical evt10 event-level split, identical
per-pT-bin F2 working points (frozen on validation), identical export formats
(ONNX + model_compact.bin for the CMSSW forest selectors) and verification.

The two flavours differ only in the input sample (the l3_tk_OI track content
differs between the pixel-selector and seeds-selector HLT productions):
  - OI_pixel_xgb.py    : pixel-selector production ntuples
  - OI_general_xgb.py  : seeds-selector production ntuples

Run via the flavour scripts:  python OI_pixel_xgb.py  /  python OI_general_xgb.py
"""

import gc
import os
import sys
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

# The OI runner reuses the IO modules (pixel_features, pixel_xgb) from the
# production/ root directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pixel_features as pf
import OI_features as oif
from pixel_xgb import (
    _base_logit_from_booster,
    compute_permutation_importance_xgb,
    export_compact_bin,
    export_onnx,
    rescore_dnn,
    verify_compact_bin,
    verify_onnx,
)


# --------------------------------------------------------------------------- #
# Base configuration (flavour scripts override via CFG.update)
# --------------------------------------------------------------------------- #
CFG = dict(
    # overridden by the flavour scripts:
    data_dir=None,
    output_dir=None,
    cache_tag=None,        # /tmp cache prefix, must differ per flavour
    dnn_onnx_path=None,    # deployed OI DNN for the comparison re-score
    ref_model_path=None,   # no previous OI forest
    # Feature flags
    useStandaloneFeatures=True,
    # Feature pruning: None = full 26-feature set (round 1). The round-2
    # configs list the dropped features by name.
    drop_features=None,
    # Sample weights (identical to the IO forests)
    signal_boost=pf.SIGNAL_BOOST,
    kin_weight_max=pf.KIN_WEIGHT_MAX,
    # XGBoost hyperparameters (identical to pixel_xgb.py / seeds_xgb.py)
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    # Tree budget + data-driven pruning (identical to the IO v2 pipeline)
    n_estimators=5000,
    early_stopping_rounds=0,
    prune_tol=0.001,
    prune_grid_step=50,
    prune_min_trees=100,
    device="cuda",
    # Event-level split ("evt10")
    split_mode="evt10",
    split_seed_offset=0,
    # pT-binned working points [GeV]
    pt_threshold_edges=[0.0, 2.0, 5.0, 10.0, 50.0, 200.0],
    min_bin_signal=100,
    # ONNX
    onnx_opset=pf.ONNX_OPSET,
    compact_bin_fp16=False,
)


def load_data(cfg):
    """Load the flavour's input files and build the 26-feature OI matrix.

    Returns (X, y, fl, ev, feature_names, files). ev is a per-track event id
    (local to its input file) used for the event-level split. Caches to /tmp
    with an input-file manifest; a stale or manifest-mismatching cache is
    rebuilt."""
    import json

    cache_path = f"/tmp/OI_{cfg['cache_tag']}_data_cache.npz"
    names_path = f"/tmp/OI_{cfg['cache_tag']}_feature_names.json"
    files = sorted(
        os.path.join(cfg["data_dir"], f)
        for f in os.listdir(cfg["data_dir"])
        if os.path.isfile(os.path.join(cfg["data_dir"], f))
    )
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
            a = rf[oif.MAIN_BRANCH].arrays(
                oif.TK_BRANCHES + oif.L2_MU_VTX_BRANCHES
            )
            ne = len(a)
            total_ev += ne
            Xc, yc, lc, ec, fn = oif.build_dataset(
                a,
                np.full(ne, i),
                useStandaloneFeatures=cfg["useStandaloneFeatures"],
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


def main(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    _log_fh = pf.tee_log(cfg["output_dir"])  # full log stored in the model dir

    # ---- Load data ------------------------------------------------------- #
    X, y, fl, ev, feature_names, files = load_data(cfg)

    # ---- Feature pruning ------------------------------------------------ #
    drop = cfg.get("drop_features")
    if drop:
        keep_idx = [i for i, fn in enumerate(feature_names) if fn not in drop]
        X = X[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]
        print(f"\nFeature pruning: dropped {len(drop)} features, kept {len(feature_names)}")
        print(f"  Dropped: {drop}")
        print(f"  Kept: {feature_names}")
    else:
        print(f"\nUsing all {len(feature_names)} features (no pruning)")

    pt_feat_idx = feature_names.index("l3_tk_OI_pt")

    # ---- Sample weights (identical to the IO forests) -------------------- #
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

    # Low-pT mask (tracked for evaluations)
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
    del tr_m, va_m, te_m, ev
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
    # booster is then sliced to it. Identical scheme to pixel_xgb.py.
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
        wpb = pf.find_f2_threshold_binned(
            y_val, scores, pt_val_raw,
            cfg["pt_threshold_edges"], cfg["min_bin_signal"],
        )
        dec = pf.apply_binned_thresholds(
            scores, pt_val_raw, cfg["pt_threshold_edges"],
            [b[2] for b in wpb["bins"]],
        )
        tn_, fp_, fn_, tp_ = confusion_matrix(y_val, dec, labels=[0, 1]).ravel()
        _, _, _, _, f2 = pf.calculate_metrics((tp_, fp_, fn_, tn_))
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
    # ---- Feature importance (XGBoost gain + low-pT permutation) ---------- #
    print("\n" + "=" * 70 + "\nFEATURE IMPORTANCE\n" + "=" * 70)

    # Gain importance (built-in)
    print("\n[A] Full test set (XGBoost gain):")
    gain = model.get_score(importance_type="gain")
    cover = model.get_score(importance_type="cover")
    name_map = {f"f{i}": fn for i, fn in enumerate(feature_names)}
    gain_named = {name_map.get(k, k): v for k, v in gain.items()}
    cover_named = {name_map.get(k, k): v for k, v in cover.items()}

    sorted_gain = sorted(gain_named.items(), key=lambda x: -x[1])
    print(f"  {'Rank':<6}{'Feature':<55}{'Gain':>12}")
    print("  " + "-" * 73)
    for rank, (fn, gv) in enumerate(sorted_gain):
        cv = cover_named.get(fn, 0.0)
        print(f"  {rank + 1:<6}{fn:<55}{gv:>12.2f}  cover={cv:.1f}")

    with open(cfg["output_dir"] + "feature_importance.txt", "w") as fo:
        fo.write(f"Baseline PR-AUC: {prauc:.6f}\n")
        fo.write(f"Test ROC-AUC: {ra:.6f}\n\n")
        fo.write(f"{'Rank':<6}{'Feature':<55}{'Gain':>12}\n")
        fo.write("-" * 73 + "\n")
        for rank, (fn, gv) in enumerate(sorted_gain):
            cv = cover_named.get(fn, 0.0)
            fo.write(f"{rank + 1:<6}{fn:<55}{gv:>12.2f}  cover={cv:.1f}\n")

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

    # Low-pT subset permutation importance
    lpm = pt_test_raw < pf.LOW_PT_CUT
    print(
        f"\n[B] Low-pT subset: {lpm.sum()} tracks ({y_true[lpm].sum():.0f} signal)"
    )
    if lpm.sum() > 1000 and len(np.unique(y_true[lpm])) == 2:
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



    # ---- Re-score the deployed OI DNN on the same split ------------------ #
    # NOTE: the legacy OI DNNs were trained with unwrapped dphi and the
    # unnormalised matching score (see OI_features.py), so this comparison
    # evaluates the deployed combination (legacy DNN scores computed on
    # C++-convention features), not what the DNN would do with its own
    # training-convention inputs.
    dnn_onnx = cfg.get("dnn_onnx_path")
    if dnn_onnx and os.path.isfile(dnn_onnx):
        print("\n" + "=" * 70 +
              "\nOI DNN RE-SCORE (exact same test set, deployed-convention inputs)\n"
              + "=" * 70)
        rescore_dnn(dnn_onnx, X_val, y_val, pt_val_raw,
                    X_test, y_test, pt_test_raw, l_test, files, cfg)
    else:
        print(f"\n  OI DNN ONNX not found ({dnn_onnx}), skipping re-score")
