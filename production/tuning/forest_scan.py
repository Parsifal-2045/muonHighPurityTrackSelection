"""
forest_scan.py - validation-only hyperparameter scan of the HP forests.

For each configuration the forest is trained on the event folds that are
neither test (ev % 10 < 3, never touched here) nor the chosen validation fold,
and scored on the validation fold with the production criterion: F2 at the
validation-chosen per-pT-bin working points, for every tree prefix (size
curve). The CMSSW inference cost of every prefix is measured as the mean
number of internal-node visits per track (the serial-traversal work of
muonhp::CompactForest::evaluate), so configurations can be compared at equal
cost.

Folds: --val-fold 3 is the production split (train ev%10 in 4..9); any other
fold k in 4..9 gives an independent confirmation (train on 3..9 minus k).

Run:
    python tuning/forest_scan.py io_pixel --data-dir DIR [--cache-dir DIR] --configs base d8 d10 ... [--val-fold 3]
    python tuning/forest_scan.py --list
Results: tuning/results/<flavour>_fold<k>_<config>.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.dirname(HERE)
sys.path.insert(0, PROD)
sys.path.insert(0, os.path.join(PROD, "oi"))

import forest_pipeline as fp  # noqa: E402
import pixel_features as pf  # noqa: E402

# Candidate configurations: overrides of forest_pipeline.DEFAULTS
CONFIGS = {
    "base": {},                                              # production v2
    "d8": dict(max_depth=8),
    "d10": dict(max_depth=10),
    "d8_lr05": dict(max_depth=8, learning_rate=0.05),
    "d6_mcw10": dict(min_child_weight=10.0),
    "d8_mcw10": dict(max_depth=8, min_child_weight=10.0),
    "d8_cs06": dict(max_depth=8, colsample_bytree=0.6),
    "d8_l2_5": dict(max_depth=8, reg_lambda=5.0),
    "d8_bin512": dict(max_depth=8, max_bin=512),
    "d6_lr02": dict(learning_rate=0.2),
    "d8_sb1": dict(max_depth=8, signal_boost=1.0),
    "d8_nokin": dict(max_depth=8, kin_weight_max=1.0),
    # round 2 (depth dominates round 1)
    "d12": dict(max_depth=12),
    "d14": dict(max_depth=14),
    "d10_lr02": dict(max_depth=10, learning_rate=0.2),
    "d12_lr02": dict(max_depth=12, learning_rate=0.2),
    "d10_bin512": dict(max_depth=10, max_bin=512),
    "d10_mcw3": dict(max_depth=10, min_child_weight=3.0),
}


def flavour_cfg(name):
    if name == "io_pixel":
        import pixel_xgb as m
    elif name == "io_seeds":
        import seeds_xgb as m
    elif name == "oi_pixel":
        import OI_pixel_xgb as m
    elif name == "oi_general":
        import OI_general_xgb as m
    else:
        raise SystemExit(f"unknown flavour {name}")
    return fp.full_config(m.CFG)


def visits_per_tree(forest, X):
    """Mean internal-node visits per track, per tree (file order)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = len(X)
    feat = forest["feat"].astype(np.int64)
    out = np.empty(len(forest["roots"]))
    for t, r in enumerate(forest["roots"]):
        node = np.full(n, r, dtype=np.int64)
        f = np.full(n, feat[r])
        act = np.nonzero(f >= 0)[0]
        visits = 0
        while act.size:
            visits += act.size
            nd = node[act]
            node[act] = np.where(X[act, f[act]] < forest["val"][nd], forest["left"][nd], forest["right"][nd])
            f[act] = feat[node[act]]
            act = act[f[act] >= 0]
        out[t] = visits / n
    return out


def split_folds(ev, val_fold):
    d = ev.astype(np.int64) % 10
    assert 3 <= val_fold <= 9
    return (d >= 3) & (d != val_fold), d == val_fold


def run_one(flavour, cfg_name, val_fold, out_dir, data):
    cfg = flavour_cfg(flavour)
    cfg.update(CONFIGS[cfg_name])
    X, y, ev, names = data
    pt = 10 ** X[:, names.index(cfg["pt_feature"])]
    w = pf.compute_sample_weights(y, pt, signal_boost=cfg["signal_boost"], kin_weight_max=cfg["kin_weight_max"])
    tr, va = split_folds(ev, val_fold)
    t0 = time.time()
    bst, device, evals = fp.train_forest(cfg, X[tr], y[tr], w[tr], X[va], y[va], w[va], verbose_eval=False)
    t_train = time.time() - t0
    curve = fp.size_curve(bst, X[va], y[va], pt[va], cfg)
    chosen, best_n, best_f2 = fp.choose_size(curve, cfg["prune_tol"])
    rng = np.random.default_rng(0)
    sample = rng.choice(np.nonzero(va)[0], size=min(20000, int(va.sum())), replace=False)
    cost = np.cumsum(visits_per_tree(fp.forest_arrays(bst), X[sample]))
    res = dict(
        flavour=flavour, config=cfg_name, overrides=CONFIGS[cfg_name], val_fold=val_fold, device=device,
        train_seconds=round(t_train, 1), n_train=int(tr.sum()), n_val=int(va.sum()),
        size_curve=[[n, f2, float(cost[n - 1])] for n, f2 in curve],
        chosen_trees=chosen, chosen_val_f2=dict(curve)[chosen], chosen_cost=float(cost[chosen - 1]),
        best_prefix=best_n, best_val_f2=best_f2, best_cost=float(cost[best_n - 1]),
        val_logloss=evals["val"]["logloss"][::50], val_aucpr=evals["val"]["aucpr"][::50],
        train_logloss=evals["train"]["logloss"][::50],
    )
    path = os.path.join(out_dir, f"{flavour}_fold{val_fold}_{cfg_name}.json")
    with open(path, "w") as f:
        json.dump(res, f)
    print(f"{flavour:10s} fold{val_fold} {cfg_name:10s} train {t_train:6.0f}s | chosen {chosen:5d} trees "
          f"F2 {res['chosen_val_f2']:.5f} cost {res['chosen_cost']:7.0f} | best {best_n:5d} F2 {best_f2:.5f} "
          f"cost {res['best_cost']:7.0f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flavour", nargs="?")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--val-fold", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--list", action="store_true")
    fp.add_data_dir_arg(ap, required=False)
    fp.add_cache_dir_arg(ap)
    a = ap.parse_args()
    if a.list:
        for k, v in CONFIGS.items():
            print(f"{k:10s} {v}")
        return
    if not a.flavour or not a.data_dir:
        ap.error("a flavour and --data-dir are required (or --list)")
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(flavour_cfg(a.flavour), data_dir=a.data_dir, cache_dir=a.cache_dir)
    X, y, fl, ev, names, files, _ = fp.load_features(cfg)
    X, names = fp.select_features(X, names, cfg["drop_features"], cfg.get("production_features"))
    del fl
    for c in a.configs:  # folds use the per-file event index, as the pipeline's evt10 split
        run_one(a.flavour, c, a.val_fold, a.out, (X, y, ev, names))


if __name__ == "__main__":
    main()
