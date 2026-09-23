#!/usr/bin/env python3
"""
make_summary.py - result tables of the production forests (model_version)
and of their predecessors (ref_version, the flavours' ref_model_path,
re-scored on the same test split), from the training records; used for the
README.

Per model: trees, nodes, .bin size, CMSSW inference cost (mean internal-node
visits per test track), test ROC-AUC / PR-AUC and precision / recall / F2 /
fake rejection at the per-pT-bin working points; per working-point bin: the
test performance of the production model.

Usage:
    python make_summary.py --pixel-data-dir DIR --seeds-data-dir DIR [--cache-dir DIR] [--json out.json]
"""

import argparse
import json
import os
import sys

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "oi"))
sys.path.insert(0, os.path.join(HERE, "tuning"))

import check_consistency as cc  # noqa: E402
import forest_pipeline as fp  # noqa: E402
import pixel_features as pf  # noqa: E402
from forest_scan import visits_per_tree  # noqa: E402


def test_split(cfg, n_cost=20000):
    X, y, fl, ev, names, _, _ = fp.load_features(cfg)
    X, names = fp.select_features(X, names, cfg["drop_features"], cfg["production_features"])
    _, va, te = pf.evt10_split(ev)
    idx = np.nonzero(te)[0]
    return X[idx], y[idx].astype(np.int32), 10 ** X[idx, names.index(cfg["pt_feature"])], names


def evaluate(bst, X, y, pt, th):
    s = bst.predict(xgb.DMatrix(X))
    dec = pf.apply_binned_thresholds(s, pt, th["pt_bin_edges"], th["pt_bin_f2_thresholds"])
    from sklearn.metrics import average_precision_score, roc_auc_score

    return dict(roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
                **fp.decision_metrics(y, dec))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fp.add_data_dir_arg(ap, "pixel", flag="--pixel-data-dir")
    fp.add_data_dir_arg(ap, "seeds", flag="--seeds-data-dir")
    fp.add_cache_dir_arg(ap)
    ap.add_argument("--json")
    a = ap.parse_args()
    data_dirs = dict(pixel=a.pixel_data_dir, seeds=a.seeds_data_dir)
    out = {}
    for cfg, _ in cc.flavours():
        cfg = dict(cfg, data_dir=data_dirs[cfg["data_chain"]], cache_dir=a.cache_dir)
        ref, prod = cfg["ref_version"], cfg["model_version"]
        d = os.path.join(cfg["output_dir"], "")
        th3 = json.load(open(d + "thresholds.json"))
        ref_dir = os.path.dirname(cfg["ref_model_path"])
        th2 = json.load(open(os.path.join(ref_dir, "thresholds.json")))
        X, y, pt, names = test_split(cfg)
        rng = np.random.default_rng(0)
        sample = rng.choice(len(X), size=min(20000, len(X)), replace=False)
        row = {}
        for tag, model_dir, th in ((ref, ref_dir, th2), (prod, cfg["output_dir"], th3)):
            bst = xgb.Booster()
            bst.load_model(os.path.join(model_dir, "model.json"))
            forest = fp.read_compact_bin(os.path.join(model_dir, "model_compact.bin"))
            # the reference is evaluated with its own deployed working points (its thresholds.json)
            m = evaluate(bst, X, y, pt, th)
            m.update(trees=int(len(forest["roots"])), nodes=int(len(forest["feat"])),
                     bin_mb=os.path.getsize(os.path.join(model_dir, "model_compact.bin")) / 1e6,
                     node_visits_per_track=float(visits_per_tree(forest, X[sample]).sum()))
            row[tag] = m
        row[f"{prod}_per_wp_bin"] = th3["test"]["per_wp_bin"]
        row[f"{prod}_thresholds"] = dict(edges=th3["pt_bin_edges"], thresholds=th3["pt_bin_f2_thresholds"])
        out[cfg["name"]] = row

    print("| Model | Version | Trees | Nodes | .bin [MB] | Node visits / track | ROC-AUC | PR-AUC | Precision | Recall | F2 | Fake rejection | Fakes kept |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, row in out.items():
        for tag in (ref, prod):
            m = row[tag]
            print(f"| {name} | {tag} | {m['trees']} | {m['nodes']:,} | {m['bin_mb']:.1f} | {m['node_visits_per_track']:,.0f} | "
                  f"{m['roc_auc']:.5f} | {m['pr_auc']:.5f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f2']:.4f} | "
                  f"{m['fake_rejection']:.4f} | {m['fp']:,} |")
    print(f"\n{prod} test performance per working-point bin (per-bin WPs):\n")
    print("| Model | pT bin [GeV] | Threshold | Tracks | Signal | Precision | Recall | Fake rejection |")
    print("|---|---|---|---|---|---|---|---|")
    for name, row in out.items():
        for b, thr in zip(row[f"{prod}_per_wp_bin"], row[f"{prod}_thresholds"]["thresholds"]):
            hi = f"{b['hi']:g}" if b["hi"] is not None else "inf"
            print(f"| {name} | [{b['lo']:g}, {hi}) | {thr:.3f} | {b['n']:,} | {b['n_sig']:,} | {b['precision']:.4f} | "
                  f"{b['recall']:.4f} | {b['fake_rejection']:.4f} |")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
