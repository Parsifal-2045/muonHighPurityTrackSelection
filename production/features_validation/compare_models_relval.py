#!/usr/bin/env python3
"""
compare_models_relval.py - compare two IO/OI forest versions on the tracks of
RelVal n-tuples (outside the training samples), threshold-free and at the
deployed working points.

For each model: ROC-AUC, PR-AUC, the signal efficiency and fakes per event at
its deployed per-pT-bin working points, and the fakes per event it keeps when
tuned to the same signal efficiency as the reference model (threshold scan on
the scores, i.e. equal-efficiency comparison of the rankings).

Usage:
    python compare_models_relval.py --family IO --prefix muon_pixel_tracks \
        --nano a.root [b.root ...] --models ../io/archive/pixel_xgb_output_33f_v2 ../io/pixel
"""

import argparse
import json
import os
import sys

import numpy as np
import uproot
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "oi"))

import forest_pipeline as fp  # noqa: E402
import OI_features as oif  # noqa: E402
import pixel_features as pf  # noqa: E402


def features(nano, family, prefix):
    if family == "IO":
        br = pf.tk_branches(prefix) + pf.L1TKMUON_BRANCHES + pf.STUB_BRANCHES
        build, kw = pf.build_dataset, dict(prefix=prefix)
        drop, abi, ptn = pf.io_drop_features(prefix), pf.io_production_features(prefix), f"{prefix}_pt"
    else:
        br = oif.TK_BRANCHES + oif.L2_MU_VTX_BRANCHES
        build, kw = oif.build_dataset, {}
        drop, abi, ptn = oif.OI_DROP_FEATURES, oif.OI_PRODUCTION_FEATURES, "l3_tk_OI_pt"
    Xs, ys, n_ev = [], [], 0
    for path in nano:
        with uproot.open(path) as f:
            arr = f["Events"].arrays(br)
        X, y, _, _, names = build(arr, np.zeros(len(arr), dtype=np.int32), event_ids=np.arange(len(arr)), **kw)
        X, names = fp.select_features(X, names, drop, abi)
        Xs.append(X); ys.append(y); n_ev += len(arr)
    X = np.concatenate(Xs)
    return X, np.concatenate(ys).astype(np.int32), 10 ** X[:, names.index(ptn)], n_ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["IO", "OI"], required=True)
    ap.add_argument("--prefix", default="muon_pixel_tracks")
    ap.add_argument("--nano", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="model dirs; the first is the reference")
    ap.add_argument("--json")
    a = ap.parse_args()
    X, y, pt, n_ev = features(a.nano, a.family, a.prefix)
    print(f"{len(y)} tracks ({y.sum()} signal) in {n_ev} events")
    out, ref_eff = {}, None
    for d in a.models:
        th = json.load(open(os.path.join(d, "thresholds.json")))
        bst = xgb.Booster()
        bst.load_model(os.path.join(d, "model.json"))
        s = bst.predict(xgb.DMatrix(X))
        dec = pf.apply_binned_thresholds(s, pt, th["pt_bin_edges"], th["pt_bin_f2_thresholds"])
        eff = float(dec[y == 1].mean())
        res = dict(model=fp.rel(d), roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
                   eff_at_wp=eff, fakes_per_event_at_wp=float((dec & (y == 0)).sum() / n_ev))
        ref_eff = eff if ref_eff is None else ref_eff
        # single threshold reaching the reference efficiency on these tracks
        thr = np.quantile(s[y == 1], 1.0 - ref_eff)
        res["fakes_per_event_at_ref_eff"] = float(((s >= thr) & (y == 0)).sum() / n_ev)
        out[os.path.basename(os.path.normpath(d))] = res
        print(f"  {os.path.basename(os.path.normpath(d)):32s} ROC-AUC {res['roc_auc']:.5f} PR-AUC {res['pr_auc']:.5f} | "
              f"deployed WPs: eff {eff:.4f}, fakes/evt {res['fakes_per_event_at_wp']:.3f} | "
              f"at eff {ref_eff:.4f}: fakes/evt {res['fakes_per_event_at_ref_eff']:.3f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
