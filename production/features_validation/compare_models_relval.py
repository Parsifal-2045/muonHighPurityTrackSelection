#!/usr/bin/env python3
"""
compare_models_relval.py - IO/OI forests on the tracks of RelVal n-tuples
(outside the training samples), at their deployed working points.

Every model directory is scored with its own feature list (thresholds.json),
so forests with different feature sets can be compared on the same tracks.
For each model: ROC-AUC, the fakes kept per event and the muon recall at the
deployed per-pT-bin working points, per working-point bin; for the IO family
the recall is given for all muon tracks and for the muons of the signal event
(generator-level muons; the rest are pileup muons).

Usage:
    python compare_models_relval.py --family IO --prefix muon_pixel_tracks \
        --nano a.root [b.root ...] --models ../io/archive/pixel_xgb_output_33f_dev5 ../io/pixel [--json out.json]
"""

import argparse
import json
import os
import sys

import numpy as np
import uproot
import xgboost as xgb
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "oi"))

import forest_pipeline as fp  # noqa: E402
import OI_features as oif  # noqa: E402
import pixel_features as pf  # noqa: E402


def features(nano, family, prefix):
    """(X with every feature a model may ask for, names, y, pT, signal-muon flag or None, events)."""
    if family == "IO":
        br = pf.io_branches(prefix)
        build = pf.build_dataset
        kw = dict(prefix=prefix, chamberMatch=True, signalMuonFlag=True)
        ptn = f"{prefix}_pt"
    else:
        br, build, kw, ptn = oif.TK_BRANCHES + oif.L2_MU_VTX_BRANCHES, oif.build_dataset, {}, "l3_tk_OI_pt"
    Xs, ys, n_ev = [], [], 0
    for path in nano:
        with uproot.open(path) as f:
            for arr in f["Events"].iterate(br, step_size=20000):
                X, y, _, _, names = build(arr, np.zeros(len(arr), dtype=np.int32), event_ids=np.arange(len(arr)), **kw)
                Xs.append(X); ys.append(y); n_ev += len(arr)
    X = np.concatenate(Xs)
    sig = X[:, names.index(pf.SIGNAL_MUON_FLAG)] > 0.5 if pf.SIGNAL_MUON_FLAG in names else None
    return X, names, np.concatenate(ys).astype(np.int32), 10 ** X[:, names.index(ptn)], sig, n_ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["IO", "OI"], required=True)
    ap.add_argument("--prefix", default="muon_pixel_tracks")
    ap.add_argument("--nano", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="model directories")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    X, names, y, pt, sig, n_ev = features(a.nano, a.family, a.prefix)
    mu, fake = y == 1, y == 0
    print(f"{len(y)} tracks in {n_ev} events: {mu.sum()} muon tracks" + (f" ({sig.sum()} of the signal event)" if sig is not None else "")
          + f", {fake.sum() / n_ev:.3f} fakes per event")
    for d in a.models:
        th = json.load(open(os.path.join(d, "thresholds.json")))
        bst = xgb.Booster()
        bst.load_model(os.path.join(d, "model.json"))
        s = bst.predict(xgb.DMatrix(X[:, [names.index(f) for f in th["feature_names"]]]))
        edges = th["pt_bin_edges"]
        dec = pf.apply_binned_thresholds(s, pt, edges, th["pt_bin_f2_thresholds"])
        bins = pf.pt_bins_from_edges(edges)
        per_bin = []
        for lo, hi in bins:
            m = (pt >= lo) & (pt < hi)
            per_bin.append(dict(lo=lo, hi=None if not np.isfinite(hi) else hi,
                                fakes_per_event=float((dec & fake & m).sum() / n_ev),
                                recall=float(dec[mu & m].mean()) if (mu & m).any() else None,
                                recall_signal_muons=float(dec[sig & m].mean()) if sig is not None and (sig & m).any() else None))
        tag = os.path.basename(os.path.normpath(d))
        res = dict(model=fp.rel(d), n_features=len(th["feature_names"]), trees=th["chosen_trees"],
                   roc_auc=float(roc_auc_score(y, s)), fakes_per_event_in=float(fake.sum() / n_ev),
                   fakes_per_event=float((dec & fake).sum() / n_ev), recall=float(dec[mu].mean()),
                   recall_signal_muons=float(dec[sig].mean()) if sig is not None else None, per_bin=per_bin)
        out[tag] = res
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"  {tag:30s} {res['n_features']:2d} features, {res['trees']:4d} trees | ROC-AUC {res['roc_auc']:.4f} | "
              f"fakes/evt {res['fakes_per_event']:.3f} | recall {res['recall']:.4f}"
              + (f", signal-event muons {res['recall_signal_muons']:.4f}" if sig is not None else ""))
        print("      per bin  fakes/evt: " + " ".join(f"{b['fakes_per_event']:.3f}" for b in per_bin)
              + " | recall: " + " ".join(fmt(b["recall"]) for b in per_bin)
              + (" | signal-event: " + " ".join(fmt(b["recall_signal_muons"]) for b in per_bin) if sig is not None else ""))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
