#!/usr/bin/env python3
"""
summarize_scan.py - markdown summary of the forest hyperparameter scan
(results/*.json written by forest_scan.py): validation F2 at the per-bin
working points reachable within a CMSSW inference budget (node visits per
track), per flavour and configuration, plus the confirmation fold.

Usage:  python tuning/summarize_scan.py [--budgets 5000 10000 15000 20000]
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[2500, 5000, 10000, 15000, 20000])
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    res = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in glob.glob(os.path.join(HERE, "results", "*.json")):
        r = json.load(open(p))
        res[r["flavour"]][r["val_fold"]][r["config"]] = r
    for fl in sorted(res):
        folds = res[fl]
        cfgs = folds[3]
        print(f"\n### {fl} (validation fold 3, {len(cfgs)} configurations)\n")
        print("| Configuration | " + " | ".join(f"F2 @ <= {b:,} visits" for b in a.budgets)
              + " | Chosen size (trees / visits / F2) |")
        print("|---|" + "---|" * (len(a.budgets) + 1))
        rows = []
        for c, r in cfgs.items():
            sc = np.array(r["size_curve"])
            vals = [sc[sc[:, 2] <= b, 1].max() if (sc[:, 2] <= b).any() else np.nan for b in a.budgets]
            rows.append((np.nan_to_num(vals[-2]), c, vals, r))
        rows.sort(key=lambda t: -t[0])
        shown = rows[: a.top] + [t for t in rows[a.top:] if t[1] == "base"]
        for _, c, vals, r in shown:
            label = f"**{c}**" if c == "base" else c
            print(f"| {label} | " + " | ".join("-" if np.isnan(v) else f"{v:.5f}" for v in vals)
                  + f" | {r['chosen_trees']} / {r['chosen_cost']:,.0f} / {r['chosen_val_f2']:.5f} |")
        for fold, cf in sorted(folds.items()):
            if fold == 3:
                continue
            print(f"\nConfirmation on validation fold {fold}: " + ", ".join(
                f"{c} {r['chosen_val_f2']:.5f} ({r['chosen_trees']} trees, {r['chosen_cost']:,.0f} visits)"
                for c, r in sorted(cf.items(), key=lambda kv: -kv[1]["chosen_val_f2"])))


if __name__ == "__main__":
    main()
