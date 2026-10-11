#!/usr/bin/env python3
"""
make_summary.py - result tables of the production forests from their training
records (thresholds.json, manifest.json and the model files); used for the
README.

Per model: features, trees, nodes, .bin size, test ROC-AUC / PR-AUC and precision / recall / fake rejection at
the per-pT-bin working points on the test split of the training sample; per
working-point bin: threshold, test performance and the recall of the muons
the targets are defined on.

The test split of the training sample shares the pileup pool with the
training: the comparison between model generations and the reference numbers
are those of features_validation/compare_models_relval.py on RelVal events.

Usage:
    python make_summary.py [--json out.json]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "oi"))

import check_consistency as cc  # noqa: E402
import forest_pipeline as fp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for cfg, _ in cc.flavours():
        d = os.path.join(cfg["output_dir"], "")
        th = json.load(open(d + "thresholds.json"))
        man = json.load(open(d + "manifest.json"))
        forest = fp.read_compact_bin(d + "model_compact.bin")
        row = dict(version=cfg["model_version"], features=th["n_features"], trees=int(len(forest["roots"])),
                   nodes=int(len(forest["feat"])), bin_mb=os.path.getsize(d + "model_compact.bin") / 1e6,
                   trained=man["started"][:10], roc_auc=th["test"]["roc_auc"], pr_auc=th["test"]["pr_auc"],
                   **{k: th["test"]["perbin_wp"][k] for k in ("precision", "recall", "fake_rejection")},
                   per_wp_bin=th["test"]["per_wp_bin"], thresholds=th["pt_bin_f2_thresholds"],
                   targets=th["wp_recall_targets"], wp_reference=th.get("wp_reference"),
                   recall_of_reference=th.get("test_recall_of_reference"))
        out[cfg["name"]] = row

    print("| Model | Version | Features | Trees | Nodes | .bin [MB] | ROC-AUC | PR-AUC | Precision | Recall | Fake rejection |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in out.items():
        print(f"| {name} | {m['version']} | {m['features']} | {m['trees']} | {m['nodes']:,} | {m['bin_mb']:.2f} | "
              f"{m['roc_auc']:.5f} | {m['pr_auc']:.5f} | {m['precision']:.4f} | {m['recall']:.4f} | "
              f"{m['fake_rejection']:.4f} |")
    print("\nTest performance per working-point bin:\n")
    print("| Model | pT bin [GeV] | Target | Threshold | Tracks | Signal | Precision | Recall | Recall of the target muons | "
          "Fake rejection |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, m in out.items():
        ref = m["recall_of_reference"] or [None] * len(m["per_wp_bin"])
        for b, thr, tgt, rr in zip(m["per_wp_bin"], m["thresholds"], m["targets"], ref):
            hi = f"{b['hi']:g}" if b["hi"] is not None else "inf"
            print(f"| {name} | [{b['lo']:g}, {hi}) | {tgt:g} | {thr:.3f} | {b['n']:,} | {b['n_sig']:,} | {b['precision']:.4f} | "
                  f"{b['recall']:.4f} | {'-' if rr is None else f'{rr:.4f}'} | {b['fake_rejection']:.4f} |")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
