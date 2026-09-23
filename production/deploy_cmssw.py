#!/usr/bin/env python3
"""
deploy_cmssw.py - install the four production forests in a CMSSW area.

For every production model (io/pixel, io/seeds, oi/pixel, oi/general):
  * copies model_compact.bin to RecoMuon/L3TrackFinder/data/<IO|OI>/<name>.bin
    (the flavour config's cmssw_bin);
  * rewrites the working points of its selector in the cfi from
    thresholds.json (modelPath, decisionThreshold, ptBinEdges,
    decisionThresholds with per-bin comments, nFeatures) and the provenance
    comment above them;
then removes every other .bin under RecoMuon/L3TrackFinder/data and writes
RecoMuon/L3TrackFinder/data/README.md describing the deployed files.
Run check_consistency.py --cmssw-src afterwards.

Usage:
    python deploy_cmssw.py --cmssw-src /path/to/CMSSW_X/src [--dry-run]
"""

import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "oi"))

import check_consistency as cc  # noqa: E402  (flavour list, md5)
import pixel_features as pf  # noqa: E402

CMSSW_DATA_DIR = "RecoMuon/L3TrackFinder/data"
CHAIN = {
    "IO pixel": ("pixel-track chain (phase2MuonPixelTracksSelector, ngtScouting)", "hltPhase2MuonPixelTracks",
                 "MuonIOTracksForestSelector"),
    "IO seeds": ("seeds chain (phase2MuonSeedsSelector)", "hltPhase2MuonIOTracks", "MuonIOTracksForestSelector"),
    "OI pixel": ("pixel-track chain (phase2MuonPixelTracksSelector, ngtScouting)",
                 "hltPhase2L3OIMuCtfWithMaterialTracks", "MuonOITracksForestSelector"),
    "OI general": ("seeds chain (phase2MuonSeedsSelector)", "hltPhase2L3OIMuCtfWithMaterialTracks",
                   "MuonOITracksForestSelector"),
}


def _fmt_thresholds(th, indent):
    lines = []
    for (lo, hi), thr, fb in zip(pf.pt_bins_from_edges(th["pt_bin_edges"]), th["pt_bin_f2_thresholds"],
                                 th["pt_bin_fallback"]):
        hi_s = "inf" if hi == float("inf") else f"{hi:g}"
        note = " - global F2 fallback (no background in validation)" if fb else ""
        lines.append(f"{indent}    {thr!r},  # pT [{lo:g}, {hi_s}){note}")
    return "cms.vdouble(\n" + "\n".join(lines) + f"\n{indent})"


def update_cfi_block(text, var, cfg, th, man):
    """Replace the working-point parameters inside the `var = ...(...)` block."""
    m = re.search(rf"^{re.escape(var)} = .*?(?=^\S)", text + "\nEND", re.S | re.M)
    if not m:
        raise KeyError(f"{var} not found")
    block = m.group(0)
    new = block
    prov = (f"    # {cfg['name']} forest {cfg['model_version']}: {th['chosen_trees']} trees, {th['n_features']} features, trained "
            f"{man['started'][:10]} (git {str(man['git_commit'])[:10]}); working points = validation per-pT-bin\n"
            f"    # F2 set points from production/{os.path.relpath(cfg['output_dir'], HERE)}/thresholds.json "
            f"(written by deploy_cmssw.py)\n")
    # comment-only lines inside the block are regenerated (provenance, per-bin labels)
    new = re.sub(r"\n[ \t]+#[^\n]*(?=\n)", "", new)
    new = re.sub(r"    modelPath = cms\.FileInPath\([^)]*\),\n",
                 lambda _: prov + f"    modelPath = cms.FileInPath('{cfg['cmssw_bin']}'),\n", new)
    new = re.sub(r"decisionThreshold = cms\.double\([^)]*\)", f"decisionThreshold = cms.double({th['global_f2_threshold']!r})", new)
    new = re.sub(r"ptBinEdges = cms\.vdouble\([^)]*\)",
                 f"ptBinEdges = cms.vdouble({', '.join(repr(float(e)) for e in th['pt_bin_edges'])})", new)
    new = re.sub(r"decisionThresholds = cms\.vdouble\(\n.*?\n    \)",
                 lambda _: "decisionThresholds = " + _fmt_thresholds(th, "    "), new, flags=re.S)
    new = re.sub(r"nFeatures = cms\.int32\(\d+\)", f"nFeatures = cms.int32({th['n_features']})", new)
    for p in ("modelPath", "decisionThreshold", "decisionThresholds"):
        if f"{p} = " not in new:
            raise ValueError(f"{var}: parameter {p} missing from the cfi block")
    if "ptBinEdges = " not in new and ".clone(" not in new.split("\n", 1)[0]:
        raise ValueError(f"{var}: ptBinEdges missing")
    return text.replace(block, new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmssw-src", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = os.path.abspath(a.cmssw_src)
    fl = cc.flavours()
    readme_rows = []
    cfi_texts = {}
    for cfg, var in fl:
        d = os.path.join(cfg["output_dir"], "")
        th = json.load(open(d + "thresholds.json"))
        man = json.load(open(d + "manifest.json"))
        if th["feature_names"] != list(cfg["production_features"]):
            raise SystemExit(f"{cfg['name']}: thresholds.json feature ABI differs from the production ABI")
        dst = os.path.join(src, cfg["cmssw_bin"])
        print(f"{cfg['name']}: {d}model_compact.bin -> {cfg['cmssw_bin']}")
        if not a.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(d + "model_compact.bin", dst)
        cfi = os.path.join(src, cfg["cmssw_cfi"])
        cfi_texts[cfi] = update_cfi_block(cfi_texts.get(cfi, open(cfi).read()), var, cfg, th, man)
        chain, tracks, plugin = CHAIN[cfg["name"]]
        t = th["test"]
        readme_rows.append(
            f"| `{os.path.relpath(cfg['cmssw_bin'], CMSSW_DATA_DIR)}` | {cfg['name']} | {chain} | `{tracks}` | "
            f"`{plugin}` via `{var}` | {th['n_features']} | {th['chosen_trees']} | "
            f"{t['roc_auc']:.5f} / {t['pr_auc']:.5f} | {t['perbin_wp']['precision']:.4f} / {t['perbin_wp']['recall']:.4f} "
            f"/ {t['perbin_wp']['fake_rejection']:.4f} | {man['started'][:10]} | `{str(man['git_commit'])[:10]}` | "
            f"`{th['compact_bin']['md5']}` |")
    for cfi, text in cfi_texts.items():
        print(f"  cfi updated: {os.path.relpath(cfi, src)}")
        if not a.dry_run:
            open(cfi, "w").write(text)
    keep = {os.path.join(src, cfg["cmssw_bin"]) for cfg, _ in fl}
    for root, _, files in os.walk(os.path.join(src, CMSSW_DATA_DIR)):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".bin") and p not in keep:
                print(f"  removing stale {os.path.relpath(p, src)}")
                if not a.dry_run:
                    os.remove(p)
    readme = f"""# Muon HLT high-purity track selectors: forest models

Compact XGBoost forests (`.bin`, format in `interface/CompactForest.h`) of the
Phase-2 muon HLT high-purity selections, one per track family and HLT chain:

- `IO/`: inside-out tracks, read by `MuonIOTracksForestSelector` (33 features,
  `interface/IOTrackSelectorFeatures.h`);
- `OI/`: outside-in tracks, read by `MuonOITracksForestSelector` (22 features,
  `interface/OITrackSelectorFeatures.h`).

File names give family, HLT chain and model version (deployed: `{fl[0][0]['model_version']}`; every
deployed retraining gets a new version, history in the training repository
README). The selectors validate every file against the feature extractor when
they load it.

| File | Selector | HLT chain | Input tracks | Plugin / cfi module | Features | Trees | Test ROC-AUC / PR-AUC | Test precision / recall / fake rejection (per-bin WPs) | Trained | Training git | md5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(readme_rows)}

Training, validation and working points: muonHighPurityTrackSelection
(`production/`, forest_pipeline.py). The cfi working points are written from
the trainings' `thresholds.json` by `production/deploy_cmssw.py`, and
`production/check_consistency.py --cmssw-src` checks that files, checksums and
cfi values agree with the training records.
"""
    if not a.dry_run:
        with open(os.path.join(src, CMSSW_DATA_DIR, "README.md"), "w") as f:
            f.write(readme)
    print("  wrote", os.path.join(CMSSW_DATA_DIR, "README.md"))


if __name__ == "__main__":
    main()
