#!/usr/bin/env python3
"""
validate_cmssw.py - track-by-track cross-check of the CMSSW HP forest
selectors against the training code, for one HLT chain (IO + OI selector).

Inputs, from ONE cmsRun job (cmsDriver -s L1P2GT,HLT:75e33,NANO:@MUHLTTraining
with the chain's process modifier and dumpFeatures=True on its IO and OI
forest selectors):
  * the training NANO it wrote (same events, same track collections);
  * its log, holding one line per track and selector:
        MUONHP_FEATURES,<module label>,<run>,<lumi>,<event>,<track idx>,<f0..fN-1>,<score>

Per selector the script
  1. rebuilds the features from the NANO with the training extraction
     (pixel_features / OI_features + the production feature ABI) and joins
     them with the C++ dump on (run, lumi, event, track index);
  2. compares the features (max |diff| per feature, exact-match fraction);
  3. compares scores: C++ score vs the deployed model evaluated in Python
     (XGBoost on the Python features) and vs the .bin replay on the C++
     features (isolates the C++ inference from the feature extraction);
  4. counts selection-decision differences at the deployed working points.

Usage:
    python validate_cmssw.py --chain pixel --nano val_pixel_NANO.root --log val_pixel.log \
        --io-model ../io/pixel --oi-model ../oi/pixel [--json out.json]
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "oi"))

import forest_pipeline as fp  # noqa: E402
import OI_features as oif  # noqa: E402
import pixel_features as pf  # noqa: E402

CHAINS = {
    "pixel": dict(io_label="hltPhase2MuonPixelTracksHighPurityForest", io_prefix=pf.PIXEL_PREFIX),
    "seeds": dict(io_label="hltPhase2MuonIOTrackSelectionHighPurityForest", io_prefix=pf.SEEDS_PREFIX),
}
OI_LABEL = "hltPhase2L3OIMuonTrackSelectionHighPurity"


def parse_dump(log_path):
    rows = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            i = line.find("MUONHP_FEATURES,")
            if i < 0:
                continue
            parts = line[i:].strip().split(",")
            label = parts[1]
            key = tuple(int(p) for p in parts[2:6])
            vals = np.array([float(p) for p in parts[6:]], dtype=np.float32)
            rows.setdefault(label, {})[key] = vals
    return rows


def python_features(arr, prefix_field, build, build_kwargs, drop, expected):
    """Features of every track (keyed by run, lumi, event, track index)."""
    ids = np.stack([ak.to_numpy(arr[k]).astype(np.int64) for k in ("run", "luminosityBlock", "event")], axis=1)
    n_ev = len(arr)
    mask = arr[prefix_field] > 0
    counts = ak.to_numpy(ak.num(arr[prefix_field]))
    ev_idx = ak.to_numpy(ak.flatten(ak.unflatten(np.repeat(np.arange(n_ev), counts), counts)[mask]))
    trk_idx = ak.to_numpy(ak.flatten(ak.local_index(arr[prefix_field], axis=1)[mask]))
    X, y, _, _, names = build(arr, np.zeros(n_ev, dtype=np.int32), event_ids=np.arange(n_ev), **build_kwargs)
    if len(X) != len(ev_idx):
        raise RuntimeError("build_dataset dropped non-finite rows; track alignment lost")
    X, names = fp.select_features(X, names, drop, expected)
    keys = [tuple(ids[e]) + (int(t),) for e, t in zip(ev_idx, trk_idx)]
    return keys, X, y, names


def check_selector(title, dump, keys, X_py, y, names, model_dir, pt_idx):
    th = json.load(open(os.path.join(model_dir, "thresholds.json")))
    bst = xgb.Booster()
    bst.load_model(os.path.join(model_dir, "model.json"))
    forest = fp.read_compact_bin(os.path.join(model_dir, "model_compact.bin"))
    common = [k for k in keys if k in dump]
    only_py = len(keys) - len(common)
    only_cpp = len(set(dump) - set(keys))
    pos = {k: i for i, k in enumerate(keys)}
    idx = np.array([pos[k] for k in common])
    Xp = X_py[idx]
    Xc = np.stack([dump[k][:-1] for k in common])
    s_cpp = np.array([dump[k][-1] for k in common], dtype=np.float32)
    if Xc.shape[1] != Xp.shape[1]:
        raise RuntimeError(f"{title}: C++ dumps {Xc.shape[1]} features, training ABI has {Xp.shape[1]}")

    diff = np.abs(Xc.astype(np.float64) - Xp.astype(np.float64))
    per_feat = {n: dict(max_abs_diff=float(diff[:, j].max()), exact_fraction=float((diff[:, j] == 0).mean()))
                for j, n in enumerate(names)}
    s_py = bst.predict(xgb.DMatrix(Xp))
    s_bin_cpp_inputs = fp.compact_forest_predict(forest, Xc)
    pt = 10 ** Xp[:, pt_idx]
    thr = np.asarray(th["pt_bin_f2_thresholds"])[
        np.searchsorted(np.asarray(th["pt_bin_edges"]), pt, side="right") - 1]
    res = dict(
        selector=title, model_dir=fp.rel(model_dir), bin_md5=fp._md5(os.path.join(model_dir, "model_compact.bin")),
        tracks_matched=len(common), only_in_python=only_py, only_in_cmssw=only_cpp, signal_tracks=int(y[idx].sum()),
        features_max_abs_diff=float(diff.max()), features_exact_fraction=float((diff == 0).mean()),
        per_feature=per_feat,
        score_max_abs_diff_vs_python_model=float(np.abs(s_cpp - s_py).max()),
        score_max_abs_diff_vs_bin_replay_on_cpp_features=float(np.abs(s_cpp - s_bin_cpp_inputs).max()),
        decision_differences_vs_python=int(np.sum((s_cpp >= thr) != (s_py >= thr))),
        selected_cmssw=int(np.sum(s_cpp >= thr)),
    )
    worst = sorted(per_feat.items(), key=lambda kv: -kv[1]["max_abs_diff"])[:5]
    print(f"\n== {title}: {len(common)} tracks matched ({only_py} only in NANO, {only_cpp} only in the C++ dump), "
          f"{res['signal_tracks']} signal")
    print(f"   features: max|diff| {res['features_max_abs_diff']:.3g}, bit-identical fraction "
          f"{res['features_exact_fraction']:.6f}; largest: "
          + ", ".join(f"{n}={v['max_abs_diff']:.2g}" for n, v in worst))
    print(f"   score: C++ vs Python model max|diff| {res['score_max_abs_diff_vs_python_model']:.2e}; "
          f"C++ vs .bin replay on the C++ features {res['score_max_abs_diff_vs_bin_replay_on_cpp_features']:.2e}")
    print(f"   decisions at the deployed WPs: {res['selected_cmssw']} selected in CMSSW, "
          f"{res['decision_differences_vs_python']} differ from Python")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", choices=CHAINS, required=True)
    ap.add_argument("--nano", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--io-model", required=True)
    ap.add_argument("--oi-model", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    ch = CHAINS[a.chain]
    dump = parse_dump(a.log)
    for lab in (ch["io_label"], OI_LABEL):
        if lab not in dump:
            raise SystemExit(f"no MUONHP_FEATURES lines for {lab} in {a.log}")
    io_pfx = ch["io_prefix"]
    br = (pf.tk_branches(io_pfx) + pf.L1TKMUON_BRANCHES + pf.STUB_BRANCHES + oif.TK_BRANCHES
          + oif.L2_MU_VTX_BRANCHES + ["run", "luminosityBlock", "event"])
    with uproot.open(a.nano) as f:
        arr = f["Events"].arrays(sorted(set(br)))
    out = []
    keys, X, y, names = python_features(
        arr, f"{io_pfx}_pt", pf.build_dataset, dict(prefix=io_pfx), pf.io_drop_features(io_pfx),
        pf.io_production_features(io_pfx))
    out.append(check_selector(f"IO {a.chain} ({ch['io_label']})", dump[ch["io_label"]], keys, X, y, names,
                              a.io_model, names.index(f"{io_pfx}_pt")))
    keys, X, y, names = python_features(
        arr, "l3_tk_OI_pt", oif.build_dataset, {}, oif.OI_DROP_FEATURES, oif.OI_PRODUCTION_FEATURES)
    out.append(check_selector(f"OI {a.chain} ({OI_LABEL})", dump[OI_LABEL], keys, X, y, names, a.oi_model,
                              names.index("l3_tk_OI_pt")))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
