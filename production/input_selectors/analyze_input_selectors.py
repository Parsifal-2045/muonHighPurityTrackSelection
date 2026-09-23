#!/usr/bin/env python3
"""
analyze_input_selectors.py - how many genuine muon tracks the L1TkMuon input
selectors of the HP forests keep, from NANO files of cmsRun jobs customised
with input_selector_study_cff.py.

Denominator ("reachable" muons): muon TrackingParticles (|pdgId| = 13,
pT > 2 GeV, |eta| < 2.4) associated to a track of the selector INPUT
(all_pixel_tracks / all_lst_seeds) that have their own L1TkMuon: one within
dR < L1_DR_OWN = 0.05 (L1TkMu_eta/phi are the L1 tracker-track coordinates;
a muon's own L1 track lies within ~0.01, while candidates at 0.05-0.3 were
found to be other particles, with cm-level dz and large pT mismatches). The
number reachable with the loose dR < 0.3 cone is reported for reference. Efficiency of a collection =
fraction of reachable muons it still contains; also reported: tracks and
unmatched (fake) tracks per event, and - with --io-model - the muons and fakes
that survive the IO HP forest at its per-pT-bin working points.

Usage:
    python analyze_input_selectors.py --chain pixel --nano A.root [B.root ...] --label A [B ...]
        [--io-model ../io/pixel] [--json out.json]
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

import forest_pipeline as fp  # noqa: E402
import pixel_features as pf  # noqa: E402

PT_BINS = [2.0, 5.0, 10.0, 25.0, 50.0, np.inf]
L1_DR_OWN = 0.05
L1_DR_LOOSE = 0.3
COLLECTIONS = {
    "pixel": dict(input="all_pixel_tracks",
                  selected=["muon_pixel_tracks", "muon_pixel_tracks_keep2", "muon_pixel_tracks_keep3"],
                  hp=["muon_pixel_tracks", "muon_pixel_tracks_keep2", "muon_pixel_tracks_keep3"]),
    "seeds": dict(input="all_lst_seeds",
                  selected=["muon_seeds", "muon_seeds_keep3", "muon_seeds_keep4", "muon_general_tracks"],
                  hp=["muon_general_tracks"]),
}


def tp_keys(arr, name, evt):
    """Per event: set of muon TPs (identified by their kinematics) matched by
    a track of table `name`."""
    m = (arr[f"{name}_matched"] == 1) & (abs(arr[f"{name}_tpPdgId"]) == 13)
    pt, eta, phi = (ak.to_list(arr[f"{name}_tp{v}"][m][evt]) for v in ("Pt", "Eta", "Phi"))
    return {(round(a, 5), round(b, 5), round(c, 5)) for a, b, c in zip(pt, eta, phi)}


def reachable(arr, name, evt, dr_max=L1_DR_OWN):
    l1eta = np.asarray(arr["L1TkMu_eta"][evt]); l1phi = np.asarray(arr["L1TkMu_phi"][evt])
    out = set()
    for k in tp_keys(arr, name, evt):
        pt, eta, phi = k
        if pt < PT_BINS[0] or abs(eta) > 2.4 or len(l1eta) == 0:
            continue
        dphi = (phi - l1phi + np.pi) % (2 * np.pi) - np.pi
        if np.min((eta - l1eta) ** 2 + dphi ** 2) < dr_max ** 2:
            out.add(k)
    return out


def hp_pass(arr, prefix, model_dir):
    """Per-track HP decision of the IO forest on table `prefix` (jagged)."""
    th = json.load(open(os.path.join(model_dir, "thresholds.json")))
    bst = xgb.Booster()
    bst.load_model(os.path.join(model_dir, "model.json"))
    X, _, _, _, names = pf.build_dataset(arr, np.zeros(len(arr), dtype=np.int32), event_ids=np.arange(len(arr)),
                                         prefix=prefix)
    X, names = fp.select_features(X, names, pf.io_drop_features(prefix), pf.io_production_features(prefix))
    s = bst.predict(xgb.DMatrix(X))
    pt = 10 ** X[:, names.index(f"{prefix}_pt")]
    dec = pf.apply_binned_thresholds(s, pt, th["pt_bin_edges"], th["pt_bin_f2_thresholds"])
    counts = ak.num(arr[f"{prefix}_pt"])
    if len(dec) != ak.sum(counts):
        raise RuntimeError("non-finite feature rows dropped; cannot align HP decisions")
    return ak.unflatten(dec, counts)


def analyse(path, chain, model_dir):
    cols = COLLECTIONS[chain]
    with uproot.open(path) as f:
        tree = f["Events"]
        br = [b for b in tree.keys() if b.split("_")[0] in ("L1TkMu", "L1TkMuStub", "run", "event", "luminosityBlock")
              or any(b.startswith(c + "_") for c in [cols["input"]] + cols["selected"])]
        arr = tree.arrays(br)
    hp = {c: hp_pass(arr, c, model_dir) for c in cols["hp"]} if model_dir else {}
    n_ev = len(arr)
    reach_pt, found = [], {c: [] for c in cols["selected"]}
    n_loose = 0
    found_hp = {c: [] for c in hp}
    stats = {c: dict(tracks=int(ak.sum(ak.num(arr[f"{c}_pt"]))),
                     fakes=int(ak.sum(arr[f"{c}_matched"] == 0))) for c in [cols["input"]] + cols["selected"]}
    for c in hp:
        stats[c]["fakes_after_hp"] = int(ak.sum(hp[c] & (arr[f"{c}_matched"] == 0)))
        stats[c]["tracks_after_hp"] = int(ak.sum(hp[c]))
    for e in range(n_ev):
        r = reachable(arr, cols["input"], e)
        n_loose += len(reachable(arr, cols["input"], e, L1_DR_LOOSE))
        sets = {c: tp_keys(arr, c, e) for c in cols["selected"]}
        hp_sets = {}
        for c in hp:
            m = (arr[f"{c}_matched"][e] == 1) & (abs(arr[f"{c}_tpPdgId"][e]) == 13) & hp[c][e]
            hp_sets[c] = {(round(a, 5), round(b, 5), round(d, 5)) for a, b, d in zip(
                ak.to_list(arr[f"{c}_tpPt"][e][m]), ak.to_list(arr[f"{c}_tpEta"][e][m]), ak.to_list(arr[f"{c}_tpPhi"][e][m]))}
        for k in r:
            reach_pt.append(k[0])
            for c in cols["selected"]:
                found[c].append(k in sets[c])
            for c in hp:
                found_hp[c].append(k in hp_sets[c])
    reach_pt = np.asarray(reach_pt)
    res = dict(file=os.path.basename(path), events=n_ev, reachable_muons=int(len(reach_pt)), reachable_loose_cone=n_loose,
               per_event=stats, efficiency={}, efficiency_after_hp={})
    edges = PT_BINS
    for c, v in list(found.items()) + [(c + "+HP", v) for c, v in found_hp.items()]:
        v = np.asarray(v, dtype=bool)
        per_bin = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (reach_pt >= lo) & (reach_pt < hi)
            per_bin.append(dict(lo=lo, hi=None if np.isinf(hi) else hi, n=int(m.sum()),
                                eff=float(v[m].mean()) if m.any() else None, lost=int((~v[m]).sum())))
        (res["efficiency_after_hp"] if c.endswith("+HP") else res["efficiency"])[c] = dict(
            overall=float(v.mean()) if len(v) else None, lost=int((~v).sum()), per_pt_bin=per_bin)
    return res


def print_result(label, r):
    print(f"\n=== {label}: {r['file']}\n    {r['events']} events, {r['reachable_muons']} reachable muons with their own "
          f"L1TkMu ({r['reachable_loose_cone']} with any L1TkMu within dR < {L1_DR_LOOSE})")
    head = "    " + f"{'collection':32s}{'eff':>8s}{'lost':>6s}" + "".join(
        f"{f'{lo:g}-{hi:g}' if hi else f'>{lo:g}':>10s}" for lo, hi in zip(PT_BINS[:-1], [*PT_BINS[1:-1], None]))
    print(head + f"{'trk/evt':>9s}{'fake/evt':>9s}")
    for grp in ("efficiency", "efficiency_after_hp"):
        for c, e in r[grp].items():
            base = c.replace("+HP", "")
            st = r["per_event"][base]
            ntrk = st["tracks_after_hp"] if c.endswith("+HP") else st["tracks"]
            nfk = st["fakes_after_hp"] if c.endswith("+HP") else st["fakes"]
            bins = "".join(f"{b['eff']:>10.4f}" if b["eff"] is not None else f"{'-':>10s}" for b in e["per_pt_bin"])
            print(f"    {c:32s}{e['overall']:>8.4f}{e['lost']:>6d}{bins}{ntrk / r['events']:>9.2f}{nfk / r['events']:>9.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", choices=COLLECTIONS, required=True)
    ap.add_argument("--nano", nargs="+", required=True)
    ap.add_argument("--label", nargs="+")
    ap.add_argument("--io-model")
    ap.add_argument("--json")
    a = ap.parse_args()
    labels = a.label or a.nano
    out = {}
    for lab, path in zip(labels, a.nano):
        out[lab] = analyse(path, a.chain, a.io_model)
        print_result(lab, out[lab])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
