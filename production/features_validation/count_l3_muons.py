#!/usr/bin/env python3
"""
count_l3_muons.py - L3 muons before and after the muon identification in
NANO files made with l3_muon_tables_cff.py: fake muons per event (muon tracks
not matched to a muon TrackingParticle), muons per event and the efficiency
for the generated muons of the signal event (status 1, pT > 2 GeV,
|eta| < 2.4: a generated muon counts when an L3 muon is matched to its
TrackingParticle).

One line per --sample; run it on the same events processed with different
selector configurations (no selection, previous models, deployed models) to
compare them at the end of the chain.

Usage:
    python count_l3_muons.py --sample "v1 pixel ZMM+Bs" v1_pixel_ZMM.root v1_pixel_Bs.root \
                             --sample "dev5 pixel ZMM+Bs" dev5_pixel_ZMM.root dev5_pixel_Bs.root [--json out.json]
"""

import argparse
import json

import awkward as ak
import numpy as np
import uproot

STAGES = (("l3_mu_no_ID", "before_id"), ("l3_mu_ID", "after_id"))


def dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def count(path):
    with uproot.open(path) as f:
        a = f["Events"].arrays([f"{c}_{v}" for c, _ in STAGES for v in ("matched", "tpPdgId", "tpPt", "tpEta", "tpPhi")]
                               + ["GenPart_pdgId", "GenPart_status", "GenPart_pt", "GenPart_eta", "GenPart_phi"])
    gen = (abs(a["GenPart_pdgId"]) == 13) & (a["GenPart_status"] == 1) & (abs(a["GenPart_eta"]) < 2.4) & (a["GenPart_pt"] > 2.0)
    g_eta, g_phi, g_pt = (a[f"GenPart_{v}"][gen][:, :, np.newaxis] for v in ("eta", "phi", "pt"))
    out = dict(events=len(a), generated=int(ak.sum(gen)))
    for table, stage in STAGES:
        muon = (a[f"{table}_matched"] == 1) & (abs(a[f"{table}_tpPdgId"]) == 13)
        t_eta, t_phi, t_pt = (a[f"{table}_{v}"][muon][:, np.newaxis, :] for v in ("tpEta", "tpPhi", "tpPt"))
        found = ((g_eta - t_eta) ** 2 + dphi(g_phi, t_phi) ** 2 < 0.02 ** 2) & (abs(t_pt / g_pt - 1.0) < 0.1)
        out[stage] = dict(fakes=int(ak.sum(~muon)), muons=int(ak.sum(muon)), found=int(ak.sum(ak.any(found, axis=2))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", nargs="+", action="append", required=True, metavar=("LABEL", "FILE"),
                    help="a label followed by the NANO files of the sample")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    print("| Sample | Events | Fake muons / event, before ID | after ID | Muons / event, before ID | after ID | "
          "Efficiency, before ID | after ID |")
    print("|---|---|---|---|---|---|---|---|")
    for label, *files in a.sample:
        if not files:
            ap.error(f"--sample {label}: no files")
        parts = [count(f) for f in files]
        n = sum(p["events"] for p in parts)
        gen = sum(p["generated"] for p in parts)
        r = dict(events=n, generated=gen)
        for _, stage in STAGES:
            c = {k: sum(p[stage][k] for p in parts) for k in ("fakes", "muons", "found")}
            r[stage] = dict(c, fakes_per_event=c["fakes"] / n, muons_per_event=c["muons"] / n, efficiency=c["found"] / gen)
        res[label] = r
        b, d = r["before_id"], r["after_id"]
        print(f"| {label} | {n} | {b['fakes_per_event']:.3f} | {d['fakes_per_event']:.3f} | {b['muons_per_event']:.3f} | "
              f"{d['muons_per_event']:.3f} | {b['efficiency']:.4f} | {d['efficiency']:.4f} |")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
