#!/usr/bin/env python3
"""
chamber_match.py - matching of tracker tracks to the muon chambers (segments
and rec-hits): the raw quantities of the candidates of every muon station,
inputs of the track forest.

A muon of charge q, transverse momentum pT and direction (eta, phi) at its
point of closest approach leaves, in each muon station it crosses, a segment
(or rec-hits) displaced from the track direction by the bending in the
magnetic field. For a (track, object) pair

    bend    = q * dphi(position)  * pT      dphi = phi(object position)   - phi(track)      [rad GeV]
    dirBend = q * dphi(direction) * pT      dphi = phi(segment direction) - phi(track)      (segments)
    dEta    = eta(object position) - eta(track)

bend and dirBend are negative for an object displaced on the side the charge
bends to; the factor pT removes most of the momentum dependence (dphi ~ 1/pT).
Where a muon's own objects lie in these quantities depends on the detector,
the station and |eta|. Nothing of this is tabulated or fitted here: the forest
gets the raw quantities of the candidates and learns it.

Stations:
    segments   ME0        ME0 (GEM segments of station 0)
               St1-St4    DT or CSC stations 1-4
    rec-hits   GE11, GE21 GE1/1 and GE2/1 (GEM rec-hits; the HLT has no segments there)
               RPC1-RPC4  RPC stations 1-4, barrel or endcap

Candidates of a track in a station: the objects inside the search window
(|dEta| < MAX_DETA, bend on the charge side and not beyond MAX_BEND, rec-hits
within |bx| <= MAX_HIT_BX) whose direction, for segments, is also bent on the
charge side. They are ordered by |dEta| (the coordinate the field does not
bend); the first two are kept. The search window is the only set of numbers
of the matching: a wide road, not a working point (a muon's own objects have
|bend| < 4 rad GeV in every station but the innermost endcap ones, where the
tail reaches 6.5 at the lowest momenta).

Features, per station:
    cm_bend_<st>, cm_dEta_<st>            nearest candidate
    cm_bendDiff_<st>                      its dirBend - bend (segment stations): how much more the direction is bent
                                          than the position
    cm_bend2_<st>, cm_dEta2_<st>, cm_bendDiff2_<st>     second candidate
A station without a (second) candidate has the NO_* values, outside the range of any candidate.

Everything is a function of quantities the CMSSW selector has (track parameters and the global position and
direction of segments and hits); the C++ counterpart is RecoMuon/L3TrackFinder/interface/MuonChamberMatch.h.

The RPC rec-hits are read with the global eta and phi of the rpcRecHit table (rpcRecHit_g_eta, rpcRecHit_g_phi).
"""

import awkward as ak
import numpy as np

DETS = ("dt", "csc", "me0", "gem", "rpcb", "rpce")           # order of the objects of a station
SEGMENT_DETS = ("dt", "csc", "me0")                          # the others give rec-hits (no direction)
SEGMENT_STATIONS = ("ME0", "St1", "St2", "St3", "St4")
HIT_STATIONS = ("GE11", "GE21", "RPC1", "RPC2", "RPC3", "RPC4")
STATIONS = SEGMENT_STATIONS + HIT_STATIONS
# Search window
MAX_DETA = 0.3
MAX_BEND = 8.0          # |q * dphi * pT| [rad GeV]
MAX_HIT_BX = 1
# GE1/1 and GE2/1 by the hit |z| [cm] (the gemRecHit table has no station: the selector uses the one of the
# detector id); ME0 hits (|z| < 560) are not used: the ME0 segments are.
GEM_STATION_Z = {1: (560.0, 600.0), 2: (780.0, 820.0)}
# Values of a station without a candidate: outside the range of any candidate (bend < 0, bendDiff < MAX_BEND,
# |dEta| < MAX_DETA), so their exact value is irrelevant to a forest
NO_BEND, NO_DETA = 9.0, 0.5

FEATURES = ([f"cm_bend_{s}" for s in STATIONS] + [f"cm_dEta_{s}" for s in STATIONS] + [f"cm_bendDiff_{s}" for s in SEGMENT_STATIONS]
            + [f"cm_bend2_{s}" for s in STATIONS] + [f"cm_dEta2_{s}" for s in STATIONS] + [f"cm_bendDiff2_{s}" for s in SEGMENT_STATIONS])

SEGMENT_BRANCHES = [
    "dtSegment_seg4D_posGlb_eta", "dtSegment_seg4D_posGlb_phi", "dtSegment_seg4D_dirGlb_phi", "dtSegment_station",
    "dtSegment_seg4D_hasPhi", "dtSegment_seg4D_hasZed", "dtSegment_seg2D_phi_nHits",
    "cscSegment_posGlb_eta", "cscSegment_posGlb_phi", "cscSegment_dirGlb_phi", "cscSegment_station", "cscSegment_nHits",
    "gemSegment_posGlb_eta", "gemSegment_posGlb_phi", "gemSegment_dirGlb_phi", "gemSegment_station",
]
GEM_HIT_BRANCHES = ["gemRecHit_g_phi", "gemRecHit_g_r", "gemRecHit_g_z", "gemRecHit_layer", "gemRecHit_bx"]
RPC_HIT_BRANCHES = ["rpcRecHit_g_eta", "rpcRecHit_g_phi", "rpcRecHit_region", "rpcRecHit_station", "rpcRecHit_layer", "rpcRecHit_bx"]
OBJECT_TABLES = ("dtSegment_", "cscSegment_", "gemSegment_", "gemRecHit_", "rpcRecHit_")


def branches():
    """Input branches of the chamber matching."""
    return SEGMENT_BRANCHES + GEM_HIT_BRANCHES + RPC_HIT_BRANCHES


def _dphi(a, b):
    d = a - b
    return np.where(np.abs(d) <= np.pi, d, d - np.rint(d * (1.0 / (2.0 * np.pi))) * (2.0 * np.pi))


def station_index(det, station):
    """Index in STATIONS of a detector's station."""
    if det in ("dt", "csc"):
        return station
    if det == "me0":
        return 0
    return (4 if det == "gem" else 6) + station


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
def _objects(arr, det):
    """Per-event jagged (eta, phi, dirPhi, station) of one detector. dirPhi is 0 for rec-hits."""
    if det == "dt":
        ok = arr["dtSegment_seg4D_hasPhi"] > 0
        g = lambda v: arr[f"dtSegment_{v}"][ok]
        return g("seg4D_posGlb_eta"), g("seg4D_posGlb_phi"), g("seg4D_dirGlb_phi"), g("station")
    if det == "csc":
        g = lambda v: arr[f"cscSegment_{v}"]
        return g("posGlb_eta"), g("posGlb_phi"), g("dirGlb_phi"), g("station")
    if det == "me0":
        ok = arr["gemSegment_station"] == 0
        g = lambda v: arr[f"gemSegment_{v}"][ok]
        return g("posGlb_eta"), g("posGlb_phi"), g("dirGlb_phi"), g("station")
    if det == "gem":
        z, r = arr["gemRecHit_g_z"], arr["gemRecHit_g_r"]
        st = ak.zeros_like(z)
        for s, (lo, hi) in GEM_STATION_Z.items():
            st = ak.where((abs(z) > lo) & (abs(z) < hi), float(s), st)
        ok = (st > 0) & (abs(arr["gemRecHit_bx"]) <= MAX_HIT_BX)
        # eta from the stored z and r, rounded to float (the selector builds its hit list in float)
        eta = ak.values_astype(ak.values_astype(np.arcsinh(z / r), np.float32), np.float64)
        return eta[ok], arr["gemRecHit_g_phi"][ok], ak.zeros_like(z)[ok], st[ok]
    # RPC, barrel ("rpcb") or endcap ("rpce")
    ok = (abs(arr["rpcRecHit_bx"]) <= MAX_HIT_BX) & ((arr["rpcRecHit_region"] == 0) == (det == "rpcb"))
    eta, phi = arr["rpcRecHit_g_eta"], arr["rpcRecHit_g_phi"]
    st = arr["rpcRecHit_station"]
    return eta[ok], phi[ok], ak.zeros_like(st)[ok], st[ok]


def candidates(arr, prefix):
    """Every candidate of every track, in the order: detector (DETS), event, track, object.
    Returns float64 arrays (bend, dirBend, dEta) and int64 (index of the track in the flattened track list,
    station index). dirBend is 0 for rec-hits. arr in the float64 convention (pixel_features.as_float64)."""
    n_trk = ak.num(arr[f"{prefix}_pt"])
    t = ak.zip({"pt": arr[f"{prefix}_pt"], "eta": arr[f"{prefix}_eta"], "phi": arr[f"{prefix}_phi"],
                "q": np.sign(arr[f"{prefix}_qoverp"]), "i": ak.unflatten(np.arange(int(ak.sum(n_trk))), n_trk)})
    f = lambda x: ak.to_numpy(x).astype(np.float64)
    out = [[] for _ in range(5)]
    for det in DETS:
        se, sp, sd, st = _objects(arr, det)
        P = ak.cartesian({"t": t, "s": ak.zip({"eta": se, "phi": sp, "dir": sd, "st": st})}, axis=1)
        P = ak.flatten(P[np.abs(P.s.eta - P.t.eta) < MAX_DETA])
        pt, q, tphi = f(P.t.pt), f(P.t.q), f(P.t.phi)
        bend = q * _dphi(f(P.s.phi), tphi) * pt
        ok = (bend < 0.0) & (bend > -MAX_BEND)
        if det in SEGMENT_DETS:
            dir_bend = q * _dphi(f(P.s.dir), tphi) * pt
            ok &= dir_bend < 0.0
        else:
            dir_bend = np.zeros(len(pt))
        station = np.array([station_index(det, s) for s in range(5)])[f(P.s.st)[ok].astype(np.int64)]
        for o, v in zip(out, (bend[ok], dir_bend[ok], (f(P.s.eta) - f(P.t.eta))[ok], ak.to_numpy(P.t.i)[ok].astype(np.int64), station)):
            o.append(v)
    return tuple(np.concatenate(o) for o in out)


# --------------------------------------------------------------------------- #
# Track features
# --------------------------------------------------------------------------- #
def track_features(arr, prefix):
    """Dict {feature name: float64 array over all tracks of `prefix` (flattened)}: FEATURES."""
    n = int(ak.sum(ak.num(arr[f"{prefix}_pt"])))
    bend, dir_bend, d_eta, trk, station = candidates(arr, prefix)
    # rank of a candidate among those of its track and station, by |dEta|; equal ones keep the order of candidates()
    group = trk * len(STATIONS) + station
    order = np.lexsort((np.abs(d_eta), group))
    g = group[order]
    start = np.ones(len(order), dtype=bool)
    start[1:] = g[1:] != g[:-1]
    idx = np.arange(len(order))
    rank = idx - np.maximum.accumulate(np.where(start, idx, 0))
    out = {}
    for k, tag in ((0, ""), (1, "2")):
        sel = order[rank == k]
        B = np.full((n, len(STATIONS)), NO_BEND); E = np.full((n, len(STATIONS)), NO_DETA); D = np.full((n, len(STATIONS)), NO_BEND)
        B[trk[sel], station[sel]] = bend[sel]
        E[trk[sel], station[sel]] = d_eta[sel]
        D[trk[sel], station[sel]] = dir_bend[sel] - bend[sel]
        for i, s in enumerate(STATIONS):
            out[f"cm_bend{tag}_{s}"] = B[:, i]
            out[f"cm_dEta{tag}_{s}"] = E[:, i]
            if s in SEGMENT_STATIONS:
                out[f"cm_bendDiff{tag}_{s}"] = D[:, i]
    return out
