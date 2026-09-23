"""
OI_features.py - Feature extraction for the OI (outside-in) muon track
high-purity selectors (l3_tk_OI_* track collection, L2 standalone-muon
matching), used by OI_pixel_xgb.py and OI_general_xgb.py.

IMPORTANT: the extraction mirrors the CMSSW extractor
(RecoMuon/L3TrackFinder/interface/OITrackSelectorFeatures.h,
muonhp::extractOITrackFeatures) EXACTLY (same epsilons/imputation; the C++
emits the 22-feature production subset OI_PRODUCTION_FEATURES of the 26
built here, in that order), including the details where the legacy OI DNN
training scripts (tests/OI_*_model.py) differed from the deployed C++:

  * delta phi between the track and a standalone muon is WRAPPED
    (reco::deltaPhi equivalent), while the legacy python used the raw
    difference;
  * the matching score is (chi2Eta + chi2Phi + chi2Pt + chi2Dz) / 9 (each
    component divided by kOIMatchChi2=9), while the legacy python used the
    unnormalised sum;
  * the no-match imputation for l2_mu_vtx_matchingScore is 10.0 in raw space
    (kOIImputeMatchScore), while the legacy python imputed log10(10) ~ 1.0.

Scoring semantics: bestScore is the minimum score over standalone muons
(initial 25 i.e. 5 sigma^2 in the /9 units); hasMatch = bestScore < 25,
matchingScore = log10(bestScore + 1e-6) when matched else 10.0.

Exports build_dataset(), the OI branch lists and the production feature ABI;
evaluation helpers come from pixel_features (shared with the IO models).
"""
import os
import sys

import awkward as ak
import numpy as np

# pixel_features (shared evaluation helpers) lives in the production/ root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pixel_features as pf  # noqa: F401  (re-exported helpers by convention)


# --------------------------------------------------------------------------- #
# Branch definitions
# --------------------------------------------------------------------------- #
MAIN_BRANCH = "Events"

TK_BRANCHES = [
    "l3_tk_OI_p",
    "l3_tk_OI_pt",
    "l3_tk_OI_ptErr",
    "l3_tk_OI_eta",
    "l3_tk_OI_etaErr",
    "l3_tk_OI_phi",
    "l3_tk_OI_phiErr",
    "l3_tk_OI_chi2",
    "l3_tk_OI_normalizedChi2",
    "l3_tk_OI_nPixelHits",
    "l3_tk_OI_nTrkLays",
    "l3_tk_OI_nFoundHits",
    "l3_tk_OI_nLostHits",
    "l3_tk_OI_dsz",
    "l3_tk_OI_dszErr",
    "l3_tk_OI_dxy",
    "l3_tk_OI_dxyErr",
    "l3_tk_OI_dz",
    "l3_tk_OI_dzErr",
    "l3_tk_OI_qoverp",
    "l3_tk_OI_qoverpErr",
    "l3_tk_OI_lambdaErr",
    "l3_tk_OI_matched",
    "l3_tk_OI_duplicate",
    "l3_tk_OI_tpPdgId",
    "l3_tk_OI_tpPt",
    "l3_tk_OI_tpEta",
    "l3_tk_OI_tpPhi",
]

L2_MU_VTX_BRANCHES = [
    "l2_mu_vtx_pt",
    "l2_mu_vtx_ptErr",
    "l2_mu_vtx_eta",
    "l2_mu_vtx_etaErr",
    "l2_mu_vtx_phi",
    "l2_mu_vtx_phiErr",
    "l2_mu_vtx_dz",
    "l2_mu_vtx_dzErr",
]

# Features stored as log10(|x| + eps)
LOG_FEATURES = [
    "l3_tk_OI_p",
    "l3_tk_OI_pt",
    "l3_tk_OI_ptErr",
    "l3_tk_OI_chi2",
    "l3_tk_OI_normalizedChi2",
    "l3_tk_OI_etaErr",
    "l3_tk_OI_phiErr",
    "l3_tk_OI_dszErr",
    "l3_tk_OI_dxyErr",
    "l3_tk_OI_dzErr",
    "l3_tk_OI_qoverpErr",
    "l3_tk_OI_lambdaErr",
]

PLAIN_FEATURES = [
    "l3_tk_OI_eta",
    "l3_tk_OI_nPixelHits",
    "l3_tk_OI_nTrkLays",
    "l3_tk_OI_nFoundHits",
    "l3_tk_OI_nLostHits",
]

LABEL_FIELD = "l3_tk_OI_matched"

# Production feature ABI: build_dataset() emits 26 features, the deployed
# forests use 22. Round-2 pruning (26 -> 22) dropped these four (duplicates of
# qoverpErr's pT-uncertainty information and of normalizedChi2/chi2PerHit):
OI_DROP_FEATURES = [
    "l3_tk_OI_chi2",
    "l3_tk_OI_ptErr",
    "l3_tk_OI_sigmaPtOverPt",
    "l3_tk_OI_relUncertaintyProduct",
]
# The 22 kept features in training order = muonhp::OITrackFeatures::toArray()
# (asserted by the forest pipeline).
OI_PRODUCTION_FEATURES = [
    "l3_tk_OI_p", "l3_tk_OI_pt", "l3_tk_OI_normalizedChi2", "l3_tk_OI_etaErr",
    "l3_tk_OI_phiErr", "l3_tk_OI_dszErr", "l3_tk_OI_dxyErr", "l3_tk_OI_dzErr",
    "l3_tk_OI_qoverpErr", "l3_tk_OI_lambdaErr",
    "l3_tk_OI_eta", "l3_tk_OI_nPixelHits", "l3_tk_OI_nTrkLays", "l3_tk_OI_nFoundHits",
    "l3_tk_OI_nLostHits",
    "l3_tk_OI_impact3D", "l3_tk_OI_impactSignificance", "l3_tk_OI_chi2PerHit",
    "l3_tk_OI_hitEfficiency",
    "l2_mu_vtx_hasMatch", "l2_mu_vtx_matchingScore",
    "is_low_pt",
]

# Matching constants (must equal OITrackSelectorFeatures.h)
MATCH_CHI2 = 9.0
NO_MATCH_BEST_SCORE = 25.0
IMPUTE_MATCH_SCORE = 10.0  # raw space (C++ kOIImputeMatchScore)


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def build_dataset(arr, file_labels_in, useStandaloneFeatures=True,
                  verbose=False, event_ids=None):
    """
    Builds the 26-feature OI matrix from an awkward array of events, matching
    the C++ extractor feature-for-feature.

    Returns (X, y, file_labels_masked, final_feature_names); if event_ids is
    given, the per-track event ids survive the same masks as a 5th element.
    """
    if verbose:
        print("Building dataset...")

    arr = pf.as_float64(arr)  # numeric convention shared with the C++: see pf.as_float64()
    mask = arr["l3_tk_OI_pt"] > 0

    n_tracks_per_event = ak.num(arr["l3_tk_OI_pt"])
    file_labels_jagged = ak.unflatten(
        np.repeat(file_labels_in, n_tracks_per_event), n_tracks_per_event
    )
    file_labels_masked = ak.to_numpy(ak.flatten(file_labels_jagged[mask]))
    event_ids_masked = None
    if event_ids is not None:
        event_ids_jagged = ak.unflatten(
            np.repeat(event_ids, n_tracks_per_event), n_tracks_per_event
        )
        event_ids_masked = ak.to_numpy(ak.flatten(event_ids_jagged[mask]))

    cols = []
    final_feature_names = []

    trk_pt = arr["l3_tk_OI_pt"]
    available_keys = arr.fields

    # 0-11: log features
    for f in LOG_FEATURES:
        if f in available_keys:
            flat = ak.to_numpy(ak.flatten(arr[f][mask]))
            cols.append(np.log10(np.abs(flat) + 1e-6))
            final_feature_names.append(f)

    # 12-16: plain features (raw eta, hit counts)
    for f in PLAIN_FEATURES:
        if f in available_keys:
            cols.append(ak.to_numpy(ak.flatten(arr[f][mask])))
            final_feature_names.append(f)

    # 17-22: derived features
    if verbose:
        print("Adding derived features...")

    trk_dxy = arr["l3_tk_OI_dxy"]
    trk_dz = arr["l3_tk_OI_dz"]
    trk_dxyErr = arr["l3_tk_OI_dxyErr"]
    trk_dzErr = arr["l3_tk_OI_dzErr"]

    # 17: impact3D (log of the SQUARED 3D IP, dxy^2+dz^2 - matches C++)
    ip3d = trk_dxy**2 + trk_dz**2
    cols.append(ak.to_numpy(ak.flatten(np.log10(ip3d + 1e-6)[mask])).astype(np.float32))
    final_feature_names.append("l3_tk_OI_impact3D")

    # 18: impact significance (log)
    sip_combined = np.sqrt(
        (trk_dxy / np.maximum(trk_dxyErr, 1e-6)) ** 2
        + (trk_dz / np.maximum(trk_dzErr, 1e-6)) ** 2
    )
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sip_combined + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("l3_tk_OI_impactSignificance")

    trk_chi2 = arr["l3_tk_OI_chi2"]
    trk_nFound = arr["l3_tk_OI_nFoundHits"]
    trk_nLost = arr["l3_tk_OI_nLostHits"]

    # 19: chi2 per hit (log)
    chi2_hit = trk_chi2 / np.maximum(trk_nFound, 1)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(chi2_hit + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("l3_tk_OI_chi2PerHit")

    # 20: hit efficiency (linear)
    hit_eff = trk_nFound / np.maximum(trk_nFound + trk_nLost, 1)
    cols.append(ak.to_numpy(ak.flatten(hit_eff[mask])).astype(np.float32))
    final_feature_names.append("l3_tk_OI_hitEfficiency")

    trk_ptErr = arr["l3_tk_OI_ptErr"]
    trk_qoverp = arr["l3_tk_OI_qoverp"]
    trk_qoverpErr = arr["l3_tk_OI_qoverpErr"]

    # 21: sigmaPt/pt (log)
    sigmaPtOverPt = trk_ptErr / np.maximum(trk_pt, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sigmaPtOverPt + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("l3_tk_OI_sigmaPtOverPt")

    # 22: relative uncertainty product (log)
    relUncertProd = sigmaPtOverPt * (
        trk_qoverpErr / np.maximum(np.abs(trk_qoverp), 1e-6)
    )
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(relUncertProd + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("l3_tk_OI_relUncertaintyProduct")

    # 23-24: standalone (L2 muon vertex) matching - C++ semantics
    if useStandaloneFeatures:
        if verbose:
            print("Computing standalone muon matching (C++ semantics)...")

        t_eta = arr["l3_tk_OI_eta"][:, :, np.newaxis]
        t_etaErr = arr["l3_tk_OI_etaErr"][:, :, np.newaxis]
        t_phi = arr["l3_tk_OI_phi"][:, :, np.newaxis]
        t_phiErr = arr["l3_tk_OI_phiErr"][:, :, np.newaxis]
        t_pt = arr["l3_tk_OI_pt"][:, :, np.newaxis]
        t_ptErr = arr["l3_tk_OI_ptErr"][:, :, np.newaxis]
        t_dz = arr["l3_tk_OI_dz"][:, :, np.newaxis]
        t_dzErr = arr["l3_tk_OI_dzErr"][:, :, np.newaxis]

        s_eta = arr["l2_mu_vtx_eta"][:, np.newaxis, :]
        s_etaErr = arr["l2_mu_vtx_etaErr"][:, np.newaxis, :]
        s_phi = arr["l2_mu_vtx_phi"][:, np.newaxis, :]
        s_phiErr = arr["l2_mu_vtx_phiErr"][:, np.newaxis, :]
        s_pt = arr["l2_mu_vtx_pt"][:, np.newaxis, :]
        s_ptErr = arr["l2_mu_vtx_ptErr"][:, np.newaxis, :]
        s_dz = arr["l2_mu_vtx_dz"][:, np.newaxis, :]
        s_dzErr = arr["l2_mu_vtx_dzErr"][:, np.newaxis, :]

        # Wrapped delta phi (reco::deltaPhi)
        d_phi = pf.delta_phi(t_phi, s_phi)

        chi2_eta = (t_eta - s_eta) ** 2 / (t_etaErr**2 + s_etaErr**2 + 1e-12)
        chi2_phi = d_phi**2 / (t_phiErr**2 + s_phiErr**2 + 1e-12)
        chi2_pt = (t_pt - s_pt) ** 2 / (t_ptErr**2 + s_ptErr**2 + 1e-12)
        chi2_dz = (t_dz - s_dz) ** 2 / (t_dzErr**2 + s_dzErr**2 + 1e-12)

        score = (chi2_eta + chi2_phi + chi2_pt + chi2_dz) / MATCH_CHI2
        best_score = ak.min(score, axis=2)
        has_match = ak.fill_none(best_score < NO_MATCH_BEST_SCORE, False)

        # 23: hasMatch
        cols.append(ak.to_numpy(ak.flatten(has_match[mask])).astype(np.float32))
        final_feature_names.append("l2_mu_vtx_hasMatch")

        # 24: log10(bestScore + eps) when matched, 10.0 (raw) when not
        ms = ak.where(has_match, np.log10(best_score + 1e-6), IMPUTE_MATCH_SCORE)
        cols.append(ak.to_numpy(ak.flatten(ms[mask])).astype(np.float32))
        final_feature_names.append("l2_mu_vtx_matchingScore")

    # 25: soft low-pT indicator (sigmoid around 5 GeV)
    flat_pt = ak.to_numpy(ak.flatten(trk_pt[mask]))
    exponent = (flat_pt - pf.LOW_PT_CUT) * 2.0
    exponent = np.clip(exponent, -20.0, 20.0)
    low_pt_indicator = 1.0 / (1.0 + np.exp(exponent))
    cols.append(low_pt_indicator)
    final_feature_names.append("is_low_pt")

    X = np.column_stack(cols).astype(np.float32)  # single float64 -> float32 rounding
    y = ak.to_numpy(ak.flatten(arr[LABEL_FIELD][mask])).astype(np.int8)

    finite_mask = np.isfinite(X).all(axis=1)
    if not finite_mask.all():
        n_bad = (~finite_mask).sum()
        print(f"  Warning: Removing {n_bad} non-finite rows")
        X = X[finite_mask]
        y = y[finite_mask]
        file_labels_masked = file_labels_masked[finite_mask]
        if event_ids_masked is not None:
            event_ids_masked = event_ids_masked[finite_mask]

    if event_ids is not None:
        return X, y, file_labels_masked, event_ids_masked, final_feature_names
    return X, y, file_labels_masked, final_feature_names
