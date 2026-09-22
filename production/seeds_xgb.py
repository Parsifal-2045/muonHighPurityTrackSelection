"""
seeds_xgb.py - XGBoost model for high-purity muon IO-track (seeds) selection.

Structural clone of pixel_xgb.py but using muon_general_tracks_* branches
(seeds_model.py data) instead of muon_pixel_tracks_* (pixel_model.py data).
The feature extraction, 33-feature pruning, sample weights, train/val/test
split, evaluation suite, and both export formats (ONNX + compact .bin) are
identical to pixel_xgb.py -- the C++ module MuonIOTracksForestSelector handles
both selectors with the same 33-feature extraction, differing only in the
.bin model file and decision threshold.

Run:
    python seeds_xgb.py
"""

import gc
import os
import time

import awkward as ak
import numpy as np
import uproot
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import pixel_features as pf

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CFG = dict(
    data_dir="/cms-hlt-nfs/user/lferragi/seedsSelector/",
    output_dir="seeds_xgb_output_33f/",
    # Feature flags (match DNN)
    useL1TkMuFeatures=True,
    useL1TkMuStubFeatures=True,
    # Feature pruning: drop these 11 features (by name).
    # Identical pruning to pixel_xgb.py -- same 11 features dropped from the
    # original 44, producing the same 33-feature ABI the C++ module expects.
    # The branch prefix differs (muon_general_tracks_* vs muon_pixel_tracks_*)
    # but the feature semantics and ordering are identical.
    drop_features=[
        "muon_general_tracks_nLostHits",          # zero gain
        "muon_general_tracks_hitEfficiency",      # zero gain
        "muon_general_tracks_normalizedChi2",      # gain 64.6
        "muon_general_tracks_chi2PerHit",           # gain 90.3
        "muon_general_tracks_chi2",                 # gain 105.8
        "muon_general_tracks_impactSignificance",  # gain 113.7
        "muon_general_tracks_dxyErr",               # gain 115.5
        "muon_general_tracks_dszErr",              # gain 128.8
        "muon_general_tracks_eta",                   # redundant with absEta
        "muon_general_tracks_ptErr",               # redundant with qoverpErr
        "muon_general_tracks_relUncertaintyProduct",  # redundant with qoverpErr
    ],
    # Sample weights (match DNN exactly)
    signal_boost=pf.SIGNAL_BOOST,
    kin_weight_max=pf.KIN_WEIGHT_MAX,
    # XGBoost hyperparameters (validated via pixel_xgb_optimize.py)
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    # Early stopping
    early_stopping_rounds=100,
    # ONNX
    onnx_opset=pf.ONNX_OPSET,
    # Compact binary export (forest module format).
    # fp32: the accumulated rounding across 2000 trees with fp16 produces max
    # score error ~0.18 -- too lossy for a fixed-threshold selector.
    compact_bin_fp16=False,
)

# --------------------------------------------------------------------------- #
# Input files and branch definitions (muon_general_tracks_* = seeds model)
# --------------------------------------------------------------------------- #
files = sorted(
    [
        os.path.join(CFG["data_dir"], f)
        for f in os.listdir(CFG["data_dir"])
        if os.path.isfile(os.path.join(CFG["data_dir"], f))
    ]
)
print(f"Selected {len(files)} input files:")
print(files)

main_branch = "Events"
tk_branches = [
    "muon_general_tracks_p",
    "muon_general_tracks_pt",
    "muon_general_tracks_ptErr",
    "muon_general_tracks_eta",
    "muon_general_tracks_etaErr",
    "muon_general_tracks_phi",
    "muon_general_tracks_phiErr",
    "muon_general_tracks_chi2",
    "muon_general_tracks_normalizedChi2",
    "muon_general_tracks_nPixelHits",
    "muon_general_tracks_nTrkLays",
    "muon_general_tracks_nFoundHits",
    "muon_general_tracks_nLostHits",
    "muon_general_tracks_dsz",
    "muon_general_tracks_dszErr",
    "muon_general_tracks_dxy",
    "muon_general_tracks_dxyErr",
    "muon_general_tracks_dz",
    "muon_general_tracks_dzErr",
    "muon_general_tracks_qoverp",
    "muon_general_tracks_qoverpErr",
    "muon_general_tracks_lambdaErr",
    "muon_general_tracks_matched",
    "muon_general_tracks_duplicate",
    "muon_general_tracks_tpPdgId",
    "muon_general_tracks_tpPt",
    "muon_general_tracks_tpEta",
    "muon_general_tracks_tpPhi",
]
l1tkMuon_branches = ["L1TkMu_pt", "L1TkMu_eta", "L1TkMu_phi"]
stub_branches = [
    "L1TkMuStub_type",
    "L1TkMuStub_quality",
    "L1TkMuStub_parentL1TkMu",
    "L1TkMuStub_etaRegion",
    "L1TkMuStub_phiRegion",
    "L1TkMuStub_depthRegion",
]

log_features = [
    "muon_general_tracks_p",
    "muon_general_tracks_pt",
    "muon_general_tracks_ptErr",
    "muon_general_tracks_chi2",
    "muon_general_tracks_normalizedChi2",
    "muon_general_tracks_etaErr",
    "muon_general_tracks_phiErr",
    "muon_general_tracks_dszErr",
    "muon_general_tracks_dxyErr",
    "muon_general_tracks_dzErr",
    "muon_general_tracks_qoverpErr",
    "muon_general_tracks_lambdaErr",
]
plain_features = [
    "muon_general_tracks_eta",
    "muon_general_tracks_nPixelHits",
    "muon_general_tracks_nTrkLays",
    "muon_general_tracks_nFoundHits",
    "muon_general_tracks_nLostHits",
]
LABEL_FIELD = "muon_general_tracks_matched"


# --------------------------------------------------------------------------- #
# Helpers (identical to pixel_features.py, kept inline for data-source isolation)
# --------------------------------------------------------------------------- #
def delta_phi(phi1, phi2):
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


def impute_and_log(vals, mask, fill=-1.0):
    v = np.asarray(ak.to_numpy(ak.flatten(ak.fill_none(vals, fill))), dtype=np.float64)
    m = ak.to_numpy(ak.flatten(mask))
    v[~m] = fill
    return np.log10(np.abs(v) + 1e-6).astype(np.float32)


def impute_linear(vals, mask, fill=0.0):
    v = np.asarray(ak.to_numpy(ak.flatten(ak.fill_none(vals, fill))), dtype=np.float64)
    m = ak.to_numpy(ak.flatten(mask))
    v[~m] = fill
    return v.astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def build_dataset(arr, file_labels_in, useL1TkMuFeatures=True, useL1TkMuStubFeatures=True,
                  verbose=False):
    """Build the 44-feature matrix from muon_general_tracks events.
    Identical feature extraction to seeds_model.py build_dataset()."""
    print("Building dataset...")

    mask = arr["muon_general_tracks_pt"] > 0

    n_tracks_per_event = ak.num(arr["muon_general_tracks_pt"])
    file_labels_jagged = ak.unflatten(
        np.repeat(file_labels_in, n_tracks_per_event), n_tracks_per_event
    )
    file_labels_masked = ak.to_numpy(ak.flatten(file_labels_jagged[mask]))

    cols = []
    final_feature_names = []

    trk_pt = arr["muon_general_tracks_pt"]
    available_keys = arr.fields

    # Standard features (log and linear)
    for f in log_features:
        if f in available_keys:
            flat = ak.to_numpy(ak.flatten(arr[f][mask])).astype(np.float32)
            cols.append(np.log10(np.abs(flat) + 1e-6))
            final_feature_names.append(f)

    for f in plain_features:
        if f in available_keys:
            flat = ak.to_numpy(ak.flatten(arr[f][mask])).astype(np.float32)
            cols.append(flat)
            final_feature_names.append(f)

    # Derived features
    print("Adding derived features...")

    trk_dxy = arr["muon_general_tracks_dxy"]
    trk_dz = arr["muon_general_tracks_dz"]
    trk_dxyErr = arr["muon_general_tracks_dxyErr"]
    trk_dzErr = arr["muon_general_tracks_dzErr"]

    # Impact Parameter 3D (log)
    ip3d = trk_dxy**2 + trk_dz**2
    cols.append(ak.to_numpy(ak.flatten(np.log10(ip3d + 1e-6)[mask])).astype(np.float32))
    final_feature_names.append("muon_general_tracks_impact3D")

    # Combined Impact Significance (log)
    sip_combined = np.sqrt(
        (trk_dxy / np.maximum(trk_dxyErr, 1e-6)) ** 2
        + (trk_dz / np.maximum(trk_dzErr, 1e-6)) ** 2
    )
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sip_combined + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_general_tracks_impactSignificance")

    # Track Quality
    trk_chi2 = arr["muon_general_tracks_chi2"]
    trk_nFound = arr["muon_general_tracks_nFoundHits"]
    trk_nLost = arr["muon_general_tracks_nLostHits"]

    # Chi2 per hit (log)
    chi2_hit = trk_chi2 / np.maximum(trk_nFound, 1)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(chi2_hit + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_general_tracks_chi2PerHit")

    # Hit Efficiency
    hit_eff = trk_nFound / np.maximum(trk_nFound + trk_nLost, 1)
    cols.append(ak.to_numpy(ak.flatten(hit_eff[mask])).astype(np.float32))
    final_feature_names.append("muon_general_tracks_hitEfficiency")

    # Relative Uncertainties
    trk_ptErr = arr["muon_general_tracks_ptErr"]
    trk_p = arr["muon_general_tracks_p"]
    trk_qoverp = arr["muon_general_tracks_qoverp"]
    trk_qoverpErr = arr["muon_general_tracks_qoverpErr"]

    # SigmaPt / Pt (log)
    sigmaPtOverPt = trk_ptErr / np.maximum(trk_pt, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sigmaPtOverPt + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_general_tracks_sigmaPtOverPt")

    # Relative Uncertainty Product (log)
    relUncertProd = sigmaPtOverPt * (
        trk_qoverpErr / np.maximum(np.abs(trk_qoverp), 1e-6)
    )
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(relUncertProd + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_general_tracks_relUncertaintyProduct")

    # Separated 2D impact parameter significance (log)
    sip_2d = np.abs(trk_dxy) / np.maximum(trk_dxyErr, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sip_2d + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_general_tracks_sip2D")

    # Longitudinal impact parameter significance (log)
    sip_z = np.abs(trk_dz) / np.maximum(trk_dzErr, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sip_z + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_general_tracks_sipZ")

    # |dxy| / pT
    dxy_over_pt = np.abs(trk_dxy) / np.maximum(trk_pt, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(dxy_over_pt + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_general_tracks_dxyOverPt")

    # ptErr / p
    ptErr_over_p = trk_ptErr / np.maximum(trk_p, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(ptErr_over_p + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_general_tracks_ptErrOverP")

    # |dz| / |dxy| ratio
    dz_over_dxy = np.abs(trk_dz) / (np.abs(trk_dxy) + 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(dz_over_dxy + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_general_tracks_dzOverDxy")

    # |eta|
    trk_eta = arr["muon_general_tracks_eta"]
    cols.append(ak.to_numpy(ak.flatten(np.abs(trk_eta)[mask])).astype(np.float32))
    final_feature_names.append("muon_general_tracks_absEta")

    # L1 Matching
    if useL1TkMuFeatures:
        print("Computing L1 matching...")

        t_eta = arr["muon_general_tracks_eta"][:, :, np.newaxis]
        t_phi = arr["muon_general_tracks_phi"][:, :, np.newaxis]
        t_pt = arr["muon_general_tracks_pt"][:, :, np.newaxis]
        t_ptErr = arr["muon_general_tracks_ptErr"][:, :, np.newaxis]

        l1_eta = arr["L1TkMu_eta"][:, np.newaxis, :]
        l1_phi = arr["L1TkMu_phi"][:, np.newaxis, :]
        l1_pt = arr["L1TkMu_pt"][:, np.newaxis, :]

        dEta = t_eta - l1_eta
        dPhi = delta_phi(t_phi, l1_phi)
        dR2_matrix = dEta**2 + dPhi**2

        ratio_matrix = (t_pt - l1_pt) ** 2 / (t_ptErr**2 + 1e-12)

        match_chi2Pt_cut = 9.0
        is_compatible = ratio_matrix < match_chi2Pt_cut

        dR2_matrix_masked = ak.mask(dR2_matrix, is_compatible)
        min_dR2 = ak.min(dR2_matrix_masked, axis=2)

        min_vals_broad = ak.fill_none(min_dR2[:, :, np.newaxis], -1.0)
        is_best_match = (dR2_matrix == min_vals_broad) & is_compatible

        dPt_matrix = np.abs(t_pt - l1_pt) / (l1_pt + 1e-9)

        best_dPt_jagged = dPt_matrix[is_best_match]
        best_ratio_jagged = ratio_matrix[is_best_match]

        matched_dPt = ak.firsts(best_dPt_jagged, axis=2)
        matched_ratio = ak.firsts(best_ratio_jagged, axis=2)

        matched_score = min_dR2 * (1.0 + matched_dPt)

        match_dR_cut = 0.3**2
        has_match = (min_dR2 < match_dR_cut) & (~ak.is_none(min_dR2))

        # Stubs features
        if "L1TkMuStub_parentL1TkMu" in arr.fields and useL1TkMuStubFeatures:
            s_parent = arr["L1TkMuStub_parentL1TkMu"]
            s_type = arr["L1TkMuStub_type"]
            s_qual = arr["L1TkMuStub_quality"]
            s_etaRegion = arr["L1TkMuStub_etaRegion"]
            s_phiRegion = arr["L1TkMuStub_phiRegion"]
            s_depthRegion = arr["L1TkMuStub_depthRegion"]

            l1_indices = ak.local_index(arr["L1TkMu_pt"], axis=1)
            l1_idx_b = l1_indices[:, :, np.newaxis]
            s_parent_b = s_parent[:, np.newaxis, :]
            is_stub_for_l1 = s_parent_b == l1_idx_b

            l1_nStubs = ak.sum(is_stub_for_l1, axis=2)
            s_type_b = s_type[:, np.newaxis, :]
            l1_nStubs_endcap = ak.sum(is_stub_for_l1 & (s_type_b == 0), axis=2)
            l1_nStubs_barrel = ak.sum(is_stub_for_l1 & (s_type_b == 1), axis=2)

            s_qual_b = s_qual[:, np.newaxis, :]
            masked_qual = ak.mask(s_qual_b, is_stub_for_l1)
            l1_maxQual = ak.fill_none(ak.max(masked_qual, axis=2), 0)

            l1_maxQual_b = l1_maxQual[:, :, np.newaxis]
            is_max_qual_stub = (s_qual_b == l1_maxQual_b) & is_stub_for_l1

            s_etaRegion_b = s_etaRegion[:, np.newaxis, :]
            s_phiRegion_b = s_phiRegion[:, np.newaxis, :]
            s_depthRegion_b = s_depthRegion[:, np.newaxis, :]

            masked_depth_maxqual = ak.mask(s_depthRegion_b, is_max_qual_stub)
            l1_minDepth_maxQual = ak.fill_none(
                ak.min(masked_depth_maxqual, axis=2), 999
            )
            l1_minDepth_maxQual_b = l1_minDepth_maxQual[:, :, np.newaxis]

            is_best_stub = is_max_qual_stub & (s_depthRegion_b == l1_minDepth_maxQual_b)

            s_etaR_full, _ = ak.broadcast_arrays(s_etaRegion_b, is_best_stub)
            s_phiR_full, _ = ak.broadcast_arrays(s_phiRegion_b, is_best_stub)
            s_depthR_full, _ = ak.broadcast_arrays(s_depthRegion_b, is_best_stub)

            l1_bestStub_etaR = ak.fill_none(
                ak.firsts(s_etaR_full[is_best_stub], axis=2), -1
            )
            l1_bestStub_phiR = ak.fill_none(
                ak.firsts(s_phiR_full[is_best_stub], axis=2), -1
            )
            l1_bestStub_depthR = ak.fill_none(
                ak.firsts(s_depthR_full[is_best_stub], axis=2), -1
            )

            def extract_matched_feature(feat_per_l1):
                feat_expanded = feat_per_l1[:, np.newaxis, :]
                feat_matrix, _ = ak.broadcast_arrays(feat_expanded, t_pt)
                best_feat_jagged = feat_matrix[is_best_match]
                return ak.firsts(best_feat_jagged, axis=2)

            matched_nStubs = extract_matched_feature(l1_nStubs)
            matched_nEndcap = extract_matched_feature(l1_nStubs_endcap)
            matched_nBarrel = extract_matched_feature(l1_nStubs_barrel)
            matched_maxQual = extract_matched_feature(l1_maxQual)
            matched_bestStub_etaR = extract_matched_feature(l1_bestStub_etaR)
            matched_bestStub_phiR = extract_matched_feature(l1_bestStub_phiR)
            matched_bestStub_depthR = extract_matched_feature(l1_bestStub_depthR)

            cols.append(impute_linear(matched_nStubs[mask], has_match[mask], fill=0.0))
            final_feature_names.append("L1TkMu_nStubs")

            cols.append(impute_linear(matched_nEndcap[mask], has_match[mask], fill=0.0))
            final_feature_names.append("L1TkMu_nStubs_Endcap")

            cols.append(impute_linear(matched_nBarrel[mask], has_match[mask], fill=0.0))
            final_feature_names.append("L1TkMu_nStubs_Barrel")

            cols.append(impute_linear(matched_maxQual[mask], has_match[mask], fill=0.0))
            final_feature_names.append("L1TkMu_stubQual_max")

            cols.append(
                impute_linear(matched_bestStub_etaR[mask], has_match[mask], fill=-1.0)
            )
            final_feature_names.append("L1TkMu_stubMax_etaRegion")

            cols.append(
                impute_linear(matched_bestStub_phiR[mask], has_match[mask], fill=-1.0)
            )
            final_feature_names.append("L1TkMu_stubMax_phiRegion")

            cols.append(
                impute_linear(matched_bestStub_depthR[mask], has_match[mask], fill=-1.0)
            )
            final_feature_names.append("L1TkMu_stubMax_depthRegion")

        # L1TkMu_hasMatch
        cols.append(ak.to_numpy(ak.flatten(has_match[mask])).astype(np.float32))
        final_feature_names.append("L1TkMu_hasMatch")

        # L1TkMu_dR2min
        cols.append(impute_and_log(min_dR2[mask], has_match[mask], fill=0.1))
        final_feature_names.append("L1TkMu_dR2min")

        # L1TkMu_dPtNorm
        cols.append(impute_and_log(matched_dPt[mask], has_match[mask], fill=1.0))
        final_feature_names.append("L1TkMu_dPtNorm")

        # L1TkMu_chi2Pt
        cols.append(impute_and_log(matched_ratio[mask], has_match[mask], fill=10.0))
        final_feature_names.append("L1TkMu_chi2Pt")

        # L1TkMu_matchingScore
        cols.append(impute_and_log(matched_score[mask], has_match[mask], fill=0.2))
        final_feature_names.append("L1TkMu_matchingScore")

        # Number of compatible L1 candidates (looser cuts)
        loose_dR_cut = 0.5**2
        loose_chi2_cut = 25.0
        is_loose_compatible = (dR2_matrix < loose_dR_cut) & (
            ratio_matrix < loose_chi2_cut
        )
        n_compatible = ak.sum(is_loose_compatible, axis=2)

        cols.append(ak.to_numpy(ak.flatten(n_compatible[mask])).astype(np.float32))
        final_feature_names.append("L1TkMu_nCompatible")

        # Second-best dR2 (sentinel-based to avoid OptionType nesting)
        SENTINEL = 999.0
        dR2_for_second = ak.where(is_best_match, SENTINEL, dR2_matrix)
        dR2_for_second = ak.where(is_compatible, dR2_for_second, SENTINEL)
        second_dR2_raw = ak.min(dR2_for_second, axis=2)
        second_dR2_filled = ak.fill_none(second_dR2_raw, SENTINEL)

        has_second = second_dR2_filled < (SENTINEL - 1.0)

        cols.append(impute_and_log(second_dR2_filled[mask], has_second[mask], fill=1.0))
        final_feature_names.append("L1TkMu_secondBest_dR2")

    # Low pT indicator
    flat_pt = ak.to_numpy(ak.flatten(trk_pt[mask])).astype(np.float32)

    exponent = (flat_pt - 5.0) * 2.0
    exponent = np.clip(exponent, -20.0, 20.0)
    low_pt_indicator = 1.0 / (1.0 + np.exp(exponent))
    cols.append(low_pt_indicator.astype(np.float32))
    final_feature_names.append("is_low_pt")

    # Assemble
    X = np.column_stack(cols).astype(np.float32)
    y = ak.to_numpy(ak.flatten(arr[LABEL_FIELD][mask])).astype(np.int8)

    finite_mask = np.isfinite(X).all(axis=1)
    if not finite_mask.all():
        n_bad = (~finite_mask).sum()
        print(f"  Warning: Removing {n_bad} non-finite rows")
        X = X[finite_mask]
        y = y[finite_mask]
        file_labels_masked = file_labels_masked[finite_mask]

    return X, y, file_labels_masked, final_feature_names


# --------------------------------------------------------------------------- #
# Data loading with caching
# --------------------------------------------------------------------------- #
def load_data(cfg):
    """Load all input files and build the feature matrix.
    Caches to /tmp to avoid re-reading from NFS on restart."""
    import json

    cache_path = "/tmp/seeds_data_cache.npz"
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path} ...")
        cached = np.load(cache_path, allow_pickle=True)
        X = cached["X"]
        y = cached["y"]
        fl = cached["fl"]
        with open("/tmp/seeds_feature_names.json") as f:
            feature_names = json.load(f)
        files_list = pf.get_files(cfg["data_dir"])
        print(f"  Cached: {X.shape}, {len(feature_names)} features")
        return X, y, fl, feature_names, files_list

    files_list = pf.get_files(cfg["data_dir"])
    print(f"Selected {len(files_list)} input files:")
    print(files_list)

    X_list, y_list, fl_list = [], [], []
    feature_names = []
    total_ev = 0
    print(f"Processing {len(files_list)} files ...")
    for i, f in enumerate(files_list):
        print(f"  [{i + 1}/{len(files_list)}] {f}")
        with uproot.open(f) as rf:
            a = rf[main_branch].arrays(tk_branches + l1tkMuon_branches + stub_branches)
            ne = len(a)
            total_ev += ne
            Xc, yc, lc, fn = build_dataset(
                a,
                np.full(ne, i),
                useL1TkMuFeatures=cfg["useL1TkMuFeatures"],
                useL1TkMuStubFeatures=cfg["useL1TkMuStubFeatures"],
                verbose=True,
            )
            X_list.append(Xc)
            y_list.append(yc)
            fl_list.append(lc)
            if i == 0:
                feature_names = fn
            del a, Xc, yc, lc
            gc.collect()

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    fl = np.concatenate(fl_list)
    print(
        f"\nTotal events: {total_ev}  |  Features: {X.shape}  |  "
        f"{len(feature_names)} features"
    )
    print(f"Features: {feature_names}")

    # Cache to local disk for fast restart
    print(f"Caching data to {cache_path} ...")
    np.savez(cache_path, X=X, y=y, fl=fl)
    with open("/tmp/seeds_feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print(f"  Cached: {os.path.getsize(cache_path) / 1e9:.1f} GB")

    return X, y, fl, feature_names, files_list


# --------------------------------------------------------------------------- #
# Main training and evaluation
# --------------------------------------------------------------------------- #
def main():
    cfg = CFG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- Load data ------------------------------------------------------- #
    X, y, fl, feature_names, files_list = load_data(cfg)

    # ---- Feature pruning ------------------------------------------------ #
    drop = cfg.get("drop_features")
    if drop:
        keep_idx = [i for i, fn in enumerate(feature_names) if fn not in drop]
        dropped = [fn for fn in feature_names if fn not in
                   [feature_names[i] for i in keep_idx]]
        X = X[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]
        print(f"\nFeature pruning: dropped {len(drop)} features, kept {len(feature_names)}")
        print(f"  Dropped: {drop}")
        print(f"  Kept: {feature_names}")
    else:
        print(f"\nUsing all {len(feature_names)} features (no pruning)")

    pt_feat_idx = feature_names.index(
        [fn for fn in feature_names if fn.endswith("_pt")][0]
    )

    # ---- Sample weights (identical to DNN) ------------------------------ #
    print("\nComputing sample weights ...")
    pt_vals = 10 ** X[:, pt_feat_idx]
    weights = pf.compute_sample_weights(
        y, pt_vals, signal_boost=cfg["signal_boost"], kin_weight_max=cfg["kin_weight_max"]
    )
    sig = y == 1
    bg = y == 0
    print(
        f"  Signal weight mean: {weights[sig].mean():.3f}  |  "
        f"Background: {weights[bg].mean():.3f}"
    )

    # Low-pT mask (before scaling; trees don't need scaling but we track it for eval)
    low_pt_mask = (pt_vals < pf.LOW_PT_CUT).astype(np.float32)
    print(f"  Low-pT tracks: {low_pt_mask.sum():.0f} ({100 * low_pt_mask.mean():.1f}%)")

    # ---- Split (identical stratification to DNN) ------------------------ #
    print("\nStratified split ...")
    strat = y * len(files_list) + fl
    X_tv, X_test, y_tv, y_test, w_tv, w_test, l_tv, l_test, lp_tv, lp_test = (
        train_test_split(
            X,
            y,
            weights,
            fl,
            low_pt_mask,
            test_size=0.2,
            stratify=strat,
            random_state=42,
        )
    )
    X_train, X_val, y_train, y_val, w_train, w_val, lp_train, lp_val = train_test_split(
        X_tv, y_tv, w_tv, lp_tv, test_size=0.2, stratify=y_tv, random_state=42
    )
    del X, y, weights, low_pt_mask, X_tv, y_tv, w_tv, lp_tv
    gc.collect()
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ---- Train XGBoost --------------------------------------------------- #
    print("\n" + "=" * 70 + "\nXGBOOST TRAINING\n" + "=" * 70)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    params = dict(
        objective="binary:logistic",
        eval_metric=["aucpr", "auc", "logloss"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        min_child_weight=cfg["min_child_weight"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print(f"  Params: {params}")
    print(f"  n_estimators={cfg['n_estimators']}  "
          f"early_stopping={cfg['early_stopping_rounds']}")

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=cfg["early_stopping_rounds"],
        verbose_eval=50,
    )

    best_iter = model.best_iteration
    best_score = model.best_score
    print(f"\n  Best iteration: {best_iter}")
    print(f"  Best val score: {best_score:.6f}")

    # ---- Evaluation ------------------------------------------------------ #
    print("\n" + "=" * 70 + "\nEVALUATION\n" + "=" * 70)

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    y_true = y_test.astype(np.int32)

    prauc = average_precision_score(y_true, y_pred)
    ra = roc_auc_score(y_true, y_pred)
    print(f"\n  Test ROC AUC: {ra:.4f}  |  Test PR AUC: {prauc:.4f}")

    # ROC + PR curves
    roc_auc_val, pr_auc_val, prec_arr, rec_arr, thresholds = pf.plot_roc_pr(
        y_true, y_pred, cfg["output_dir"]
    )

    # F2-optimal threshold
    th1, f1_best, th2, f2_best = pf.find_f2_threshold(y_true, y_pred)
    final_th = th2
    print(f"\n  Optimal threshold (F1): {th1:.4f}  F1={f1_best:.4f}")
    print(f"  Optimal threshold (F2): {th2:.4f}  F2={f2_best:.4f}")
    print(f"  Using F2 threshold: {final_th:.4f}")

    # Classification report + confusion matrix
    yb = (y_pred >= final_th).astype(int)
    print(f"\n{classification_report(y_true, yb, digits=4)}")
    cm = confusion_matrix(y_true, yb)
    print(cm)
    pf.plot_confusion_matrix(y_true, y_pred, final_th, cfg["output_dir"])

    # Per-pT-bin evaluation
    print("\n" + "=" * 70 + "\nPER-pT-BIN PERFORMANCE\n" + "=" * 70)
    pt_test_raw = 10 ** X_test[:, pt_feat_idx]
    pf.evaluate_pt_bins(y_true, y_pred, pt_test_raw, final_th, cfg["output_dir"])

    # Per-file evaluation
    print("\n" + "=" * 70 + "\nPER-FILE PERFORMANCE\n" + "=" * 70)
    for fi, fname in enumerate(files_list):
        fm = l_test == fi
        if fm.sum() == 0:
            continue
        yf, pf_ = y_true[fm], y_pred[fm]
        bf = (pf_ >= final_th).astype(int)
        a = roc_auc_score(yf, pf_) if len(np.unique(yf)) > 1 else float("nan")
        c = confusion_matrix(yf, bf, labels=[0, 1]).ravel()
        tn, fp, fn, tp = c
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f2 = 5 * p * r / (4 * p + r + 1e-6)
        print(
            f"\n  {fname.split('/')[-1]:<30s}  AUC={a:.4f}  Prec={p:.4f}  "
            f"Rec={r:.4f}  F2={f2:.4f}"
            f"\n    {'':30s}  [TN={tn} FP={fp} FN={fn} TP={tp}]"
        )

    # ---- Feature importance (XGBoost gain) ----------------------------- #
    print("\n" + "=" * 70 + "\nFEATURE IMPORTANCE\n" + "=" * 70)

    print("\n[A] Full test set (XGBoost gain):")
    gain = model.get_score(importance_type="gain")
    cover = model.get_score(importance_type="cover")
    name_map = {f"f{i}": fn for i, fn in enumerate(feature_names)}
    gain_named = {name_map.get(k, k): v for k, v in gain.items()}
    cover_named = {name_map.get(k, k): v for k, v in cover.items()}

    sorted_gain = sorted(gain_named.items(), key=lambda x: -x[1])
    print(f"  {'Rank':<6}{'Feature':<55}{'Gain':>12}")
    print("  " + "-" * 73)
    for rank, (fn, gv) in enumerate(sorted_gain):
        cv = cover_named.get(fn, 0.0)
        print(f"  {rank + 1:<6}{fn:<55}{gv:>12.2f}  cover={cv:.1f}")

    with open(cfg["output_dir"] + "feature_importance.txt", "w") as fo:
        fo.write(f"Baseline PR-AUC: {prauc:.6f}\n")
        fo.write(f"Test ROC-AUC: {ra:.6f}\n\n")
        fo.write(f"{'Rank':<6}{'Feature':<55}{'Gain':>12}\n")
        fo.write("-" * 73 + "\n")
        for rank, (fn, gv) in enumerate(sorted_gain):
            cv = cover_named.get(fn, 0.0)
            fo.write(f"{rank + 1:<6}{fn:<55}{gv:>12.2f}  cover={cv:.1f}\n")

    names_arr = [fn for fn, _ in sorted_gain]
    gain_arr = np.array([gv for _, gv in sorted_gain])
    std_arr = np.zeros_like(gain_arr)
    pf.plot_importance(
        gain_arr,
        std_arr,
        names_arr,
        "Feature importance (XGBoost gain) - full test",
        cfg["output_dir"] + "feat_imp_all.png",
    )

    # Low-pT subset importance
    lpm = pt_test_raw < pf.LOW_PT_CUT
    print(
        f"\n[B] Low-pT subset: {lpm.sum()} tracks ({y_true[lpm].sum():.0f} signal)"
    )
    if lpm.sum() > 1000 and len(np.unique(y_true[lpm])) == 2:
        print("  Computing permutation importance on low-pT subset ...")
        il, sl, bl = compute_permutation_importance_xgb(
            model, X_test[lpm], y_true[lpm], feature_names
        )
        pf.plot_importance(
            il,
            sl,
            feature_names,
            "Feature importance (permutation) - low-pT",
            cfg["output_dir"] + "feat_imp_lowpt.png",
        )

    # ---- ONNX export ---------------------------------------------------- #
    print("\n" + "=" * 70 + "\nONNX EXPORT\n" + "=" * 70)
    onnx_path = export_onnx(model, feature_names, cfg)
    if onnx_path:
        verify_onnx(onnx_path, X_test, y_pred, cfg)

    # ---- Compact binary export (forest) --------------------------------- #
    print("\n" + "=" * 70 + "\nCOMPACT BINARY EXPORT (forest)\n" + "=" * 70)
    bin_path = export_compact_bin(model, feature_names, cfg)
    if bin_path:
        verify_compact_bin(bin_path, X_test, y_pred, cfg)

    # ---- Save thresholds ------------------------------------------------ #
    with open(cfg["output_dir"] + "thresholds.txt", "w") as fo:
        fo.write(f"F1_Threshold: {th1}\nF2_Threshold: {th2}\n")
        fo.write(f"Test_ROC_AUC: {ra}\nTest_PR_AUC: {prauc}\n")
        fo.write(f"Best_iteration: {best_iter}\n")
        fo.write(f"Signal_boost: {cfg['signal_boost']}\n")
        fo.write(f"Model: XGBoost\n")

    # ---- Save model ----------------------------------------------------- #
    model_path = cfg["output_dir"] + "model.json"
    model.save_model(model_path)
    print(f"\n  Saved model: {model_path}")


# --------------------------------------------------------------------------- #
# Compact binary export functions (identical to pixel_xgb.py)
# --------------------------------------------------------------------------- #
def _base_logit_from_booster(bst):
    """logit(base_score). XGBoost stores base_score as a probability for
    binary:logistic; the kernel adds baseLogit in margin space."""
    import json

    cfg = json.loads(bst.save_config())
    bs = float(cfg["learner"]["learner_model_param"]["base_score"])
    bs = min(max(bs, 1e-7), 1.0 - 1e-7)
    return float(np.log(bs / (1.0 - bs)))


def _flatten_tree(node_json, feat, val, left, right):
    """Append one tree's nodes (BFS, sequential child allocation) to the
    running lists. Returns the global index of this tree's root."""
    import json

    root_global = len(feat)
    feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
    queue = [(node_json, root_global)]
    while queue:
        n, slot = queue.pop(0)
        if "leaf" in n:
            feat[slot] = -1
            val[slot] = float(n["leaf"])
            left[slot] = -1
            right[slot] = -1
            continue
        feat[slot] = int(n["split"][1:])  # "f12" -> 12
        val[slot] = float(n["split_condition"])
        children = {c["nodeid"]: c for c in n["children"]}
        yes_c = children[n["yes"]]
        no_c = children[n["no"]]
        l_idx = len(feat)
        feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
        r_idx = len(feat)
        feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
        left[slot] = l_idx
        right[slot] = r_idx
        queue.append((yes_c, l_idx))
        queue.append((no_c, r_idx))
    return root_global


def export_compact_bin(model, feature_names, cfg):
    """Export the XGBoost booster to the compact .bin format read by the
    CMSSW forest selector module. Returns the output path."""
    import json
    import struct

    dump = model.get_dump(dump_format="json")
    nt = len(dump)
    feat, val, left, right, roots = [], [], [], [], []
    for t in range(nt):
        r = _flatten_tree(json.loads(dump[t]), feat, val, left, right)
        roots.append(r)

    feat = np.asarray(feat, dtype=np.int8)
    val = np.asarray(val, dtype=np.float32)
    left = np.asarray(left, dtype=np.int32)
    right = np.asarray(right, dtype=np.int32)
    roots = np.asarray(roots, dtype=np.int32)
    base_logit = np.float32(_base_logit_from_booster(model))

    fp16 = cfg.get("compact_bin_fp16", False)
    val_out = val.astype(np.float16).astype(np.float32) if fp16 else val

    path = cfg["output_dir"] + "model_compact.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<iif", feat.shape[0], nt, float(base_logit)))
        f.write(feat.tobytes())
        f.write(val_out.tobytes())
        f.write(left.tobytes())
        f.write(right.tobytes())
        f.write(roots.tobytes())

    size = os.path.getsize(path)
    print(f"  Saved: {path}  ({size / 1024:.0f} KB)")
    print(f"    nNodes={feat.shape[0]}  nTrees={nt}  "
          f"baseLogit={float(base_logit):.8f}  "
          f"val precision={'fp16->fp32' if fp16 else 'fp32'}")
    return path


def verify_compact_bin(bin_path, X_test, y_pred_xgb, cfg):
    """Verify the compact .bin produces the same predictions as XGBoost by
    reimplementing the kernel traversal in numpy."""
    import struct

    with open(bin_path, "rb") as f:
        n_nodes, n_trees, base_logit = struct.unpack("<iif", f.read(12))
        feat = np.frombuffer(f.read(n_nodes), dtype=np.int8)
        val = np.frombuffer(f.read(n_nodes * 4), dtype=np.float32)
        left = np.frombuffer(f.read(n_nodes * 4), dtype=np.int32)
        right = np.frombuffer(f.read(n_nodes * 4), dtype=np.int32)
        roots = np.frombuffer(f.read(n_trees * 4), dtype=np.int32)

    n_check = min(2000, X_test.shape[0])
    X = X_test[:n_check].astype(np.float32)
    scores = np.empty(n_check, dtype=np.float32)
    for i in range(n_check):
        margin = base_logit
        for t in range(n_trees):
            node = roots[t]
            while feat[node] >= 0:
                node = left[node] if X[i, feat[node]] < val[node] else right[node]
            margin += val[node]
        scores[i] = 1.0 / (1.0 + np.exp(-margin))

    max_diff = np.max(np.abs(scores - y_pred_xgb[:n_check]))
    print(f"  Compact .bin vs XGBoost max |diff|: {max_diff:.2e}")
    assert max_diff < 1e-5, (
        f"Compact .bin / XGBoost mismatch: max |diff|={max_diff:.2e} >= 1e-5"
    )
    print(f"  Compact .bin verification PASSED (max |diff| < 1e-5)")


# --------------------------------------------------------------------------- #
# Permutation importance for XGBoost
# --------------------------------------------------------------------------- #
def compute_permutation_importance_xgb(model, X, y, names, n=5):
    """Compute permutation importance by shuffling each feature."""
    rng = np.random.default_rng(0)
    base = _prauc_xgb(model, X, y)
    print(f"  Baseline PR-AUC: {base:.6f}")
    imp = np.zeros((len(names), n), dtype=np.float32)
    for fi, fn in enumerate(names):
        for r in range(n):
            Xp = X.copy()
            rng.shuffle(Xp[:, fi])
            imp[fi, r] = base - _prauc_xgb(model, Xp, y)
        print(
            f"    [{fi:02d}] {fn:<50s}  D={imp[fi].mean():+.5f} +/- {imp[fi].std():.5f}"
        )
    return imp.mean(1), imp.std(1), base


def _prauc_xgb(model, X, y):
    d = xgb.DMatrix(X)
    p = model.predict(d)
    return average_precision_score(y, p)


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #
def export_onnx(model, feature_names, cfg):
    """Export XGBoost model to ONNX via onnxmltools."""
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        import onnx

        n_features = len(feature_names)
        initial_type = [("input", FloatTensorType([None, n_features]))]

        onnx_model = convert_xgboost(
            model,
            initial_types=initial_type,
            target_opset=cfg["onnx_opset"],
        )

        meta = {f"feature_{i}": fn for i, fn in enumerate(feature_names)}
        meta["feature_names"] = ",".join(feature_names)
        for k, v in meta.items():
            entry = onnx.StringStringEntryProto()
            entry.key = k
            entry.value = v
            onnx_model.metadata_props.append(entry)

        path = cfg["output_dir"] + "model_xgb.onnx"
        onnx.save_model(onnx_model, path)
        print(f"  Saved: {path}  ({os.path.getsize(path) / 1024:.0f} KB)")
        return path
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def verify_onnx(onnx_path, X_test, y_pred_xgb, cfg):
    """Verify ONNX model produces the same predictions as XGBoost."""
    try:
        import onnxruntime as rt

        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        outputs = sess.get_outputs()
        print(f"  ONNX outputs: {[(o.name, o.shape, o.type) for o in outputs]}")

        n_check = min(10000, X_test.shape[0])
        onnx_result = sess.run(
            None, {input_name: X_test[:n_check].astype(np.float32)}
        )
        probs = None
        for r in onnx_result:
            if r.dtype == np.float32 and r.ndim == 2 and r.shape[1] == 2:
                probs = r
                break
        if probs is None:
            raise RuntimeError(
                "Could not find probabilities output in ONNX graph. "
                f"Outputs: {[(r.shape, r.dtype) for r in onnx_result]}"
            )
        onnx_pred = probs[:, 1]
        xgb_pred = y_pred_xgb[:n_check]
        max_diff = np.abs(onnx_pred.ravel() - xgb_pred.ravel()).max()
        print(f"  ONNX vs XGBoost max |diff|: {max_diff:.2e}")
        assert max_diff < 1e-5, (
            f"ONNX/XGBoost mismatch: max |diff|={max_diff:.2e} >= 1e-5"
        )
        print(f"  ONNX verification PASSED (max |diff| < 1e-5)")

        single = X_test[:1].astype(np.float32)
        for _ in range(100):
            sess.run(None, {input_name: single})
        t0 = time.perf_counter()
        for _ in range(1000):
            sess.run(None, {input_name: single})
        dt = (time.perf_counter() - t0) / 1000
        print(f"  Inference benchmark (1000 x single-sample CPU): {dt * 1e6:.1f} us/sample")
    except Exception as e:
        print(f"  ONNX verification failed: {e}")
        raise


if __name__ == "__main__":
    main()
