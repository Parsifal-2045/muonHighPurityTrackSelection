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

import pixel_features as pf

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CFG = dict(
    data_dir="/cms-hlt-nfs/user/lferragi/seedsSelector/",
    output_dir="io/seeds/",
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
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    # Tree budget + data-driven pruning (HP-forest scheme), identical to
    # pixel_xgb.py: fit the full budget with early stopping disabled, then
    # keep the smallest tree prefix whose validation F2 at the validation-
    # chosen per-pT-bin working point is within prune_tol of the best prefix.
    n_estimators=5000,
    early_stopping_rounds=0,  # 0 = disabled; size is set by pruning
    prune_tol=0.001,
    prune_grid_step=50,
    prune_min_trees=100,
    device="cuda",  # GPU hist; falls back to CPU automatically
    # Event-level split ("evt10"): identical scheme to pixel_xgb.py.
    split_mode="evt10",
    split_seed_offset=0,
    # pT-binned working points [GeV] (first set point separates pT < 2 GeV).
    pt_threshold_edges=[0.0, 2.0, 5.0, 10.0, 50.0, 200.0],
    min_bin_signal=100,
    # Previous production forest, re-scored on the same evt10 split with
    # validation-derived working points for an honest comparison.
    ref_model_path="io/archive/seeds_xgb_output_33f/model.json",
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
                  verbose=False, event_ids=None):
    """Build the 44-feature matrix from muon_general_tracks events.
    Identical feature extraction to seeds_model.py build_dataset().
    If event_ids (one id per event) is given, a 5th return element carries
    the per-track event ids after the same masks."""
    print("Building dataset...")

    mask = arr["muon_general_tracks_pt"] > 0

    n_tracks_per_event = ak.num(arr["muon_general_tracks_pt"])
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
        if event_ids_masked is not None:
            event_ids_masked = event_ids_masked[finite_mask]

    if event_ids is not None:
        return X, y, file_labels_masked, event_ids_masked, final_feature_names
    return X, y, file_labels_masked, final_feature_names


# --------------------------------------------------------------------------- #
# Data loading with caching
# --------------------------------------------------------------------------- #
def load_data(cfg):
    """Load all input files and build the feature matrix.

    Returns (X, y, fl, ev, feature_names, files_list); ev is a per-track event
    id (local to its input file) used for the event-level split. Caches to
    /tmp; the cache carries an input-file manifest and is rebuilt when the
    manifest or the cache format (ev ids) does not match."""
    import json

    cache_path = "/tmp/seeds_data_cache.npz"
    names_path = "/tmp/seeds_feature_names.json"
    files_list = pf.get_files(cfg["data_dir"])
    if os.path.exists(cache_path):
        with open(names_path) as f:
            feature_names = json.load(f)
        cached = np.load(cache_path, allow_pickle=True)
        manifest_ok = (
            "ev" in cached and "files" in cached
            and [str(x) for x in cached["files"]] == files_list
        )
        if manifest_ok:
            print(f"Loading cached data from {cache_path} ...")
            X = cached["X"]
            y = cached["y"]
            fl = cached["fl"]
            ev = cached["ev"]
            print(f"  Cached: {X.shape}, {len(feature_names)} features")
            return X, y, fl, ev, feature_names, files_list
        print(f"Cache at {cache_path} is stale (format or inputs); rebuilding.")

    print(f"Selected {len(files_list)} input files:")
    print(files_list)

    X_list, y_list, fl_list, ev_list = [], [], [], []
    feature_names = []
    total_ev = 0
    print(f"Processing {len(files_list)} files ...")
    for i, f in enumerate(files_list):
        print(f"  [{i + 1}/{len(files_list)}] {f}")
        with uproot.open(f) as rf:
            a = rf[main_branch].arrays(tk_branches + l1tkMuon_branches + stub_branches)
            ne = len(a)
            total_ev += ne
            Xc, yc, lc, ec, fn = build_dataset(
                a,
                np.full(ne, i),
                useL1TkMuFeatures=cfg["useL1TkMuFeatures"],
                useL1TkMuStubFeatures=cfg["useL1TkMuStubFeatures"],
                verbose=True,
                event_ids=np.arange(ne, dtype=np.int64),
            )
            X_list.append(Xc)
            y_list.append(yc)
            fl_list.append(lc)
            ev_list.append(ec)
            if i == 0:
                feature_names = fn
            del a, Xc, yc, lc, ec
            gc.collect()

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    fl = np.concatenate(fl_list)
    ev = np.concatenate(ev_list)
    print(
        f"\nTotal events: {total_ev}  |  Features: {X.shape}  |  "
        f"{len(feature_names)} features"
    )
    print(f"Features: {feature_names}")

    # Cache to local disk for fast restart
    print(f"Caching data to {cache_path} ...")
    np.savez(cache_path, X=X, y=y, fl=fl, ev=ev, files=np.array(files_list))
    with open(names_path, "w") as f:
        json.dump(feature_names, f)
    print(f"  Cached: {os.path.getsize(cache_path) / 1e9:.1f} GB")

    return X, y, fl, ev, feature_names, files_list


# --------------------------------------------------------------------------- #
# Main training and evaluation
# --------------------------------------------------------------------------- #
def main():
    cfg = CFG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    _log_fh = pf.tee_log(cfg["output_dir"])  # full log stored in the model dir

    # ---- Load data ------------------------------------------------------- #
    X, y, fl, ev, feature_names, files_list = load_data(cfg)

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

    # ---- Split (event-level "evt10": train 60 / val 10 / test 30) -------- #
    # Every track of an event lands in the same split, so correlated tracks
    # cannot leak between train and test. Working points and the forest size
    # are derived on the validation split only; the test split is evaluated
    # once, with frozen choices.
    print(f"\nEvent-level split ({cfg['split_mode']}) ...")
    assert cfg["split_mode"] == "evt10", f"unknown split_mode {cfg['split_mode']}"
    tr_m, va_m, te_m = pf.evt10_split(ev, seed_offset=cfg["split_seed_offset"])
    X_train, y_train, w_train = X[tr_m], y[tr_m], weights[tr_m]
    X_val, y_val, w_val, lp_val = X[va_m], y[va_m], weights[va_m], low_pt_mask[va_m]
    X_test, y_test, w_test, l_test, lp_test = (
        X[te_m], y[te_m], weights[te_m], fl[te_m], low_pt_mask[te_m],
    )
    for name, (m_, yy) in {
        "train": (tr_m, y_train), "val": (va_m, y_val), "test": (te_m, y_test),
    }.items():
        print(
            f"  {name:5s}: {len(yy):>9d} tracks ({yy.sum():>9d} signal, "
            f"{100 * yy.mean():.1f}%)  in {len(np.unique(ev[m_]))} events"
        )
    del X, y, weights, low_pt_mask, tr_m, va_m, te_m, ev
    gc.collect()

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
        random_state=42,
    )
    esr = cfg["early_stopping_rounds"] or None
    try:
        device = cfg.get("device", "cpu")
        p = dict(params, device=device, n_jobs=-1) if device == "cpu" \
            else dict(params, device=device)
        print(f"  Params: {p}")
        print(f"  n_estimators={cfg['n_estimators']} (budget)  "
              f"early_stopping={cfg['early_stopping_rounds']}  "
              f"prune_tol={cfg['prune_tol']}")
        evals_result = {}
        model = xgb.train(
            p, dtrain,
            num_boost_round=cfg["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=esr,
            verbose_eval=100,
        )
    except Exception as e:
        if cfg.get("device", "cpu") == "cpu":
            raise
        print(f"  !! device={cfg['device']} failed ({e}); falling back to CPU")
        p = dict(params, device="cpu", n_jobs=-1)
        evals_result = {}
        model = xgb.train(
            p, dtrain,
            num_boost_round=cfg["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=esr,
            verbose_eval=100,
        )

    n_fitted = len(model.get_dump())
    print(f"\n  Fitted boosting rounds: {n_fitted} (budget {cfg['n_estimators']})")

    # ---- Data-driven pruning (size chosen on validation) ----------------- #
    # Prefix curve: validation F2 at the validation-chosen per-pT-bin working
    # point, for every prefix of the forest on a prune_grid_step grid. Tree
    # margins are accumulated in prune_grid_step-sized blocks, so the grid
    # costs one full pass over the forest, not one prediction per grid point.
    # The smallest prefix within prune_tol of the best prefix is kept; the
    # booster is then sliced to it. Identical scheme to pixel_xgb.py.
    print("\n" + "=" * 70 + "\nTREE PRUNING (validation)\n" + "=" * 70)
    pt_val_raw = 10 ** X_val[:, pt_feat_idx]
    base_margin = _base_logit_from_booster(model)
    step = cfg["prune_grid_step"]
    g0 = step * max(1, int(np.ceil(cfg["prune_min_trees"] / step)))
    grid = sorted(set(range(g0, n_fitted, step)) | {n_fitted})
    margins_at_grid = {}
    t0 = time.perf_counter()
    dval_margin = xgb.DMatrix(X_val)
    run = np.full(len(y_val), base_margin, dtype=np.float64)
    for t0_tree in range(0, n_fitted, step):
        t1_tree = min(t0_tree + step, n_fitted)
        run += model.predict(
            dval_margin, iteration_range=(t0_tree, t1_tree), output_margin=True
        ) - base_margin
        if t1_tree in grid:
            margins_at_grid[t1_tree] = run.copy()
    assert set(margins_at_grid) == set(grid)
    del dval_margin
    print(f"  Prefix margins accumulated in {time.perf_counter() - t0:.0f}s "
          f"({len(grid)} grid points)")

    def _val_f2_binned(scores):
        wpb = pf.find_f2_threshold_binned(
            y_val, scores, pt_val_raw,
            cfg["pt_threshold_edges"], cfg["min_bin_signal"],
        )
        dec = pf.apply_binned_thresholds(
            scores, pt_val_raw, cfg["pt_threshold_edges"],
            [b[2] for b in wpb["bins"]],
        )
        tn_, fp_, fn_, tp_ = confusion_matrix(y_val, dec, labels=[0, 1]).ravel()
        _, _, _, _, f2 = pf.calculate_metrics((tp_, fp_, fn_, tn_))
        return f2

    size_curve = []
    best_f2, best_n = -1.0, n_fitted
    for n in grid:
        s = 1.0 / (1.0 + np.exp(-margins_at_grid[n]))
        f2 = _val_f2_binned(s)
        size_curve.append((n, f2))
        if f2 > best_f2:
            best_f2, best_n = f2, n
    print("  Size curve (val F2 @ per-bin WP): "
          + " ".join(f"{n}:{f2:.5f}" for n, f2 in size_curve))
    chosen_n = min(n for n, f2 in size_curve if f2 >= best_f2 - cfg["prune_tol"])
    print(f"  Best prefix: {best_n} (val F2 {best_f2:.5f}); "
          f"tolerance {cfg['prune_tol']} -> keeping {chosen_n} trees")

    # Slice the booster to the chosen prefix.
    model = model[0:chosen_n]
    assert len(model.get_dump()) == chosen_n
    # Sanity: the sliced booster's predictions must reproduce the recorded
    # prefix scores (catches any base-score bookkeeping mistake).
    s_val = model.predict(dval)
    d_max = np.abs(
        s_val - 1.0 / (1.0 + np.exp(-margins_at_grid[chosen_n]))
    ).max()
    margins_at_grid.clear()
    gc.collect()
    assert d_max < 1e-5, f"sliced booster deviates from prefix margins: {d_max:.2e}"
    print(f"  Sliced booster: {chosen_n} trees (prefix-consistency {d_max:.2e})")

    # ---- Working points (validation only, frozen for test) --------------- #
    print("\n" + "=" * 70 + "\nWORKING POINTS (validation)\n" + "=" * 70)
    wp = pf.find_f2_threshold_binned(
        y_val, s_val, pt_val_raw, cfg["pt_threshold_edges"], cfg["min_bin_signal"]
    )
    th1, th2 = wp["global_f1"], wp["global_f2"]
    bin_edges = cfg["pt_threshold_edges"]
    bin_thrs = [b[2] for b in wp["bins"]]
    print(f"  Global F1 threshold: {th1:.6f}   Global F2 threshold: {th2:.6f}")
    print(f"  Per-pT-bin F2 set points (min_bin_signal={cfg['min_bin_signal']}):")
    for lo, hi, bthr, n_sig, n_rows, fb in wp["bins"]:
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.0f}"
        tag = "  (fallback->global)" if fb else ""
        print(f"    pT [{lo:6.1f}, {hi_s:>4s}) GeV: thr={bthr:.6f}  "
              f"n_sig={n_sig}  n={n_rows}{tag}")

    # ---- Evaluation (test split, frozen working points) ------------------ #
    print("\n" + "=" * 70 + "\nEVALUATION (test, frozen WPs)\n" + "=" * 70)

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    y_true = y_test.astype(np.int32)
    pt_test_raw = 10 ** X_test[:, pt_feat_idx]

    prauc = average_precision_score(y_true, y_pred)
    ra = roc_auc_score(y_true, y_pred)
    print(f"\n  Test ROC AUC: {ra:.4f}  |  Test PR AUC: {prauc:.4f}")

    # ROC + PR curves
    roc_auc_val, pr_auc_val, prec_arr, rec_arr, thresholds = pf.plot_roc_pr(
        y_true, y_pred, cfg["output_dir"]
    )

    # Decisions with the frozen working points: global F2 and per-pT-bin F2.
    dec_global = y_pred >= th2
    dec_bins = pf.apply_binned_thresholds(y_pred, pt_test_raw, bin_edges, bin_thrs)

    tn, fp, fn, tp = confusion_matrix(y_true, dec_global, labels=[0, 1]).ravel()
    p_g, r_g, a_g, f1_g, f2_g = pf.calculate_metrics((tp, fp, fn, tn))
    tn, fp, fn, tp = confusion_matrix(y_true, dec_bins, labels=[0, 1]).ravel()
    p_b, r_b, a_b, f1_b, f2_b = pf.calculate_metrics((tp, fp, fn, tn))
    print(f"\n  Test @ global F2 WP ({th2:.6f}): P={p_g:.4f} R={r_g:.4f} F2={f2_g:.4f}")
    print(f"  Test @ per-bin F2 WPs:           P={p_b:.4f} R={r_b:.4f} F2={f2_b:.4f}")

    # Classification report + confusion matrices. confusion_matrix.png holds
    # the deployment configuration (per-pT-bin set points); the global-WP
    # matrix is kept alongside for reference.
    print(f"\n{classification_report(y_true, dec_bins.astype(int), digits=4)}")
    cm_bins, _ = pf.plot_confusion_matrix(
        y_true, None, 0, cfg["output_dir"], decisions=dec_bins,
        title_suffix="(per-pT-bin F2 working points)",
    )
    print(cm_bins)
    pf.plot_confusion_matrix(
        y_true, y_pred, th2, cfg["output_dir"],
        title_suffix="(global F2 working point)",
        filename="confusion_matrix_globalthr.png",
    )

    # Per-pT-bin evaluation (decisions use the per-bin set points)
    print("\n" + "=" * 70 + "\nPER-pT-BIN PERFORMANCE\n" + "=" * 70)
    pf.evaluate_pt_bins(y_true, y_pred, pt_test_raw, th2, cfg["output_dir"],
                        decisions=dec_bins)

    # Per-file evaluation
    print("\n" + "=" * 70 + "\nPER-FILE PERFORMANCE\n" + "=" * 70)
    for fi, fname in enumerate(files_list):
        fm = l_test == fi
        if fm.sum() == 0:
            continue
        yf, pf_ = y_true[fm], y_pred[fm]
        bf = dec_bins[fm].astype(int)
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
    f2_at_chosen = dict(size_curve)[chosen_n]
    with open(cfg["output_dir"] + "thresholds.txt", "w") as fo:
        fo.write(f"F1_Threshold: {th1}\nF2_Threshold: {th2}\n")
        fo.write(f"Test_ROC_AUC: {ra}\nTest_PR_AUC: {prauc}\n")
        fo.write(f"Test_F2_global_WP: {f2_g}\nTest_F2_perbin_WP: {f2_b}\n")
        fo.write(f"Split: evt10 (event-level, train 60 / val 10 / test 30)\n")
        fo.write(f"Tree_budget: {cfg['n_estimators']}\n")
        fo.write(f"Chosen_trees: {chosen_n} (prune_tol={cfg['prune_tol']}, "
                 f"val F2 {f2_at_chosen:.6f}, best prefix {best_n} at {best_f2:.6f})\n")
        fo.write(f"Signal_boost: {cfg['signal_boost']}\n")
        fo.write("pT-bin F2 set points (validation-derived):\n")
        for lo, hi, bthr, n_sig, n_rows, fb in wp["bins"]:
            hi_s = "inf" if not np.isfinite(hi) else f"{hi:.1f}"
            tag = "  [fallback->global]" if fb else ""
            fo.write(f"  pT [{lo:.1f}, {hi_s}): thr={bthr:.6f}  "
                     f"n_val_sig={n_sig}{tag}\n")
        fo.write(f"Model: XGBoost\n")

    import json as _json

    with open(cfg["output_dir"] + "thresholds.json", "w") as fo:
        _json.dump(
            {
                "split": "evt10",
                "tree_budget": cfg["n_estimators"],
                "chosen_trees": chosen_n,
                "prune_tol": cfg["prune_tol"],
                "val_f2_at_chosen": f2_at_chosen,
                "global_f1_threshold": th1,
                "global_f2_threshold": th2,
                "pt_bin_edges": bin_edges,
                "pt_bin_f2_thresholds": bin_thrs,
                "test": {
                    "roc_auc": ra,
                    "pr_auc": prauc,
                    "f2_global_wp": f2_g,
                    "f2_perbin_wp": f2_b,
                    "precision_perbin_wp": p_b,
                    "recall_perbin_wp": r_b,
                },
                "model": "XGBoost",
            },
            fo,
            indent=2,
        )
    print(f"  Saved: {cfg['output_dir']}thresholds.txt / thresholds.json")

    # ---- Save model ----------------------------------------------------- #
    model_path = cfg["output_dir"] + "model.json"
    model.save_model(model_path)
    print(f"\n  Saved model: {model_path}")
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



    # ---- Re-score previous production forest on the same split ----------- #
    ref_path = cfg.get("ref_model_path")
    if ref_path and os.path.isfile(ref_path):
        print("\n" + "=" * 70 +
              "\nREFERENCE FOREST RE-SCORE (same evt10 split, val-derived WPs)\n"
              + "=" * 70)
        pf.rescore_reference_forest(
            ref_path, X_val, y_val, pt_val_raw, X_test, y_test, pt_test_raw, cfg
        )
    else:
        print(f"\n  Reference forest not found ({ref_path}), skipping re-score")


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
