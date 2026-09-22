"""
pixel_features.py - Shared, torch-free feature extraction and evaluation
for the high-purity muon pixel-track selector.

Both pixel_model.py (DNN) and pixel_xgb.py (XGBoost) import from this module
so that the two models see byte-identical features and use the same
evaluation suite, guaranteeing an apples-to-apples architecture comparison.

Contents:
  - Data configuration (paths, sample weights, ONNX opset)
  - Branch / feature-name lists
  - build_dataset()  - awkward -> (X, y, file_labels, feature_names)
  - calculate_metrics(), evaluate_pt_bins()
  - plot_importance(), plot_roc_pr()
"""

import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


# --------------------------------------------------------------------------- #
# Configuration (shared between DNN and XGBoost)
# --------------------------------------------------------------------------- #
DATA_DIR = "/cms-hlt-nfs/user/lferragi/tunedPixelSelector/"

# Sample-weighting (used identically by both models)
SIGNAL_BOOST = 5.0
KIN_WEIGHT_MAX = 20.0

# Low-pT boundary [GeV] - used for stratified eval and the is_low_pt feature
LOW_PT_CUT = 5.0

# ONNX export opset (shared)
ONNX_OPSET = 13


def get_files(data_dir=DATA_DIR):
    """Return sorted list of input ROOT files."""
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
    )


# --------------------------------------------------------------------------- #
# Branch and feature definitions
# --------------------------------------------------------------------------- #
MAIN_BRANCH = "Events"

TK_BRANCHES = [
    "muon_pixel_tracks_p",
    "muon_pixel_tracks_pt",
    "muon_pixel_tracks_ptErr",
    "muon_pixel_tracks_eta",
    "muon_pixel_tracks_etaErr",
    "muon_pixel_tracks_phi",
    "muon_pixel_tracks_phiErr",
    "muon_pixel_tracks_chi2",
    "muon_pixel_tracks_normalizedChi2",
    "muon_pixel_tracks_nPixelHits",
    "muon_pixel_tracks_nTrkLays",
    "muon_pixel_tracks_nFoundHits",
    "muon_pixel_tracks_nLostHits",
    "muon_pixel_tracks_dsz",
    "muon_pixel_tracks_dszErr",
    "muon_pixel_tracks_dxy",
    "muon_pixel_tracks_dxyErr",
    "muon_pixel_tracks_dz",
    "muon_pixel_tracks_dzErr",
    "muon_pixel_tracks_qoverp",
    "muon_pixel_tracks_qoverpErr",
    "muon_pixel_tracks_lambdaErr",
    "muon_pixel_tracks_matched",
    "muon_pixel_tracks_duplicate",
    "muon_pixel_tracks_tpPdgId",
    "muon_pixel_tracks_tpPt",
    "muon_pixel_tracks_tpEta",
    "muon_pixel_tracks_tpPhi",
]

L1TKMUON_BRANCHES = ["L1TkMu_pt", "L1TkMu_eta", "L1TkMu_phi"]

STUB_BRANCHES = [
    "L1TkMuStub_type",
    "L1TkMuStub_quality",
    "L1TkMuStub_parentL1TkMu",
    "L1TkMuStub_etaRegion",
    "L1TkMuStub_phiRegion",
    "L1TkMuStub_depthRegion",
]

# Features stored as log10(|x| + eps)
LOG_FEATURES = [
    "muon_pixel_tracks_p",
    "muon_pixel_tracks_pt",
    "muon_pixel_tracks_ptErr",
    "muon_pixel_tracks_chi2",
    "muon_pixel_tracks_normalizedChi2",
    "muon_pixel_tracks_etaErr",
    "muon_pixel_tracks_phiErr",
    "muon_pixel_tracks_dszErr",
    "muon_pixel_tracks_dxyErr",
    "muon_pixel_tracks_dzErr",
    "muon_pixel_tracks_qoverpErr",
    "muon_pixel_tracks_lambdaErr",
]

# Features stored as-is (plain)
PLAIN_FEATURES = [
    "muon_pixel_tracks_eta",
    "muon_pixel_tracks_nPixelHits",
    "muon_pixel_tracks_nTrkLays",
    "muon_pixel_tracks_nFoundHits",
    "muon_pixel_tracks_nLostHits",
]

LABEL_FIELD = "muon_pixel_tracks_matched"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def delta_phi(phi1, phi2):
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


def impute_and_log(vals, mask, fill=-1.0):
    # Materialise None as `fill` BEFORE going to numpy, so we never get a MaskedArray.
    v = np.asarray(ak.to_numpy(ak.flatten(ak.fill_none(vals, fill))), dtype=np.float64)
    m = ak.to_numpy(ak.flatten(mask))
    # also handles "compatible-but-far" cases (vals real, mask False)
    v[~m] = fill
    return np.log10(np.abs(v) + 1e-6).astype(np.float32)


def impute_linear(vals, mask, fill=0.0):
    # Same fix, for the same reason.
    v = np.asarray(ak.to_numpy(ak.flatten(ak.fill_none(vals, fill))), dtype=np.float64)
    m = ak.to_numpy(ak.flatten(mask))
    v[~m] = fill
    return v.astype(np.float32)


def calculate_metrics(c):
    tp, fp, fn, tn = c
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    a = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    f2 = 5 * p * r / (4 * p + r + 1e-6)
    return p.item(), r.item(), a.item(), f1.item(), f2.item()


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def build_dataset(arr, file_labels_in, useL1TkMuFeatures=True, useL1TkMuStubFeatures=True,
                  verbose=False):
    """
    Builds the feature matrix from an awkward array of events.

    Returns (X, y, file_labels_masked, final_feature_names).
    """
    if verbose:
        print("Building dataset...")

    mask = arr["muon_pixel_tracks_pt"] > 0

    # Expand file labels
    n_tracks_per_event = ak.num(arr["muon_pixel_tracks_pt"])
    file_labels_jagged = ak.unflatten(
        np.repeat(file_labels_in, n_tracks_per_event), n_tracks_per_event
    )
    file_labels_masked = ak.to_numpy(ak.flatten(file_labels_jagged[mask]))

    cols = []
    final_feature_names = []

    trk_pt = arr["muon_pixel_tracks_pt"]
    available_keys = arr.fields

    # Standard features (log and linear)
    for f in LOG_FEATURES:
        if f in available_keys:
            flat = ak.to_numpy(ak.flatten(arr[f][mask])).astype(np.float32)
            cols.append(np.log10(np.abs(flat) + 1e-6))
            final_feature_names.append(f)

    for f in PLAIN_FEATURES:
        if f in available_keys:
            flat = ak.to_numpy(ak.flatten(arr[f][mask])).astype(np.float32)
            cols.append(flat)
            final_feature_names.append(f)

    # Derived features
    if verbose:
        print("Adding derived features...")

    trk_dxy = arr["muon_pixel_tracks_dxy"]
    trk_dz = arr["muon_pixel_tracks_dz"]
    trk_dxyErr = arr["muon_pixel_tracks_dxyErr"]
    trk_dzErr = arr["muon_pixel_tracks_dzErr"]

    # Impact Parameter 3D (log)
    ip3d = trk_dxy**2 + trk_dz**2
    cols.append(ak.to_numpy(ak.flatten(np.log10(ip3d + 1e-6)[mask])).astype(np.float32))
    final_feature_names.append("muon_pixel_tracks_impact3D")

    # Combined Impact Significance (log)
    sip_combined = np.sqrt(
        (trk_dxy / np.maximum(trk_dxyErr, 1e-6)) ** 2
        + (trk_dz / np.maximum(trk_dzErr, 1e-6)) ** 2
    )
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sip_combined + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_pixel_tracks_impactSignificance")

    # Track Quality
    trk_chi2 = arr["muon_pixel_tracks_chi2"]
    trk_nFound = arr["muon_pixel_tracks_nFoundHits"]
    trk_nLost = arr["muon_pixel_tracks_nLostHits"]

    # Chi2 per hit (log)
    chi2_hit = trk_chi2 / np.maximum(trk_nFound, 1)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(chi2_hit + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_pixel_tracks_chi2PerHit")

    # Hit Efficiency
    hit_eff = trk_nFound / np.maximum(trk_nFound + trk_nLost, 1)
    cols.append(ak.to_numpy(ak.flatten(hit_eff[mask])).astype(np.float32))
    final_feature_names.append("muon_pixel_tracks_hitEfficiency")

    # Relative Uncertainties
    trk_ptErr = arr["muon_pixel_tracks_ptErr"]
    trk_p = arr["muon_pixel_tracks_p"]
    trk_qoverp = arr["muon_pixel_tracks_qoverp"]
    trk_qoverpErr = arr["muon_pixel_tracks_qoverpErr"]

    # SigmaPt / Pt (log)
    sigmaPtOverPt = trk_ptErr / np.maximum(trk_pt, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sigmaPtOverPt + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_pixel_tracks_sigmaPtOverPt")

    # Relative Uncertainty Product (log)
    relUncertProd = sigmaPtOverPt * (
        trk_qoverpErr / np.maximum(np.abs(trk_qoverp), 1e-6)
    )
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(relUncertProd + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_pixel_tracks_relUncertaintyProduct")

    # Separated 2D impact parameter significance (log)
    sip_2d = np.abs(trk_dxy) / np.maximum(trk_dxyErr, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sip_2d + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_pixel_tracks_sip2D")

    # Longitudinal impact parameter significance (log)
    sip_z = np.abs(trk_dz) / np.maximum(trk_dzErr, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sip_z + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_pixel_tracks_sipZ")

    # |dxy| / pT
    dxy_over_pt = np.abs(trk_dxy) / np.maximum(trk_pt, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(dxy_over_pt + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_pixel_tracks_dxyOverPt")

    # ptErr / p
    ptErr_over_p = trk_ptErr / np.maximum(trk_p, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(ptErr_over_p + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_pixel_tracks_ptErrOverP")

    # |dz| / |dxy| ratio
    dz_over_dxy = np.abs(trk_dz) / (np.abs(trk_dxy) + 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(dz_over_dxy + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("muon_pixel_tracks_dzOverDxy")

    # |eta|
    trk_eta = arr["muon_pixel_tracks_eta"]
    cols.append(ak.to_numpy(ak.flatten(np.abs(trk_eta)[mask])).astype(np.float32))
    final_feature_names.append("muon_pixel_tracks_absEta")

    # L1 Matching
    if useL1TkMuFeatures:
        if verbose:
            print("Computing L1 matching...")

        t_eta = arr["muon_pixel_tracks_eta"][:, :, np.newaxis]
        t_phi = arr["muon_pixel_tracks_phi"][:, :, np.newaxis]
        t_pt = arr["muon_pixel_tracks_pt"][:, :, np.newaxis]
        t_ptErr = arr["muon_pixel_tracks_ptErr"][:, :, np.newaxis]

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

    exponent = (flat_pt - LOW_PT_CUT) * 2.0
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
# Sample weights (shared between DNN and XGBoost)
# --------------------------------------------------------------------------- #
def compute_sample_weights(y, pt_vals, signal_boost=SIGNAL_BOOST,
                           kin_weight_max=KIN_WEIGHT_MAX):
    """
    Compute sample weights: signal boost + kinematic (1/pT) reweighting.
    Identical to the DNN's weighting so the comparison is fair.
    """
    weights = np.ones(len(y), dtype=np.float32)
    kin_w = np.clip(
        1.0 + np.maximum(0.0, 20.0 / (pt_vals + 0.1) - 1.0), 1.0, kin_weight_max
    ).astype(np.float32)
    sig = y == 1
    bg = y == 0
    kin_sig = kin_w[sig] / (kin_w[sig].mean() + 1e-8)
    kin_bg = kin_w[bg] / (kin_w[bg].mean() + 1e-8)
    weights[sig] = kin_sig * signal_boost
    weights[bg] = kin_bg
    weights /= weights.mean() + 1e-8
    return weights


# --------------------------------------------------------------------------- #
# Per-pT-bin evaluation
# --------------------------------------------------------------------------- #
def evaluate_pt_bins(y_true, y_pred, pt_values, threshold, output_dir):
    """Evaluate precision/recall/F2 in pT bins and produce a summary plot."""
    bins = [(0.0, 5.0), (5.0, 10.0), (10.0, 50.0), (50.0, 200.0), (200.0, 1e6)]
    labels = ["0-5", "5-10", "10-50", "50-200", ">200"]

    results = []
    print(
        f"\n  {'pT bin':>10s} {'N total':>9s} {'N sig':>7s} {'Prec':>7s} "
        f"{'Rec':>7s} {'F2':>7s} {'FN':>6s} {'PR-AUC':>8s}"
    )
    print("  " + "-" * 72)

    for (lo, hi), lab in zip(bins, labels):
        m = (pt_values >= lo) & (pt_values < hi)
        if m.sum() == 0:
            continue
        yt, yp = y_true[m], y_pred[m]
        yb = (yp >= threshold).astype(int)
        n_sig = yt.sum()
        if n_sig == 0 or n_sig == len(yt):
            continue
        cm = confusion_matrix(yt, yb, labels=[0, 1]).ravel()
        tn, fp, fn, tp = cm
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f2 = 5 * p * r / (4 * p + r + 1e-6)
        prauc = average_precision_score(yt, yp)
        results.append((lab, m.sum(), n_sig, p, r, f2, fn, prauc))
        print(
            f"  {lab:>10s} {m.sum():>9d} {n_sig:>7.0f} {p:>7.4f} {r:>7.4f} "
            f"{f2:>7.4f} {fn:>6d} {prauc:>8.4f}"
        )

    # Plot recall + PR-AUC by bin
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        labs = [r[0] for r in results]
        recs = [r[4] for r in results]
        praucs = [r[7] for r in results]
        colors = [
            "#E24B4A" if r < 0.95 else "#378ADD" if r < 0.99 else "#1D9E75"
            for r in recs
        ]

        ax1.bar(range(len(labs)), recs, color=colors, width=0.7)
        ax1.set_xticks(range(len(labs)))
        ax1.set_xticklabels(labs, fontsize=9)
        ax1.set_ylabel("Recall")
        ax1.set_xlabel("pT bin [GeV]")
        ax1.set_title("Recall by pT bin")
        ax1.set_ylim(0.85, 1.005)
        ax1.axhline(0.95, color="#888", ls="--", lw=0.8)
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(range(len(labs)), praucs, color="#4A7AC2", width=0.7)
        ax2.set_xticks(range(len(labs)))
        ax2.set_xticklabels(labs, fontsize=9)
        ax2.set_ylabel("PR-AUC")
        ax2.set_xlabel("pT bin [GeV]")
        ax2.set_title("PR-AUC by pT bin")
        ax2.set_ylim(0.85, 1.005)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir + "pt_bin_performance.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_dir}pt_bin_performance.png")

    return results


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def plot_importance(imp, std, names, title, path, top_n=None):
    top_n = top_n or len(names)
    o = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    c = ["#D85A30" if imp[i] > 0 else "#888" for i in o]
    ax.barh(
        range(top_n), imp[o], xerr=std[o], color=c, ecolor="#444", capsize=3, height=0.7
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([names[i] for i in o], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#444", lw=0.8, ls="--")
    ax.set_xlabel("Mean decrease in PR-AUC")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_pr(y_true, y_pred, output_dir):
    """Plot and save ROC and PR curves. Returns (roc_auc, pr_auc)."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc_val = auc(fpr, tpr)
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc_val = auc(rec_arr, prec_arr)

    for name, xd, yd, xl, yl in [
        ("roc_curve", fpr, tpr, "FPR", "TPR"),
        ("pr_curve", rec_arr, prec_arr, "Recall", "Precision"),
    ]:
        plt.figure(figsize=(6, 5))
        plt.plot(xd, yd, lw=2)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir + name + ".png", dpi=300)
        plt.close()

    return roc_auc_val, pr_auc_val, prec_arr, rec_arr, thresholds


def plot_confusion_matrix(y_true, y_pred, threshold, output_dir):
    """Plot and save confusion matrix. Returns (cm, yb)."""
    yb = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, yb)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=["Pred fake", "Pred signal"],
        yticklabels=["True fake", "True signal"],
        annot_kws={"size": 16},
    )
    plt.title(f"Confusion matrix (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(output_dir + "confusion_matrix.png", dpi=300)
    plt.close()
    return cm, yb


def find_f2_threshold(y_true, y_pred):
    """Find the F2-optimal threshold on the PR curve."""
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_pred)
    f2s = 5 * prec_arr * rec_arr / (4 * prec_arr + rec_arr + 1e-6)
    bi2 = np.argmax(f2s)
    th2 = thresholds[bi2] if bi2 < len(thresholds) else 0.5
    f1s = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-6)
    bi1 = np.argmax(f1s)
    th1 = thresholds[bi1] if bi1 < len(thresholds) else 0.5
    return th1, f1s[bi1], th2, f2s[bi2]
