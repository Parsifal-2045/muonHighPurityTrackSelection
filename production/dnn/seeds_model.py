"""
MuonSelectionDNN - High-Purity Muon Track Selector

Input samples produced from CMSSW with the NANO:@MUHLTTraining flavour

Run the training with:
    torchrun --standalone --nproc_per_node=<NGPUs> seeds_model.py
"""

import copy
import gc
import math
import os
import time

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import uproot
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Configuration
CFG = dict(
    data_dir="/cms-hlt-nfs/user/lferragi/seedsSelector/",
    output_dir="seeds_output/",
    # use stubs
    useL1TkMuStubFeatures=True,
    # architecture
    hidden_dim=144,
    n_res_blocks=3,
    dropout=0.1,
    # pT-stratified loss
    f2_beta_sq=4.0,
    focal_weight=0.5,
    focal_alpha=0.8,
    focal_gamma=2.0,
    low_pt_loss_weight=5.0,  # weight of low-pT F2 relative to high-pT F2
    min_stratum_sig=32,  # minimum signal tracks per stratum to stratify
    # sample weights
    signal_boost=5.0,
    kin_weight_max=20.0,
    # training
    batch_size=2048,
    max_epochs=500,
    max_lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    patience=50,
    lr_warmup_frac=0.05,
    # EMA
    ema_decay=0.9999,
    # ONNX
    onnx_opset=13,
)

# Input files
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


# Helpers
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


# Dataset construction
def build_dataset(arr, file_labels_in, useL1TkMuFeatures=True, verbose=False):
    """
    Builds the feature matrix
    """
    print("Building dataset...")

    mask = arr["muon_general_tracks_pt"] > 0

    # Expand file labels
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
        if "L1TkMuStub_parentL1TkMu" in arr.fields and CFG["useL1TkMuStubFeatures"]:
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


# Model
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class MuonSelectionDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=144, n_blocks=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.output_head(self.res_blocks(self.input_proj(x)))


# pT-stratified DifferentiableF2Loss
class StratifiedF2Loss(nn.Module):
    """
    Computes soft F2 loss separately for low-pT and high-pT tracks,
    then combines them with configurable weighting.

    Why stratify:
      The batch-level F2 in V2/V4 is dominated by the ~72% of signal at
      medium/high pT.  A FN at 100 GeV generates the same gradient as a
      FN at 1 GeV, but the former is trivially fixable while the latter
      requires learning subtle track quality patterns.  The optimizer
      therefore prioritises easy high-pT corrections.

      By computing F2 separately per stratum and weighting the low-pT
      stratum 3x, we force the optimizer to allocate gradient budget
      proportional to where the model actually struggles.

    Fallback:
      If a batch has too few low-pT signal tracks (< min_sig), the loss
      falls back to a single global F2 to avoid noisy gradients.
    """

    def __init__(
        self,
        beta_sq=4.0,
        low_pt_weight=3.0,
        focal_weight=0.5,
        focal_alpha=0.8,
        focal_gamma=2.0,
        min_sig=16,
    ):
        super().__init__()
        self.b2 = beta_sq
        self.lpw = low_pt_weight
        self.fw = focal_weight
        self.fa = focal_alpha
        self.fg = focal_gamma
        self.min_sig = min_sig
        self.bce = nn.BCELoss(reduction="none")

    def _soft_fbeta(self, p, t, w):
        """Compute soft F-beta from probabilities, targets, weights."""
        tp = (p * t * w).sum()
        fp = (p * (1 - t) * w).sum()
        fn = ((1 - p) * t * w).sum()
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        return (1 + self.b2) * prec * rec / (self.b2 * prec + rec + 1e-6)

    def forward(self, probs, targets, weights, is_low_pt):
        targets = targets.view_as(probs)
        weights = weights.view_as(probs)
        is_low_pt = is_low_pt.view_as(probs).bool()

        low = is_low_pt.squeeze(-1)
        high = ~low

        # Count signal tracks in each stratum
        n_low_sig = (targets[low] > 0.5).sum().item() if low.any() else 0
        n_high_sig = (targets[high] > 0.5).sum().item() if high.any() else 0

        # Stratified F2 if both strata have enough signal; else global
        if n_low_sig >= self.min_sig and n_high_sig >= self.min_sig:
            f2_low = self._soft_fbeta(probs[low], targets[low], weights[low])
            f2_high = self._soft_fbeta(probs[high], targets[high], weights[high])
            loss_f2 = self.lpw * (1 - f2_low) + (1 - f2_high)
        else:
            f2_all = self._soft_fbeta(probs, targets, weights)
            loss_f2 = (1 + self.lpw) * (1 - f2_all)

        # Focal component (global, per-sample — unchanged)
        bce = self.bce(probs, targets)
        pt_focal = torch.exp(-bce)
        focal = self.fa * (1 - pt_focal) ** self.fg * bce
        focal = (focal * weights).mean()

        return loss_f2 + self.fw * focal


# Dataset, EMA, BN fusion
class NumpyDataset(Dataset):
    """Returns (features, label, weight, is_low_pt_bool)."""

    def __init__(self, X, y, w, low_pt_mask):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
        self.w = torch.from_numpy(w.astype(np.float32)).unsqueeze(1)
        self.lp = torch.from_numpy(low_pt_mask.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i], self.lp[i]


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ep, mp in zip(self.ema.parameters(), model.parameters()):
            ep.data.mul_(d).add_(mp.data, alpha=1 - d)
        for eb, mb in zip(self.ema.buffers(), model.buffers()):
            eb.data.copy_(mb.data)


def fuse_bn_into_linear(linear, bn):
    W, b = (
        linear.weight.data,
        linear.bias.data
        if linear.bias is not None
        else torch.zeros(linear.out_features),
    )
    gamma, beta, mu = bn.weight.data, bn.bias.data, bn.running_mean.data
    sigma = torch.sqrt(bn.running_var.data + bn.eps)
    scale = gamma / sigma
    fused = nn.Linear(linear.in_features, linear.out_features)
    fused.weight = nn.Parameter(scale.unsqueeze(1) * W)
    fused.bias = nn.Parameter(scale * (b - mu) + beta)
    return fused


def fuse_sequential_bn(seq):
    modules = list(seq.children())
    new = []
    i = 0
    while i < len(modules):
        if (
            i + 1 < len(modules)
            and isinstance(modules[i], nn.Linear)
            and isinstance(modules[i + 1], nn.BatchNorm1d)
        ):
            new.append(fuse_bn_into_linear(modules[i], modules[i + 1]))
            i += 2
        elif isinstance(modules[i], nn.Dropout):
            i += 1
        else:
            new.append(modules[i])
            i += 1
    return nn.Sequential(*new)


def make_inference_model(state_dict, input_dim, scaler, cfg):
    model = MuonSelectionDNN(
        input_dim, cfg["hidden_dim"], cfg["n_res_blocks"], cfg["dropout"]
    )
    model.load_state_dict(state_dict)
    model.eval()
    scale_t = torch.from_numpy(scaler.scale_.astype(np.float32))
    mean_t = torch.from_numpy(scaler.mean_.astype(np.float32))
    first = model.input_proj[0]
    with torch.no_grad():
        W = first.weight.clone()
        b = first.bias.clone()
        first.weight.copy_(W / scale_t.unsqueeze(0))
        first.bias.copy_(b - ((W / scale_t.unsqueeze(0)) * mean_t.unsqueeze(0)).sum(1))
    model.input_proj = fuse_sequential_bn(model.input_proj)
    model.output_head = fuse_sequential_bn(model.output_head)
    for blk in model.res_blocks:
        blk.block = fuse_sequential_bn(blk.block)
    return model


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def gather_probs_targets(pl, tl, dev):
    lp = torch.cat(pl).cpu()
    lt = torch.cat(tl).cpu()
    ws = dist.get_world_size()
    gp, gt = [None] * ws, [None] * ws
    dist.all_gather_object(gp, lp)
    dist.all_gather_object(gt, lt)
    if dist.get_rank() == 0:
        return torch.cat(gp).numpy().ravel(), torch.cat(gt).numpy().ravel()
    return None, None


# Feature importance
def _prauc(model, X, y, bs=8192, dev="cpu"):
    model.eval()
    ps = []
    Xt = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        for i in range(0, len(X), bs):
            ps.append(model(Xt[i : i + bs].to(dev)).cpu().numpy())
    return average_precision_score(y, np.concatenate(ps).ravel())


def compute_permutation_importance(model, X, y, names, n=5, bs=8192, dev="cpu"):
    rng = np.random.default_rng(0)
    base = _prauc(model, X, y, bs, dev)
    print(f"  Baseline PR-AUC: {base:.6f}")
    imp = np.zeros((len(names), n), dtype=np.float32)
    for fi, fn in enumerate(names):
        for r in range(n):
            Xp = X.copy()
            rng.shuffle(Xp[:, fi])
            imp[fi, r] = base - _prauc(model, Xp, y, bs, dev)
        print(
            f"    [{fi:02d}] {fn:<50s}  D={imp[fi].mean():+.5f} +/- {imp[fi].std():.5f}"
        )
    return imp.mean(1), imp.std(1), base


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


# Per-pT-bin evaluation
def evaluate_pt_bins(y_true, y_pred, pt_values, threshold, output_dir):
    """Evaluate precision/recall/F2 in pT bins and produce a summary plot."""
    bins = [(0.0, 5.0), (5.0, 10.0), (10.0, 50.0), (50.0, 200.0), (200.0, 1e6)]
    labels = ["0-5", "5-10", "10-50", "50-200", ">200"]

    results = []
    print(
        f"\n  {'pT bin':>10s} {'N total':>9s} {'N sig':>7s} {'Prec':>7s} {'Rec':>7s} {'F2':>7s} {'FN':>6s} {'PR-AUC':>8s}"
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
            f"  {lab:>10s} {m.sum():>9d} {n_sig:>7.0f} {p:>7.4f} {r:>7.4f} {f2:>7.4f} {fn:>6d} {prauc:>8.4f}"
        )

    # Plot recall by bin
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


def main():
    cfg = CFG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Load data
    X_list, y_list, fl_list = [], [], []
    feature_names = []
    total_ev = 0
    print(f"Processing {len(files)} files ...")
    for i, f in enumerate(files):
        print(f"  [{i + 1}/{len(files)}] {f}")
        with uproot.open(f) as rf:
            a = rf[main_branch].arrays(tk_branches + l1tkMuon_branches + stub_branches)
            ne = len(a)
            total_ev += ne
            Xc, yc, lc, fn = build_dataset(a, np.full(ne, i))
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
        f"\nTotal events: {total_ev}  |  Features: {X.shape}  |  {len(feature_names)} features"
    )
    print(f"Features: {feature_names}")

    pt_feat_idx = feature_names.index("muon_general_tracks_pt")

    # Sample weights (signal boost)
    print("\nComputing sample weights ...")
    weights = np.ones(len(y), dtype=np.float32)
    pt_vals = 10 ** X[:, pt_feat_idx]
    kin_w = np.clip(
        1.0 + np.maximum(0.0, 20.0 / (pt_vals + 0.1) - 1.0), 1.0, cfg["kin_weight_max"]
    ).astype(np.float32)
    sig = y == 1
    bg = y == 0
    kin_sig = kin_w[sig] / (kin_w[sig].mean() + 1e-8)
    kin_bg = kin_w[bg] / (kin_w[bg].mean() + 1e-8)
    weights[sig] = kin_sig * cfg["signal_boost"]
    weights[bg] = kin_bg
    weights /= weights.mean() + 1e-8
    print(
        f"  Signal weight mean: {weights[sig].mean():.3f}  |  Background: {weights[bg].mean():.3f}"
    )

    # Precompute low-pT boolean mask (before scaling)
    low_pt_mask = (pt_vals < 5.0).astype(np.float32)
    print(f"  Low-pT tracks: {low_pt_mask.sum():.0f} ({100 * low_pt_mask.mean():.1f}%)")

    # Split
    print("\nStratified split ...")
    strat = y * len(files) + fl
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

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # DDP + Model
    ddp_setup()
    lr = int(os.environ["LOCAL_RANK"])
    gr = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])

    tds = NumpyDataset(X_train, y_train, w_train, lp_train)
    vds = NumpyDataset(X_val, y_val, w_val, lp_val)
    ts = DistributedSampler(tds)
    vs = DistributedSampler(vds, shuffle=False)
    tl = DataLoader(tds, batch_size=cfg["batch_size"], sampler=ts, pin_memory=True)
    vl = DataLoader(vds, batch_size=cfg["batch_size"], sampler=vs, pin_memory=True)

    input_dim = X_train.shape[1]
    model = MuonSelectionDNN(
        input_dim, cfg["hidden_dim"], cfg["n_res_blocks"], cfg["dropout"]
    ).to(lr)
    model = DDP(model, device_ids=[lr])

    criterion = StratifiedF2Loss(
        beta_sq=cfg["f2_beta_sq"],
        low_pt_weight=cfg["low_pt_loss_weight"],
        focal_weight=cfg["focal_weight"],
        focal_alpha=cfg["focal_alpha"],
        focal_gamma=cfg["focal_gamma"],
        min_sig=cfg["min_stratum_sig"],
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["max_lr"], weight_decay=cfg["weight_decay"]
    )

    samples_per_gpu = math.ceil(len(tds) / ws)
    steps_per_epoch = math.ceil(samples_per_gpu / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["max_epochs"]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["max_lr"],
        total_steps=total_steps,
        pct_start=cfg["lr_warmup_frac"],
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1000,
    )

    ema = None
    if gr == 0:
        ema = ModelEMA(model.module, cfg["ema_decay"])
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel ({n_params} params):")
        print(model)
        print(f"\n--- Training on {ws} GPU(s), {cfg['max_epochs']} epochs max ---")
        print(f"--- pT-stratified loss: low_pt_weight={cfg['low_pt_loss_weight']} ---")

    # Training loop
    best_prauc = 0.0
    best_state = None
    counter = 0
    stop = torch.tensor(0, device=lr)

    for epoch in range(cfg["max_epochs"]):
        ts.set_epoch(epoch)
        model.train()
        run_loss = 0.0
        tcnt = torch.zeros(4, device=lr)

        for inp, tgt, w, lp_batch in tl:
            inp, tgt, w, lp_batch = inp.to(lr), tgt.to(lr), w.to(lr), lp_batch.to(lr)
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, tgt, w, lp_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            if gr == 0:
                ema.update(model.module)
            run_loss += loss.item()
            with torch.no_grad():
                pb = (out > 0.5).float()
                tcnt[0] += ((pb == 1) & (tgt == 1)).sum()
                tcnt[1] += ((pb == 1) & (tgt == 0)).sum()
                tcnt[2] += ((pb == 0) & (tgt == 1)).sum()
                tcnt[3] += ((pb == 0) & (tgt == 0)).sum()

        model.eval()
        vloss = 0.0
        vcnt = torch.zeros(4, device=lr)
        vpl, vtl = [], []
        with torch.no_grad():
            for inp, tgt, w, lp_batch in vl:
                inp, tgt, w, lp_batch = (
                    inp.to(lr),
                    tgt.to(lr),
                    w.to(lr),
                    lp_batch.to(lr),
                )
                out = model(inp)
                vloss += criterion(out, tgt, w, lp_batch).item()
                pb = (out > 0.5).float()
                vcnt[0] += ((pb == 1) & (tgt == 1)).sum()
                vcnt[1] += ((pb == 1) & (tgt == 0)).sum()
                vcnt[2] += ((pb == 0) & (tgt == 1)).sum()
                vcnt[3] += ((pb == 0) & (tgt == 0)).sum()
                vpl.append(out.cpu())
                vtl.append(tgt.cpu())

        dist.all_reduce(tcnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(vcnt, op=dist.ReduceOp.SUM)
        vp_all, vt_all = gather_probs_targets(vpl, vtl, lr)

        cur_prauc = 0.0
        if gr == 0:
            tl_avg = run_loss / len(tl)
            vl_avg = vloss / len(vl)
            tp, tr, ta, tf1, tf2 = calculate_metrics(tcnt)
            vp, vr, va, vf1, vf2 = calculate_metrics(vcnt)
            cur_prauc = average_precision_score(vt_all, vp_all)
            clr = optimizer.param_groups[0]["lr"]
            print(
                f"\nEpoch {epoch + 1}/{cfg['max_epochs']}  (lr={clr:.2e}):"
                f"\n  Train  loss={tl_avg:.4f}  F2={tf2:.4f}  Prec={tp:.4f}  Rec={tr:.4f}"
                f"\n  Val    loss={vl_avg:.4f}  F2={vf2:.4f}  Prec={vp:.4f}  Rec={vr:.4f}"
                f"  PR-AUC={cur_prauc:.5f}"
            )
            if cur_prauc > best_prauc:
                print(f"  >> PR-AUC: {best_prauc:.5f} -> {cur_prauc:.5f}")
                best_prauc = cur_prauc
                best_state = copy.deepcopy(model.module.state_dict())
                counter = 0
            else:
                counter += 1
                print(f"   . No improvement ({counter}/{cfg['patience']})")
            if counter >= cfg["patience"]:
                print("  !! Early stopping.")
                stop = torch.tensor(1, device=lr)

        dist.broadcast(stop, src=0)
        if stop.item():
            break

    # Evaluation
    if gr == 0:
        print("\n" + "=" * 70 + "\nEVALUATION\n" + "=" * 70)
        if best_state:
            model.module.load_state_dict(best_state)
        ema_state = copy.deepcopy(ema.ema.state_dict())
        dev = torch.device(f"cuda:{lr}")

        def eval_model(m, tag):
            m.eval()
            preds, targets = [], []
            ds = NumpyDataset(X_test, y_test, w_test, lp_test)
            dl = DataLoader(ds, batch_size=8192, shuffle=False, pin_memory=True)
            with torch.no_grad():
                for xi, ti, _, _ in dl:
                    preds.append(m(xi.to(dev)).cpu().numpy())
                    targets.append(ti.numpy())
            yp = np.concatenate(preds).ravel()
            yt = np.concatenate(targets).ravel()
            prauc = average_precision_score(yt, yp)
            ra = roc_auc_score(yt, yp)
            print(f"\n  [{tag}]  ROC-AUC={ra:.4f}  PR-AUC={prauc:.4f}")
            return yp, yt, prauc

        yp_best, yt, prauc_best = eval_model(model.module, "Best checkpoint")
        ema_m = MuonSelectionDNN(
            input_dim, cfg["hidden_dim"], cfg["n_res_blocks"], cfg["dropout"]
        ).to(dev)
        ema_m.load_state_dict(ema_state)
        yp_ema, _, prauc_ema = eval_model(ema_m, "EMA model")

        if prauc_ema >= prauc_best:
            print("\n  -> Using EMA model")
            y_pred, chosen_state, chosen_tag = yp_ema, ema_state, "EMA"
        else:
            print("\n  -> Using best checkpoint")
            y_pred, chosen_state, chosen_tag = yp_best, best_state, "Checkpoint"
        y_true = yt

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc_val = auc(fpr, tpr)
        prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_pred)
        pr_auc_val = auc(rec_arr, prec_arr)
        print(f"\n  Test ROC AUC: {roc_auc_val:.4f}  |  Test PR AUC: {pr_auc_val:.4f}")

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
            plt.savefig(cfg["output_dir"] + name + ".png", dpi=300)
            plt.close()

        f1s = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-6)
        f2s = 5 * prec_arr * rec_arr / (4 * prec_arr + rec_arr + 1e-6)
        bi1 = np.argmax(f1s)
        bi2 = np.argmax(f2s)
        th1 = thresholds[bi1] if bi1 < len(thresholds) else 0.5
        th2 = thresholds[bi2] if bi2 < len(thresholds) else 0.5
        final_th = th2
        print(f"\n  Optimal threshold (F1): {th1:.4f}  F1={f1s[bi1]:.4f}")
        print(f"  Optimal threshold (F2): {th2:.4f}  F2={f2s[bi2]:.4f}")
        print(f"  Using F2 threshold: {final_th:.4f}")

        yb = (y_pred >= final_th).astype(int)
        print(f"\n{classification_report(y_true, yb, digits=4)}")
        cm = confusion_matrix(y_true, yb)
        print(cm)
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
        plt.title(f"Confusion matrix (threshold={final_th:.3f})")
        plt.tight_layout()
        plt.savefig(cfg["output_dir"] + "confusion_matrix.png", dpi=300)
        plt.close()

        # Per-pT-bin evaluation
        print("\n" + "=" * 70 + "\nPER-pT-BIN PERFORMANCE\n" + "=" * 70)
        pt_test_scaled = X_test[:, pt_feat_idx]
        pt_test_raw = 10 ** (
            pt_test_scaled * scaler.scale_[pt_feat_idx] + scaler.mean_[pt_feat_idx]
        )
        evaluate_pt_bins(y_true, y_pred, pt_test_raw, final_th, cfg["output_dir"])

        # Per-file evaluation
        print("\n" + "=" * 70 + "\nPER-FILE PERFORMANCE\n" + "=" * 70)
        for fi, fname in enumerate(files):
            fm = l_test == fi
            if fm.sum() == 0:
                continue
            yf, pf = y_true[fm], y_pred[fm]
            bf = (pf >= final_th).astype(int)
            a = roc_auc_score(yf, pf) if len(np.unique(yf)) > 1 else float("nan")
            c = confusion_matrix(yf, bf, labels=[0, 1]).ravel()
            tn, fp, fn, tp = c
            p = tp / (tp + fp + 1e-6)
            r = tp / (tp + fn + 1e-6)
            f2 = 5 * p * r / (4 * p + r + 1e-6)
            print(
                f"\n  {fname.split('/')[-1]:<30s}  AUC={a:.4f}  Prec={p:.4f}  Rec={r:.4f}  F2={f2:.4f}"
                f"\n    {'':30s}  [TN={tn} FP={fp} FN={fn} TP={tp}]"
            )

        # Feature importance
        print("\n" + "=" * 70 + "\nFEATURE IMPORTANCE\n" + "=" * 70)
        cpu_m = MuonSelectionDNN(
            input_dim, cfg["hidden_dim"], cfg["n_res_blocks"], cfg["dropout"]
        )
        cpu_m.load_state_dict(chosen_state)
        cpu_m.eval()

        print("\n[A] Full test set:")
        ia, sa, ba = compute_permutation_importance(
            cpu_m, X_test, y_true.astype(np.int32), feature_names
        )
        plot_importance(
            ia,
            sa,
            feature_names,
            "Feature importance - full test",
            cfg["output_dir"] + "feat_imp_all.png",
        )

        lpm = pt_test_raw < 5.0
        print(
            f"\n[B] Low-pT subset: {lpm.sum()} tracks ({y_true[lpm].sum():.0f} signal)"
        )
        if lpm.sum() > 1000 and len(np.unique(y_true[lpm])) == 2:
            il, sl, bl = compute_permutation_importance(
                cpu_m, X_test[lpm], y_true[lpm].astype(np.int32), feature_names
            )
            plot_importance(
                il,
                sl,
                feature_names,
                "Feature importance - low-pT",
                cfg["output_dir"] + "feat_imp_lowpt.png",
            )
            print("\n[C] Shift (low-pT vs global):")
            d = il - ia
            od = np.argsort(d)[::-1]
            for i in od:
                print(
                    f"    {'A' if d[i] > 0 else 'V'} {feature_names[i]:<50s}  D={d[i]:+.5f}"
                )

        imp_order = np.argsort(ia)[::-1]
        with open(cfg["output_dir"] + "feature_importance.txt", "w") as fo:
            fo.write(f"Baseline PR-AUC: {ba:.6f}\n")
            for r, fi in enumerate(imp_order):
                fo.write(
                    f"{r + 1:<5}{feature_names[fi]:<55}{ia[fi]:+.6f}  {sa[fi]:.6f}\n"
                )
        del cpu_m
        gc.collect()

        # ONNX export
        print("\n" + "=" * 70 + "\nONNX EXPORT\n" + "=" * 70)
        m_std = copy.deepcopy(
            MuonSelectionDNN(
                input_dim, cfg["hidden_dim"], cfg["n_res_blocks"], cfg["dropout"]
            )
        )
        m_std.load_state_dict(chosen_state)
        m_std.eval()
        scale_t = torch.from_numpy(scaler.scale_.astype(np.float32))
        mean_t = torch.from_numpy(scaler.mean_.astype(np.float32))
        with torch.no_grad():
            W = m_std.input_proj[0].weight.clone()
            b = m_std.input_proj[0].bias.clone()
            m_std.input_proj[0].weight.copy_(W / scale_t.unsqueeze(0))
            m_std.input_proj[0].bias.copy_(
                b - ((W / scale_t.unsqueeze(0)) * mean_t.unsqueeze(0)).sum(1)
            )

        m_fast = make_inference_model(chosen_state, input_dim, scaler, cfg)

        dummy = torch.randn(1, input_dim)
        orig = MuonSelectionDNN(
            input_dim, cfg["hidden_dim"], cfg["n_res_blocks"], cfg["dropout"]
        )
        orig.load_state_dict(chosen_state)
        orig.eval()
        with torch.no_grad():
            ref = orig((dummy - mean_t) / scale_t)
            o_std = m_std(dummy)
            o_fast = m_fast(dummy)
        print(f"  Standard vs ref: {(ref - o_std).abs().max().item():.2e}")
        print(f"  Fast vs ref:     {(ref - o_fast).abs().max().item():.2e}")

        for m, name, tag in [
            (m_std, "model_standard.onnx", "Standard"),
            (m_fast, "model_fast.onnx", "Fast (BN fused)"),
        ]:
            path = cfg["output_dir"] + name
            torch.onnx.export(
                m,
                dummy,
                path,
                export_params=True,
                opset_version=cfg["onnx_opset"],
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
            print(f"  {tag}: {path}  ({os.path.getsize(path) / 1024:.0f} KB)")

        print("\n  Inference benchmark (1000 x single-sample CPU):")
        for m, tag in [(m_std, "standard"), (m_fast, "fast")]:
            m.eval()
            single = torch.randn(1, input_dim)
            for _ in range(100):
                with torch.no_grad():
                    m(single)
            t0 = time.perf_counter()
            for _ in range(1000):
                with torch.no_grad():
                    m(single)
            dt = (time.perf_counter() - t0) / 1000
            print(f"    {tag:12s}: {dt * 1e6:.1f} us / sample")

        with open(cfg["output_dir"] + "thresholds.txt", "w") as fo:
            fo.write(f"F1_Threshold: {th1}\nF2_Threshold: {th2}\n")
            fo.write(f"Best_val_PRAUC: {best_prauc}\nTest_ROC_AUC: {roc_auc_val}\n")
            fo.write(f"Test_PR_AUC: {pr_auc_val}\nChosen_model: {chosen_tag}\n")
            fo.write(
                f"Signal_boost: {cfg['signal_boost']}\nLow_pT_loss_weight: {cfg['low_pt_loss_weight']}\n"
            )

    destroy_process_group()


if __name__ == "__main__":
    main()
