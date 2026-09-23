import argparse
import copy
import gc
import os

import awkward as ak

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import uproot
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
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

# Input files
_ap = argparse.ArgumentParser()
_ap.add_argument("--data-dir", required=True, help="directory with the n-tuples of this early study (pixel-track gun samples)")
data_dir = _ap.parse_args().data_dir
files = []
for path in os.listdir(data_dir):
    if os.path.isfile(os.path.join(data_dir, path)):
        files.append(os.path.join(data_dir, path))

print(f"Selected {len(files)} input files:")
print(files)

main_branch = "Events"
tk_branches = [
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

l2_mu_vtx_branches = [
    "l2_mu_vtx_pt",
    "l2_mu_vtx_ptErr",
    "l2_mu_vtx_eta",
    "l2_mu_vtx_etaErr",
    "l2_mu_vtx_phi",
    "l2_mu_vtx_phiErr",
    "l2_mu_vtx_dz",
    "l2_mu_vtx_dzErr",
]

# Features
# Variables to log
log_features = [
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

# Variables to keep as-is (linear scale)
plain_features = [
    "l3_tk_OI_eta",
    "l3_tk_OI_nPixelHits",
    "l3_tk_OI_nTrkLays",
    "l3_tk_OI_nFoundHits",
    "l3_tk_OI_nLostHits",
]

# Variables to drop
drop_features = [
    "l3_tk_OI_phi",
    "l3_tk_OI_qoverp",
    "l3_tk_OI_dsz",
    "l3_tk_OI_dz",
    "l3_tk_OI_dxy",
]

LABEL_FIELD = "l3_tk_OI_matched"


# Helper Functions
def delta_phi(phi1, phi2):
    """Vectorized delta phi calculation handling periodicity"""
    dphi = phi1 - phi2
    # optimized wrapping: (dphi + pi) % 2pi - pi
    out = (dphi + np.pi) % (2 * np.pi) - np.pi
    return out


def impute_and_log(values_jagged, mask_jagged, fill_value=-1.0):
    """
    1. Flattens array.
    2. Imputes missing values (where mask is False) with 'fill_value'.
    3. Applies Log10.
    """
    flat_vals = ak.to_numpy(ak.flatten(values_jagged))
    flat_mask = ak.to_numpy(ak.flatten(mask_jagged))

    # Fill non-matches with fixed value
    flat_vals[~flat_mask] = fill_value

    # Log transform
    return np.log10(np.abs(flat_vals) + 1e-6).astype(np.float32)


def impute_linear(values_jagged, mask_jagged, fill_value=0.0):
    """
    Same as impute_and_log but without log transform.
    Useful for integer counts or bounded qualities.
    """
    flat_vals = ak.to_numpy(ak.flatten(values_jagged))
    flat_mask = ak.to_numpy(ak.flatten(mask_jagged))

    # Fill non-matches with fixed value
    flat_vals[~flat_mask] = fill_value
    return flat_vals.astype(np.float32)


def calculate_metrics(counts_tensor):
    """
    Computes Precision, Recall, Accuracy, F1, F2 from a counts tensor.
    counts_tensor: [TP, FP, FN, TN]
    """
    tp, fp, fn, tn = (
        counts_tensor[0],
        counts_tensor[1],
        counts_tensor[2],
        counts_tensor[3],
    )
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    f2 = 5 * (precision * recall) / (4 * precision + recall + 1e-6)
    return precision.item(), recall.item(), accuracy.item(), f1.item(), f2.item()


# Dataset creation
def build_dataset(arr, file_labels_in, useStandaloneFeatures=True, verbose=False):
    print("Building dataset with robust masking...")

    mask = arr["l3_tk_OI_pt"] > 0

    # Expand file labels
    n_tracks_per_event = ak.num(arr["l3_tk_OI_pt"])
    file_labels_jagged = ak.unflatten(
        np.repeat(file_labels_in, n_tracks_per_event), n_tracks_per_event
    )
    file_labels_masked = ak.to_numpy(ak.flatten(file_labels_jagged[mask]))

    cols = []
    final_feature_names = []

    trk_pt = arr["l3_tk_OI_pt"]

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

    # Impact Parameters
    trk_dxy = arr["l3_tk_OI_dxy"]
    trk_dz = arr["l3_tk_OI_dz"]
    trk_dxyErr = arr["l3_tk_OI_dxyErr"]
    trk_dzErr = arr["l3_tk_OI_dzErr"]

    # Impact Parameter 3D (log)
    ip3d = trk_dxy**2 + trk_dz**2
    cols.append(ak.to_numpy(ak.flatten(np.log10(ip3d + 1e-6)[mask])).astype(np.float32))
    final_feature_names.append("l3_tk_OI_impact3D")

    # Impact Significance
    sip2d = np.sqrt(
        (trk_dxy / np.maximum(trk_dxyErr, 1e-6)) ** 2
        + (trk_dz / np.maximum(trk_dzErr, 1e-6)) ** 2
    )
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sip2d + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("l3_tk_OI_impactSignificance")

    # Track Quality
    trk_chi2 = arr["l3_tk_OI_chi2"]
    trk_nFound = arr["l3_tk_OI_nFoundHits"]
    trk_nLost = arr["l3_tk_OI_nLostHits"]

    # Chi2 per hit (log)
    chi2_hit = trk_chi2 / np.maximum(trk_nFound, 1)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(chi2_hit + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("l3_tk_OI_chi2PerHit")

    # Hit Efficiency (linear) - found / (found + lost)
    hit_eff = trk_nFound / np.maximum(trk_nFound + trk_nLost, 1)
    cols.append(ak.to_numpy(ak.flatten(hit_eff[mask])).astype(np.float32))
    final_feature_names.append("l3_tk_OI_hitEfficiency")

    # Relative Uncertainties
    trk_ptErr = arr["l3_tk_OI_ptErr"]
    trk_qoverp = arr["l3_tk_OI_qoverp"]
    trk_qoverpErr = arr["l3_tk_OI_qoverpErr"]

    # SigmaPt / Pt (log)
    sigmaPtOverPt = trk_ptErr / np.maximum(trk_pt, 1e-6)
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(sigmaPtOverPt + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("l3_tk_OI_sigmaPtOverPt")

    # Relative Uncertainty Product (log) - (ptErr/pt) * (qoverpErr/qoverp)
    relUncertProd = (sigmaPtOverPt) * (
        trk_qoverpErr / np.maximum(np.abs(trk_qoverp), 1e-6)
    )
    cols.append(
        ak.to_numpy(ak.flatten(np.log10(relUncertProd + 1e-6)[mask])).astype(np.float32)
    )
    final_feature_names.append("l3_tk_OI_relUncertaintyProduct")

    # Standalone muon matching
    if useStandaloneFeatures:
        print("Computing Standalone Muon matching...")

        # Prepare Inputs with Shapes:
        # Tracks: (Event, nTracks, 1)
        # L1s:    (Event, 1, nL1)

        t_eta = arr["l3_tk_OI_eta"][:, :, np.newaxis]
        t_etaErr = arr["l3_tk_OI_eta"][:, :, np.newaxis]
        t_phi = arr["l3_tk_OI_phi"][:, :, np.newaxis]
        t_phiErr = arr["l3_tk_OI_phiErr"][:, :, np.newaxis]
        t_pt = arr["l3_tk_OI_pt"][:, :, np.newaxis]
        t_ptErr = arr["l3_tk_OI_ptErr"][:, :, np.newaxis]
        t_dz = arr["l3_tk_OI_dz"][:, :, np.newaxis]
        t_dzErr = arr["l3_tk_OI_dzErr"][:, :, np.newaxis]

        standalone_eta = arr["l2_mu_vtx_eta"][:, np.newaxis, :]
        standalone_etaErr = arr["l2_mu_vtx_etaErr"][:, np.newaxis, :]
        standalone_phi = arr["l2_mu_vtx_phi"][:, np.newaxis, :]
        standalone_phiErr = arr["l2_mu_vtx_phiErr"][:, np.newaxis, :]
        standalone_pt = arr["l2_mu_vtx_pt"][:, np.newaxis, :]
        standalone_ptErr = arr["l2_mu_vtx_ptErr"][:, np.newaxis, :]
        standalone_dz = arr["l2_mu_vtx_dz"][:, np.newaxis, :]
        standalone_dzErr = arr["l2_mu_vtx_dzErr"][:, np.newaxis, :]

        # Find Best Match
        chi2EtaMatrix = (t_eta - standalone_eta) ** 2 / (t_etaErr**2 + standalone_etaErr**2 + 1e-12)
        chi2PhiMatrix = (t_phi - standalone_phi) ** 2 / (t_phiErr**2 + standalone_phiErr**2 + 1e-12)
        chi2PtMatrix = (t_pt - standalone_pt) ** 2 / (t_ptErr**2 + standalone_ptErr**2 + 1e-12)
        chi2DzMatrix = (t_dz - standalone_dz) ** 2 / (t_dzErr**2 + standalone_dzErr**2 + 1e-12)

        maxChi2 = 9.0
        is_compatible = (chi2EtaMatrix < maxChi2) & (chi2PhiMatrix < maxChi2) & (chi2PtMatrix < maxChi2) & (chi2DzMatrix < maxChi2)

        qualityScoreMatrix = chi2EtaMatrix + chi2PhiMatrix + chi2PtMatrix + chi2DzMatrix
        qualityMatrixMasked = ak.mask(qualityScoreMatrix, is_compatible)

        min_quality = ak.min(qualityMatrixMasked, axis=2)

        # Broadcasting: (Event, nTracks, nL1) == (Event, nTracks, 1) -> (Event, nTracks, nL1)
        # Note: ak.fill_none handles cases where no Standalone exist
        min_vals_broad = ak.fill_none(min_quality[:, :, np.newaxis], -1.0)

        # The best match must equal the minimum score AND be compatible
        is_best_match = (qualityScoreMatrix == min_vals_broad) & is_compatible

        has_match = (min_quality < maxChi2) & (~ak.is_none(min_quality))

        # l2_mu_vtx_hasMatch
        cols.append(ak.to_numpy(ak.flatten(has_match[mask])).astype(np.float32))
        final_feature_names.append("l2_mu_vtx_hasMatch")

        # l2_mu_vtx_matchingScore (Imputed with 10.0)
        cols.append(
            impute_and_log(min_quality[mask], has_match[mask], fill_value=10.0)
        )
        final_feature_names.append("l2_mu_vtx_matchingScore")

    # --- Low pT Explicit Features ---
    # Add flag for very low pT tracks to help network switch regimes
    # Soft indicator (sigmoid) instead of hard binary to keep gradients smooth
    # Cutoff around 5 GeV.
    # Sigmoid function: 1 / (1 + exp(k * (pt - cutoff)))
    # For pt << 5, exp is small -> 1. For pt >> 5, exp is huge -> 0.

    flat_pt = ak.to_numpy(ak.flatten(trk_pt[mask])).astype(np.float32)

    # Clip exponent to avoid overflow in float32.
    exponent = (flat_pt - 5.0) * 2.0
    exponent = np.clip(exponent, -20.0, 20.0)

    low_pt_indicator = 1.0 / (1.0 + np.exp(exponent))

    cols.append(low_pt_indicator.astype(np.float32))
    final_feature_names.append("is_low_pt")

    X = np.column_stack(cols).astype(np.float32)
    y = ak.to_numpy(ak.flatten(arr[LABEL_FIELD][mask])).astype(np.int8)

    finite_mask = np.isfinite(X).all(axis=1)
    if not finite_mask.all():
        X = X[finite_mask]
        y = y[finite_mask]
        file_labels_masked = file_labels_masked[finite_mask]

    return X, y, file_labels_masked, final_feature_names


# Architecture & Loss
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class PixelTrackDNN(nn.Module):
    def __init__(self, input_dim=16):
        super(PixelTrackDNN, self).__init__()

        hidden_dim = 144

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.1),
        )

        # Residual blocks allow learning identity mapping easily
        # helping preservation of simple features while learning complex ones
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=0.1),
            ResidualBlock(hidden_dim, dropout=0.1),
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Initialize weights for better start
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_head(x)


class RecallLoss(nn.Module):
    """
    Differentiable soft coverage loss (1 - Recall).
    Pushes the model to maintain high True Positive Rate.
    Differs from standard Recall by handling sample weights.
    """

    def __init__(self):
        super(RecallLoss, self).__init__()

    def forward(self, probabilities, targets, weights=None):
        if probabilities.shape != targets.shape:
            targets = targets.view_as(probabilities)

        if weights is not None:
            if weights.shape != targets.shape:
                weights = weights.view_as(targets)

            # Weighted TP: P * y * w
            true_positives = (probabilities * targets * weights).sum()
            total_positives = (targets * weights).sum() + 1e-6
        else:
            true_positives = (probabilities * targets).sum()
            total_positives = targets.sum() + 1e-6

        recall = true_positives / total_positives
        return 1.0 - recall


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, inputs, targets, weights=None):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if weights is not None:
            focal_loss = focal_loss * weights

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, recall_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.recall = RecallLoss()
        self.recall_weight = recall_weight

    def forward(self, inputs, targets, weights=None):
        loss_focal = self.focal(inputs, targets, weights)
        loss_recall = self.recall(
            inputs, targets, weights
        )  # Pass weights to RecallLoss
        return loss_focal + (self.recall_weight * loss_recall)


class NumpyDataset(Dataset):
    def __init__(self, X, y, w=None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
        if w is None:
            self.w = torch.ones(self.y.shape, dtype=torch.float32)
        else:
            self.w = torch.from_numpy(w.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


# DDP Setup
def ddp_setup():
    """
    Initializes the distributed process group.
    torchrun sets environment variables: RANK, LOCAL_RANK, WORLD_SIZE
    """
    init_process_group(backend="nccl")  # NCCL is optimized for NVIDIA GPUs
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main():
    X_list = []
    y_list = []
    file_labels_list = []
    feature_names = []
    total_events = 0

    print(f"Processing {len(files)} files incrementally...")

    for i, f in enumerate(files):
        print(f"[{i + 1}/{len(files)}] Loading {f}...")
        with uproot.open(f) as file:
            arrays_f = file[main_branch].arrays(
                tk_branches + l2_mu_vtx_branches
            )
            n_events = len(arrays_f)
            total_events += n_events

            file_labels_temp = np.full(n_events, i)

            X_chunk, y_chunk, labels_chunk, feats = build_dataset(
                arrays_f, file_labels_temp
            )

            X_list.append(X_chunk)
            y_list.append(y_chunk)
            file_labels_list.append(labels_chunk)

            if i == 0:
                feature_names = feats

            # Free memory immediately
            del arrays_f
            del X_chunk, y_chunk, labels_chunk
            gc.collect()
            print(f"Loaded {n_events} events from {f}")

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    file_labels_flat = np.concatenate(file_labels_list)

    print(f"Loaded {total_events} events from {len(files)} files")
    print("Feature matrix shape:", X.shape)

    # Sample Weighting for Low pT
    print("Computing sample weights to boost Low pT performance...")
    # Default weights
    weights = np.ones(len(y), dtype=np.float32)

    # Try to find pT feature index
    if "l3_tk_OI_pt" in feature_names:
        pt_idx = feature_names.index("l3_tk_OI_pt")
        # Recover pT (approx, since it's log10(|pt|+1e-6))
        pt_values = 10 ** (X[:, pt_idx])

        # Boost weights for pT < 10 GeV
        # Symetric Weighting: Apply boost to component based on pT regardless of label

        # Calculate Base Kinematic Weight (High for low pT)
        kin_weights = 1.0 + np.maximum(0, (20.0 / (pt_values + 0.1)) - 1.0)
        kin_weights = np.clip(kin_weights, 1.0, 20.0).astype(np.float32)

        # Assign Weights
        # Signal (y=1): Get kinematic weight * small boost
        # Background (y=0): Get kinematic weight (so low pT fakes are penalized heavily)
        weights = kin_weights
        weights = np.where(y == 1, weights * 5.0, weights).astype(np.float32)

        # Normalize weights to maintain training stability
        weights = weights / weights.mean()

        print(
            f"Weights stats: Min={weights.min():.2f}, Max={weights.max():.2f}, Mean={weights.mean():.2f}"
        )
        print(
            f"Signal weights mean: {weights[y == 1].mean():.2f}, Background weights mean: {weights[y == 0].mean():.2f}"
        )
    else:
        print("Warning: 'l3_tk_OI_pt' not found. Using uniform weights.")

    print("Performing stratified split (Train/Val/Test)...")

    # Create stratification label to balance classes AND file sources
    stratify_label = y * len(files) + file_labels_flat

    # Split Train+Val vs Test (80/20 split logic)
    # 20% held out for final Testing
    (
        X_train_val,
        X_test,
        y_train_val,
        y_test,
        w_train_val,
        w_test,
        labels_train_val,
        labels_test,
    ) = train_test_split(
        X,
        y,
        weights,
        file_labels_flat,
        test_size=0.2,
        stratify=stratify_label,
        random_state=42,
    )

    # Split Train vs Val
    # Remaining 80% split into 20% Val and 80% Train
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_train_val,
        y_train_val,
        w_train_val,
        test_size=0.2,
        stratify=y_train_val,
        random_state=42,
    )

    del X, y, weights, X_train_val, y_train_val, w_train_val
    gc.collect()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Fit Standard Scaler (On Train only)
    print("Fitting StandardScaler on Training set...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # Apply to Val and Test
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    ddp_setup()

    # Get local rank (GPU ID on this node)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    train_dataset = NumpyDataset(X_train, y_train, w_train)
    val_dataset = NumpyDataset(X_val, y_val, w_val)

    # DistributedSampler ensures each GPU gets a distinct subset of the data
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2048,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=True,
    )

    # Initialize Model on specific GPU
    input_dim = X_train.shape[1]
    model = PixelTrackDNN(input_dim=input_dim).to(local_rank)
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])

    # Use Combined Loss (Focal + Recall) with increased recall weight
    # recall_weight=5.0 strongly forces validation of low-pT signal
    criterion = CombinedLoss(alpha=0.8, gamma=2.0, recall_weight=5.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler to reduce LR when plateauing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20
    )

    # Training Loop
    if global_rank == 0:
        print("Model architecture:")
        print(model)
        print(f"--- Starting Training on {os.environ['WORLD_SIZE']} GPUs ---")
        print(
            f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}"
        )

    # Early Stopping variables
    best_val_f2 = 0.0
    patience = 30
    counter = 0
    max_epochs = 200
    stop_training = torch.tensor(
        0, device=local_rank
    )  # Flag to sync stopping across GPUs

    # Placeholder for best model state (on CPU to save memory on GPU)
    best_model_state = None

    for epoch in range(max_epochs):
        # Set epoch for sampler so shuffling changes each epoch
        train_sampler.set_epoch(epoch)
        model.train()

        running_loss = 0.0
        # Counter: [TP, FP, FN, TN]
        train_counts = torch.zeros(4, device=local_rank)

        for inputs, targets, weights in train_loader:
            inputs = inputs.to(local_rank)
            targets = targets.to(local_rank)
            weights = weights.to(local_rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            # Pass weights to loss
            loss = criterion(outputs, targets, weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                preds = (outputs > 0.5).float()
                train_counts[0] += ((preds == 1) & (targets == 1)).sum()  # TP
                train_counts[1] += ((preds == 1) & (targets == 0)).sum()  # FP
                train_counts[2] += ((preds == 0) & (targets == 1)).sum()  # FN
                train_counts[3] += ((preds == 0) & (targets == 0)).sum()  # TN

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_counts = torch.zeros(4, device=local_rank)

        with torch.no_grad():
            for inputs, targets, weights in val_loader:
                inputs = inputs.to(local_rank)
                targets = targets.to(local_rank)
                weights = weights.to(local_rank)

                outputs = model(inputs)
                # Validation loss also weighted for consistency
                loss = criterion(outputs, targets, weights)
                val_loss_accum += loss.item()

                preds = (outputs > 0.5).float()
                val_counts[0] += ((preds == 1) & (targets == 1)).sum()
                val_counts[1] += ((preds == 1) & (targets == 0)).sum()
                val_counts[2] += ((preds == 0) & (targets == 1)).sum()
                val_counts[3] += ((preds == 0) & (targets == 0)).sum()

        # Aggregate metrics across GPUs
        dist.all_reduce(train_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_counts, op=dist.ReduceOp.SUM)

        # Calculate metrics on Rank 0
        current_val_f2 = 0.0

        if global_rank == 0:
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss_accum / len(val_loader)

            t_prec, t_rec, t_acc, t_f1, t_f2 = calculate_metrics(train_counts)
            v_prec, v_rec, v_acc, v_f1, v_f2 = calculate_metrics(val_counts)

            current_val_f2 = v_f2

            # Step the scheduler based on F2
            scheduler.step(current_val_f2)

            print(f"\nEpoch {epoch + 1}/{max_epochs}:")
            print(
                f"  Train Loss: {avg_train_loss:.4f} | F1: {t_f1:.4f} | F2: {t_f2:.4f} | Prec: {t_prec:.4f} | Rec: {t_rec:.4f}"
            )
            print(
                f"  Val   Loss: {avg_val_loss:.4f} | F1: {v_f1:.4f} | F2: {v_f2:.4f} | Prec: {v_prec:.4f} | Rec: {v_rec:.4f}"
            )

            # Check for improvement (Targeting F2 for high Recall preference)
            if current_val_f2 > best_val_f2:
                print(
                    f"  --> Improvement! (F2: {best_val_f2:.4f} -> {current_val_f2:.4f}) Saving checkpoint."
                )
                best_val_f2 = current_val_f2
                best_model_state = copy.deepcopy(model.module.state_dict())
                counter = 0
            else:
                counter += 1
                print(f"  --> No improvement. Patience: {counter}/{patience}")

            if counter >= patience:
                print("  --> Early stopping triggered.")
                stop_training = torch.tensor(1, device=local_rank)

        # Sync stopping decision to all ranks
        dist.broadcast(stop_training, src=0)

        if stop_training.item() == 1:
            break

    # Detailed Evaluation (Only on Rank 0)
    if global_rank == 0:
        print("\n--- Training Finished. Loading Best Model for Evaluation ---")
        # Load best weights into the model
        if best_model_state is not None:
            model.module.load_state_dict(best_model_state)

        print("--- Starting Detailed Evaluation on Test Set ---")

        # Test set dataset needs weights in constructor even if ignored by model
        test_dataset = NumpyDataset(X_test, y_test, w_test)
        test_loader = DataLoader(
            test_dataset, batch_size=8192, shuffle=False, pin_memory=True
        )

        # Unwrap DDP to get the original model
        raw_model = model.module
        raw_model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets, _ in test_loader:
                inputs = inputs.to(local_rank)
                outputs = raw_model(inputs)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())

        y_pred = np.concatenate(all_preds).ravel()
        y_true = np.concatenate(all_targets).ravel()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        print(f"Test ROC AUC: {roc_auc:.4f}")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig("OIPixelOutput/roc_curve.png")
        plt.close()

        # Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        print(f"Test PR AUC: {pr_auc:.4f}")

        plt.subplot(1, 2, 2)
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.3f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig("OIPixelOutput/metrics_curve.png")
        plt.close()

        # Threshold Optimization
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-6)

        best_f1_idx = np.argmax(f1_scores)
        best_f2_idx = np.argmax(f2_scores)

        best_thresh_f1 = (
            thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        )
        best_f1_val = f1_scores[best_f1_idx]

        best_thresh_f2 = (
            thresholds[best_f2_idx] if best_f2_idx < len(thresholds) else 0.5
        )
        best_f2_val = f2_scores[best_f2_idx]

        print("-" * 30)
        print(
            f"Optimal Threshold (F1): {best_thresh_f1:.4f} -> F1 Score: {best_f1_val:.4f}"
        )
        print(
            f"Optimal Threshold (F2): {best_thresh_f2:.4f} -> F2 Score: {best_f2_val:.4f}"
        )
        print("-" * 30)

        # Use F2 threshold for final evaluation to favor Recall
        final_thresh = best_thresh_f2
        print(
            f"Using F2-optimal threshold ({final_thresh:.4f}) for final evaluation to favor Recall"
        )

        # Compare with standard threshold
        print("\n--- Comparison with Standard 0.5 Threshold ---")
        y_pred_05 = (y_pred >= 0.5).astype(int)
        f1_05 = f1_score(y_true, y_pred_05)
        cm_05 = confusion_matrix(y_true, y_pred_05)
        tn, fp, fn, tp = cm_05.ravel()
        prec_05 = tp / (tp + fp + 1e-6)
        rec_05 = tp / (tp + fn + 1e-6)
        print(
            f"Metrics @ 0.5 :: F1: {f1_05:.4f} | Prec: {prec_05:.4f} | Rec: {rec_05:.4f}"
        )
        print("-" * 30)

        # Classification report at F2 optimal threshold
        print(f"\nClassification Report (Threshold = {final_thresh:.4f})")
        y_pred_bin = (y_pred >= final_thresh).astype(int)
        print(classification_report(y_true, y_pred_bin, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred_bin))

        # Per-File Performance Analysis
        print("\n" + "=" * 70)
        print("PER-FILE PERFORMANCE ON TEST SET")
        print("=" * 70)

        for file_idx, fname in enumerate(files):
            mask = labels_test == file_idx
            if mask.sum() == 0:
                continue

            y_file_true = y_true[mask]
            y_file_probs = y_pred[mask]
            y_file_pred_bin = (y_file_probs >= final_thresh).astype(int)

            # Safe metrics calculation
            if len(np.unique(y_file_true)) > 1:
                auc_file = roc_auc_score(y_file_true, y_file_probs)
            else:
                auc_file = float("nan")

            f1_file = f1_score(y_file_true, y_file_pred_bin, zero_division=0)

            # Confusion matrix elements
            cm = confusion_matrix(y_file_true, y_file_pred_bin, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            recall_file = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_file = tp / (tp + fp) if (tp + fp) > 0 else 0
            f2_file = (
                5
                * (precision_file * recall_file)
                / (4 * precision_file + recall_file + 1e-6)
            )

            print(f"\nFile: {fname.split('/')[-1]}")
            print(
                f"  Samples: {mask.sum()} (Pos: {y_file_true.sum()}, Neg: {(y_file_true == 0).sum()})"
            )
            print(f"  AUC: {auc_file:.4f}")
            print(f"  Precision: {precision_file:.4f}")
            print(f"  Recall: {recall_file:.4f}")
            print(f"  F1: {f1_file:.4f} (at global thresh {final_thresh:.3f})")
            print(f"  F2: {f2_file:.4f}")
            print(f"  Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

    # Export with Embedded Scaling (Only on Rank 0)
    if global_rank == 0:
        print("\n--- Exporting to ONNX with Embedded Scaling ---")

        # Create a copy of the model for export on CPU
        model_for_export = copy.deepcopy(raw_model).cpu()
        model_for_export.eval()

        # Extract Scaler parameters
        scale = torch.from_numpy(scaler.scale_.astype(np.float32))
        mean = torch.from_numpy(scaler.mean_.astype(np.float32))

        # Fuse Scaler into the first Linear layer
        # The first layer is now model_for_export.input_proj[0]
        first_layer = model_for_export.input_proj[0]

        with torch.no_grad():
            W = first_layer.weight.clone()  # [out_dim, in_dim]
            b = first_layer.bias.clone()  # [out_dim]

            # W_new = W / scale
            # b_new = b - (W_new * mean)
            W_prime = W / scale.unsqueeze(0)
            b_prime = b - (W_prime * mean.unsqueeze(0)).sum(dim=1)

            first_layer.weight.copy_(W_prime)
            first_layer.bias.copy_(b_prime)

        print("Scaler fused into first layer weights/biases.")

        # Verify
        dummy_input = torch.randn(1, input_dim)
        with torch.no_grad():
            # Original model expects: (x - mean) / scale
            scaled_input = (dummy_input - mean) / scale
            out_orig = raw_model.cpu()(scaled_input)

            # Fused model path (takes raw input)
            out_fused = model_for_export(dummy_input)

            # They should be identical
            diff = torch.abs(out_orig - out_fused).max().item()
            print(f"Fused model check - Logits mean: {out_fused.mean().item():.4f}")
            # Should be ~1e-6
            print(f"Fused model check - Max difference: {diff:.6f}")

        # Export
        torch.onnx.export(
            model_for_export,
            dummy_input,
            "OIPixelOutput/model.onnx",
            export_params=True,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print("Saved: model.onnx (expects RAW unscaled inputs)")

        # Save threshold info
        with open("OIPixelOutput/thresholds.txt", "w") as f:
            f.write(f"F1_Threshold: {best_thresh_f1}\n")
            f.write(f"F2_Threshold: {best_thresh_f2}\n")
            f.write("Recommended: F2_Threshold for high efficiency triggers.\n")

    destroy_process_group()


if __name__ == "__main__":
    main()

