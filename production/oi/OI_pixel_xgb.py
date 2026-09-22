"""
OI_pixel_xgb.py - XGBoost forest for the OI high-purity selection of muon
(pixel) tracks in the pixel-selector HLT chain (l3_tk_OI tracks of the
tunedPixelSelector production ntuples).

Round 1: full 26-feature set; the feature-importance results drive the
drop_features list of the round-2 (pruned) re-training.

Training setup: identical to pixel_xgb.py / seeds_xgb.py (see
OI_xgb_common.py for the full v2 pipeline description).

Run:
    python OI_pixel_xgb.py
"""

import pixel_features as pf
from OI_xgb_common import CFG, main

# Deployed OI DNN for the comparison re-score (26-input model in the
# 20_1_X_muonTracking CMSSW branch, RecoMuon/L3TrackFinder/data/).
_OI_DNN = (
    "/shared/muons/phase2MuonTrackingPR/CMSSW_20_1_0_pre2/src"
    "/RecoMuon/L3TrackFinder/data/OI_pixel_model.onnx"
)

CFG.update(
    data_dir=pf.DATA_DIR,  # pixel-selector production ntuples
    output_dir="pixel/",
    cache_tag="pixel",
    dnn_onnx_path=_OI_DNN,
    ref_model_path=None,
    # Round 2: pruned feature set. Dropped per the round-1 gain ranking and
    # low-pT permutation importance (both marginal): ptErr, sigmaPtOverPt and
    # relUncertaintyProduct duplicate qoverpErr's pT-uncertainty information;
    # chi2 duplicates normalizedChi2/chi2PerHit.
    drop_features=[
        "l3_tk_OI_chi2",
        "l3_tk_OI_ptErr",
        "l3_tk_OI_sigmaPtOverPt",
        "l3_tk_OI_relUncertaintyProduct",
    ],
)

if __name__ == "__main__":
    main(CFG)
