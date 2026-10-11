"""
OI_xgb_common.py - common configuration of the OI (outside-in) high-purity
forests (pixel-chain and seeds-chain variants).

The OI forests run the shared pipeline (../forest_pipeline.py) on the
C++-faithful OI extraction (OI_features.py). The two flavours differ only in
the input sample: the l3_tk_OI track content follows the HLT chain of the
production (the OI step is seeded by the L2 muons the IO step did not use):
  - OI_pixel_xgb.py    : pixel-selector production n-tuples
  - OI_general_xgb.py  : seeds-selector production n-tuples
Both are deployed through MuonOITracksForestSelector
(cfi: hltPhase2L3OIMuonTrackSelectionHighPurityForest_cfi.py).
"""

import os
import sys

# forest_pipeline and pixel_features live in the production/ root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import forest_pipeline as fp  # noqa: E402
import OI_features as oif  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

OI_BASE = dict(
    branches=oif.TK_BRANCHES + oif.L2_MU_VTX_BRANCHES,
    build=oif.build_dataset,
    build_kwargs=dict(useStandaloneFeatures=True),
    pt_feature="l3_tk_OI_pt",
    # Round-2 pruning 26 -> 22 (round-1 gain + low-pT permutation importance,
    # production/oi/*_26f): ptErr, sigmaPtOverPt, relUncertaintyProduct
    # duplicate qoverpErr's pT-uncertainty information; chi2 duplicates
    # normalizedChi2/chi2PerHit. The 26-feature round-1 setup is
    # --set drop_features=[] --set production_features=null.
    copy_group_features=["l3_tk_OI_pt", "l3_tk_OI_eta", "l3_tk_OI_nPixelHits"],
    drop_features=oif.OI_DROP_FEATURES,
    production_features=oif.OI_PRODUCTION_FEATURES,
    cmssw_cfi="HLTrigger/Configuration/python/HLT_75e33/modules/hltPhase2L3OIMuonTrackSelectionHighPurityForest_cfi.py",
    # Hyperparameters: the pipeline defaults (shallow forests, copy-aware split).
    # Working points: the recall targets of the IO forests (98% below 3 GeV, 99% up to 5 GeV, 99.5% above).
    wp_recall_targets=[0.98, 0.99, 0.995, 0.995, 0.995, 0.995],
)


def oi_config(**kw):
    return dict(OI_BASE, **kw)


main_cli = fp.main_cli
