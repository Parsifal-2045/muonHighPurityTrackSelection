"""
pixel_xgb.py - HP forest for the muon pixel tracks (hltPhase2MuonPixelTracks)
of the pixel-track IO chain (process modifier phase2MuonPixelTracksSelector).

Configuration only; the training pipeline (evt10 split, validation-driven
forest size and per-pT-bin working points, verified exports) is
forest_pipeline.py. Deployed in CMSSW through MuonIOTracksForestSelector
(cfi: hltPhase2MuonPixelTracksHighPurityForest_cfi.py).

Run:
    python pixel_xgb.py --data-dir DIR [--cache-dir DIR] [--output-dir DIR] [--force] [--set key=value ...]
"""

import os

import forest_pipeline as fp
import pixel_features as pf

HERE = os.path.dirname(os.path.abspath(__file__))

CFG = dict(
    name="IO pixel",
    data_chain="pixel",  # --data-dir: pixel-track chain training n-tuples
    output_dir=os.path.join(HERE, "io/pixel/"),
    cache_tag="io_pixel",
    branches=pf.tk_branches(pf.PIXEL_PREFIX) + pf.L1TKMUON_BRANCHES + pf.STUB_BRANCHES,
    build=pf.build_dataset,
    build_kwargs=dict(prefix=pf.PIXEL_PREFIX, useL1TkMuFeatures=True, useL1TkMuStubFeatures=True),
    pt_feature=f"{pf.PIXEL_PREFIX}_pt",
    # 44 -> 33 pruning (gain + permutation importance of the 44-feature
    # model): zero gain (nLostHits, hitEfficiency), near-zero gain
    # (normalizedChi2, chi2PerHit, chi2, impactSignificance, dxyErr, dszErr),
    # redundant (eta ~ absEta; ptErr, relUncertaintyProduct ~ qoverpErr).
    drop_features=pf.io_drop_features(pf.PIXEL_PREFIX),
    production_features=pf.io_production_features(pf.PIXEL_PREFIX),
    ref_model_path=os.path.join(HERE, "io/archive/pixel_xgb_output_33f_{ref_version}/model.json"),  # previous production forest
    cmssw_bin="RecoMuon/L3TrackFinder/data/IO/muonHP_IO_pixelPath_forest_{version}.bin",
    cmssw_cfi="HLTrigger/Configuration/python/HLT_75e33/modules/hltPhase2MuonPixelTracksHighPurityForest_cfi.py",
)

if __name__ == "__main__":
    fp.main_cli(CFG)
