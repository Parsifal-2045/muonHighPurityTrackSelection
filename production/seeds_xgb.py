"""
seeds_xgb.py - HP forest for the muon IO tracks built from LST seeds
(hltPhase2MuonIOTracks) of the seeds IO chain (process modifier
phase2MuonSeedsSelector).

Configuration only; the training pipeline is forest_pipeline.py and the
feature extraction is the IO one of pixel_features.py on the
muon_general_tracks_* branches (same 33-feature ABI as the pixel selector).
Deployed in CMSSW through MuonIOTracksForestSelector
(cfi: hltPhase2MuonIOTrackSelectionHighPurityForest_cfi.py).

Run:
    python seeds_xgb.py --data-dir DIR [--cache-dir DIR] [--output-dir DIR] [--force] [--set key=value ...]
"""

import os

import forest_pipeline as fp
import pixel_features as pf

HERE = os.path.dirname(os.path.abspath(__file__))

CFG = dict(
    name="IO seeds",
    data_chain="seeds",  # --data-dir: seeds chain training n-tuples
    output_dir=os.path.join(HERE, "io/seeds/"),
    cache_tag="io_seeds",
    branches=pf.tk_branches(pf.SEEDS_PREFIX) + pf.L1TKMUON_BRANCHES + pf.STUB_BRANCHES,
    build=pf.build_dataset,
    build_kwargs=dict(prefix=pf.SEEDS_PREFIX, useL1TkMuFeatures=True, useL1TkMuStubFeatures=True),
    pt_feature=f"{pf.SEEDS_PREFIX}_pt",
    # Same 44 -> 33 pruning as the pixel selector (shared C++ extractor).
    drop_features=pf.io_drop_features(pf.SEEDS_PREFIX),
    production_features=pf.io_production_features(pf.SEEDS_PREFIX),
    ref_model_path=os.path.join(HERE, "io/archive/seeds_xgb_output_33f_{ref_version}/model.json"),  # previous production forest
    cmssw_bin="RecoMuon/L3TrackFinder/data/IO/muonHP_IO_seedsPath_forest_{version}.bin",
    cmssw_cfi="HLTrigger/Configuration/python/HLT_75e33/modules/hltPhase2MuonIOTrackSelectionHighPurityForest_cfi.py",
)

if __name__ == "__main__":
    fp.main_cli(CFG)
