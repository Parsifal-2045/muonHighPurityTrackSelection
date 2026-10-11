"""
seeds_xgb.py - HP forest for the muon IO tracks built from LST seeds
(hltPhase2MuonIOTracks) of the seeds IO chain (process modifier
phase2MuonSeedsSelector).

Configuration only: features, recipe and working points are in
io_xgb_common.py (the same feature ABI as the pixel selector, on the
muon_general_tracks_* branches), the training pipeline in forest_pipeline.py.
Deployed in CMSSW through MuonIOTracksForestSelector
(cfi: hltPhase2MuonIOTrackSelectionHighPurityForest_cfi.py).

Run:
    python seeds_xgb.py --data-dir DIR [--cache-dir DIR] [--output-dir DIR] [--force] [--set key=value ...]
"""

import os

import forest_pipeline as fp
import io_xgb_common

HERE = os.path.dirname(os.path.abspath(__file__))

CFG = dict(
    io_xgb_common.io_config("seeds"),
    name="IO seeds",
    data_chain="seeds",  # --data-dir: seeds chain training n-tuples
    output_dir=os.path.join(HERE, "io/seeds/"),
    cache_tag="io_seeds",
    cmssw_bin="RecoMuon/L3TrackFinder/data/IO/muonHP_IO_seedsPath_forest_{version}.bin",
    cmssw_cfi="HLTrigger/Configuration/python/HLT_75e33/modules/hltPhase2MuonIOTrackSelectionHighPurityForest_cfi.py",
)

if __name__ == "__main__":
    fp.main_cli(CFG)
