"""
pixel_xgb.py - HP forest for the muon pixel tracks (hltPhase2MuonPixelTracks)
of the pixel-track IO chain (process modifier phase2MuonPixelTracksSelector).

Configuration only: features, recipe and working points are in
io_xgb_common.py, the training pipeline in forest_pipeline.py. Deployed in
CMSSW through MuonIOTracksForestSelector
(cfi: hltPhase2MuonPixelTracksHighPurityForest_cfi.py).

Run:
    python pixel_xgb.py --data-dir DIR [--cache-dir DIR] [--output-dir DIR] [--force] [--set key=value ...]
"""

import os

import forest_pipeline as fp
import io_xgb_common

HERE = os.path.dirname(os.path.abspath(__file__))

CFG = dict(
    io_xgb_common.io_config("pixel"),
    name="IO pixel",
    data_chain="pixel",  # --data-dir: pixel-track chain training n-tuples
    output_dir=os.path.join(HERE, "io/pixel/"),
    cache_tag="io_pixel",
    cmssw_bin="RecoMuon/L3TrackFinder/data/IO/muonHP_IO_pixelPath_forest_{version}.bin",
    cmssw_cfi="HLTrigger/Configuration/python/HLT_75e33/modules/hltPhase2MuonPixelTracksHighPurityForest_cfi.py",
)

if __name__ == "__main__":
    fp.main_cli(CFG)
