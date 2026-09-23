"""
OI_pixel_xgb.py - HP forest for the OI muon tracks
(hltPhase2L3OIMuCtfWithMaterialTracks) of the pixel-track chain
(phase2MuonPixelTracksSelector / ngtScouting), trained on the l3_tk_OI tracks
of the pixel-selector production n-tuples.

Run (from any directory):
    python OI_pixel_xgb.py --data-dir DIR [--cache-dir DIR] [--output-dir DIR] [--force] [--set key=value ...]
"""

import os

from OI_xgb_common import HERE, main_cli, oi_config  # sets up the production/ import path


CFG = oi_config(
    name="OI pixel",
    data_chain="pixel",  # --data-dir: pixel-track chain training n-tuples
    output_dir=os.path.join(HERE, "pixel/"),
    ref_model_path=os.path.join(HERE, "archive/pixel_22f_{ref_version}/model.json"),  # previous production forest
    cache_tag="oi_pixel",
    cmssw_bin="RecoMuon/L3TrackFinder/data/OI/muonHP_OI_pixelPath_forest_{version}.bin",
)

if __name__ == "__main__":
    main_cli(CFG)
