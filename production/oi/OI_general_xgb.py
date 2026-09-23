"""
OI_general_xgb.py - HP forest for the OI muon tracks
(hltPhase2L3OIMuCtfWithMaterialTracks) of the seeds (LST/mkFit) chain
(phase2MuonSeedsSelector), trained on the l3_tk_OI tracks of the
seeds-selector production n-tuples.

Run (from any directory):
    python OI_general_xgb.py --data-dir DIR [--cache-dir DIR] [--output-dir DIR] [--force] [--set key=value ...]
"""

import os

from OI_xgb_common import HERE, main_cli, oi_config  # sets up the production/ import path

CFG = oi_config(
    name="OI general",
    data_chain="seeds",  # --data-dir: seeds chain training n-tuples
    output_dir=os.path.join(HERE, "general/"),
    ref_model_path=os.path.join(HERE, "archive/general_22f_{ref_version}/model.json"),  # previous production forest
    cache_tag="oi_general",
    cmssw_bin="RecoMuon/L3TrackFinder/data/OI/muonHP_OI_seedsPath_forest_{version}.bin",
)

if __name__ == "__main__":
    main_cli(CFG)
