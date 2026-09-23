"""
input_selector_study_cff.py - cmsRun customisation measuring what the
L1TkMuon-based input selectors of the HP forests keep and lose.

Adds to the NANO:@MUHLTTraining output MC-truth tables (MuonAssociatorByHits,
same association as the training labels) for
  pixel chain (customise_pixel):
    all_pixel_tracks           hltPhase2PixelTracks, the selector input
    muon_pixel_tracks_keep2/3  hltPhase2MuonPixelTracks with nTracksToKeep = 2/3
  seeds chain (customise_seeds):
    all_lst_seeds              hltInitialStepTrajectorySeedsLST (as tracks)
    muon_seeds                 hltPhase2MuonIOTrackSeeds (as tracks)
    muon_seeds_keep3/4         hltPhase2MuonIOTrackSeeds with nSeedsToKeep = 3/4
The default selections are already in the training n-tuple (muon_pixel_tracks,
muon_general_tracks). Analysis: analyze_input_selectors.py.

Usage (cmsDriver ... NANO:@MUHLTTraining ...):
    --customise_commands "exec(open('<path>/input_selector_study_cff.py').read()); process = customise_pixel(process)"
"""

import FWCore.ParameterSet.Config as cms


def _truth_table(process, tracks, name, label, doc):
    """Associator + value maps + flat table for a track collection, cloned
    from the training table of the muon pixel tracks."""
    assoc = process.Phase2tpToMuonPixelTracksAssociation.clone(tracksTag=tracks)
    vmaps = process.muonPixelTracksV.clone(trackCollection=tracks, associator=f"{label}Assoc")
    ext = cms.PSet()
    ref = process.muonPixelTracksTableTraining.externalVariables
    for var in ref.parameterNames_():
        setattr(ext, var, getattr(ref, var).clone(src=cms.InputTag(f"{label}V", var)))
    table = process.muonPixelTracksTableTraining.clone(src=tracks, name=name, doc=doc, externalVariables=ext)
    setattr(process, f"{label}Assoc", assoc)
    setattr(process, f"{label}V", vmaps)
    setattr(process, f"{label}Table", table)
    return [assoc, vmaps, table]


def _seed_tracks(process, seeds, label):
    mod = cms.EDProducer(
        "TrackFromSeedProducer",
        src=cms.InputTag(seeds),
        beamSpot=cms.InputTag("hltOnlineBeamSpot"),
        TTRHBuilder=cms.string("hltESPTTRHBuilderWithoutRefit"),
    )
    setattr(process, label, mod)
    return mod


def _attach(process, modules):
    process.inputSelectorStudyTask = cms.Task(*modules)
    process.nanoAOD_step.associate(process.inputSelectorStudyTask)
    return process


def customise_pixel(process):
    mods = []
    for n in (2, 3):
        sel = process.hltPhase2MuonPixelTracks.clone(nTracksToKeep=n)
        setattr(process, f"hltPhase2MuonPixelTracksKeep{n}", sel)
        mods.append(sel)
        mods += _truth_table(process, f"hltPhase2MuonPixelTracksKeep{n}", f"muon_pixel_tracks_keep{n}",
                             f"studyPixelKeep{n}", f"hltPhase2MuonPixelTracks with nTracksToKeep={n}")
    mods += _truth_table(process, "hltPhase2PixelTracks", "all_pixel_tracks", "studyAllPixel",
                         "hltPhase2PixelTracks (input of the L1TkMuon pixel-track selector)")
    return _attach(process, mods)


def customise_seeds(process):
    mods = [_seed_tracks(process, "hltInitialStepTrajectorySeedsLST", "studyAllLSTSeedTracks"),
            _seed_tracks(process, "hltPhase2MuonIOTrackSeeds", "studyMuonSeedTracks")]
    mods += _truth_table(process, "studyAllLSTSeedTracks", "all_lst_seeds", "studyAllLST",
                         "hltInitialStepTrajectorySeedsLST as tracks (input of the L1TkMuon seed selector)")
    mods += _truth_table(process, "studyMuonSeedTracks", "muon_seeds", "studyMuonSeeds",
                         "hltPhase2MuonIOTrackSeeds as tracks")
    for n in (3, 4):
        sel = process.hltPhase2MuonIOTrackSeeds.clone(nSeedsToKeep=n)
        setattr(process, f"hltPhase2MuonIOTrackSeedsKeep{n}", sel)
        mods += [sel, _seed_tracks(process, f"hltPhase2MuonIOTrackSeedsKeep{n}", f"studyMuonSeedTracksKeep{n}")]
        mods += _truth_table(process, f"studyMuonSeedTracksKeep{n}", f"muon_seeds_keep{n}", f"studySeedsKeep{n}",
                             f"hltPhase2MuonIOTrackSeeds with nSeedsToKeep={n}, as tracks")
    return _attach(process, mods)
