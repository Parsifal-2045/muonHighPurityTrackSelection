"""
l3_muon_tables_cff.py - cmsRun customisation adding to the training NANO the
tracks of the L3 muons before and after the muon identification, with their
TrackingParticle match, for count_l3_muons.py (fake muons and muon efficiency
at the end of the chain).

Tables (same columns as the training track tables): l3_tk_IO (the IO tracks,
hltPhase2MuonIOTracks), l3_tk_merged (IO + OI tracks, hltPhase2L3MuonMerged),
l3_mu_no_ID (tracks of hltPhase2L3MuonsNoID) and l3_mu_ID (tracks of
hltPhase2L3Muons). The job must run the HLT validation (it defines the track
collections of the muons and their associations).

With pass_through=True the forest selectors keep every track: the chain
without a high-purity selection.

Usage (cmsDriver ... -s L1P2GT,HLT:75e33,NANO:@MUHLTTraining,VALIDATION:@hltValidation --customise_commands):
    import sys; sys.path.insert(0, '<dir>'); from l3_muon_tables_cff import customise_l3_muon_tables
    process = customise_l3_muon_tables(process, pass_through=False)
"""

import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.trackingAssocValueMapsProducer_cfi import trackingAssocValueMapsProducer

TABLES = (  # table name, track collection, association, producer of the track collection (None: made by the HLT)
    ("l3_tk_IO", "hltPhase2MuonIOTracks", "Phase2tpToL3IOTkNoHPAssociation", None),
    ("l3_tk_merged", "hltPhase2L3MuonMerged", "Phase2tpToL3TkMergedAssociation", None),
    ("l3_mu_no_ID", "hltPhase2L3MuonNoIdTracks", "Phase2tpToL3MuonNoIdAssociation", "hltPhase2L3MuonNoIdTracks"),
    ("l3_mu_ID", "hltPhase2L3MuonIdTracks", "Phase2tpToL3MuonIdAssociation", "hltPhase2L3MuonIdTracks"),
)


def customise_l3_muon_tables(process, pass_through=False):
    if pass_through:
        for label, module in process.producers_().items():
            if module.type_() in ("MuonIOTracksForestSelector", "MuonOITracksForestSelector"):
                module.decisionThreshold = -1.0
                module.decisionThresholds = [-1.0] * len(module.decisionThresholds)
    # the training track table of the chain, as template
    pixel = hasattr(process, "muonPixelTracksTableTraining") and process.nanoAOD_step.contains(process.muonPixelTracksTableTraining)
    template = process.muonPixelTracksTableTraining if pixel else process.muonGeneralTracksTableTraining
    sequence = cms.Sequence()
    for name, tracks, association, producer in TABLES:
        label = name.replace("_", "")
        values = trackingAssocValueMapsProducer.clone(
            trackCollection=cms.InputTag(tracks), associator=cms.InputTag(association),
            trackingParticles=cms.InputTag("mix", "MergedTrackTruth"), storeTPKinematics=cms.bool(True),
            useMuonAssociators=cms.bool(True))
        table = template.clone(src=cms.InputTag(tracks), name=cms.string(name), doc=cms.string(name))
        for par in table.externalVariables.parameterNames_():
            getattr(table.externalVariables, par).src = cms.InputTag(label + "V", par)
        setattr(process, label + "V", values)
        setattr(process, label + "L3Table", table)
        if producer:
            sequence += getattr(process, producer)
        sequence += getattr(process, association) + values + table
    process.l3MuonTables = sequence
    process.nanoAOD_step += process.l3MuonTables
    return process
