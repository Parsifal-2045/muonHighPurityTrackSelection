"""
forest_timing_cff.py - cmsRun customisation timing the deployed forest
selectors of one HLT chain against alternative models on the same events.

For the chain's IO and OI selector it adds unscheduled clones (no feature
dump) with the deployed model and with each alternative model, consumed by an
EndPath, and switches the TimeReport on: the per-module "per event" times of
the clones compare the models on identical inputs. Alternative .bin files must
be in the CMSSW search path (e.g. $CMSSW_BASE/external/<arch>/data/...).

Usage (cmsDriver ... --customise_commands):
    import sys; sys.path.insert(0, '<dir>'); from forest_timing_cff import customise_timing
    process = customise_timing(process, 'pixel', {'v2': {'IO': '<rel path>', 'OI': '<rel path>'}})
"""

import FWCore.ParameterSet.Config as cms

SELECTORS = {
    "pixel": {"IO": "hltPhase2MuonPixelTracksHighPurityForest", "OI": "hltPhase2L3OIMuonTrackSelectionHighPurity"},
    "seeds": {"IO": "hltPhase2MuonIOTrackSelectionHighPurityForest", "OI": "hltPhase2L3OIMuonTrackSelectionHighPurity"},
}


def customise_timing(process, chain, alternatives):
    labels = []
    for family, label in SELECTORS[chain].items():
        base = getattr(process, label)
        variants = {"deployed": base.modelPath.value()}
        variants.update({tag: paths[family] for tag, paths in alternatives.items()})
        for tag, path in variants.items():
            clone = base.clone(modelPath=cms.FileInPath(path), dumpFeatures=False)
            name = f"timing{family}{tag.capitalize()}"
            setattr(process, name, clone)
            labels.append(name)
    process.timingConsumer = cms.EDAnalyzer("GenericConsumer", eventProducts=cms.untracked.vstring(*labels))
    process.timingTask = cms.Task(*[getattr(process, n) for n in labels])
    process.timingEndPath = cms.EndPath(process.timingConsumer, process.timingTask)
    process.schedule.append(process.timingEndPath)
    process.options.wantSummary = True
    return process
