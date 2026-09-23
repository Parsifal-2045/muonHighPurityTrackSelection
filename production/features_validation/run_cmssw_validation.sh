#!/bin/bash
# run_cmssw_validation.sh - end-to-end CMSSW validation of the deployed HP forests.
#
# For both HLT chains (pixel: phase2MuonPixelTracksSelector, seeds:
# phase2MuonSeedsSelector) and two samples (ZMM, BsToMuMu at 200 PU) runs
# L1P2GT + HLT:75e33 + NANO:@MUHLTTraining (+ HLT validation, which carries the
# input customisations these RelVals need) with
#   * dumpFeatures on the chain's IO and OI forest selectors
#     -> features_validation/validate_cmssw.py (CMSSW vs Python, track by track);
#   * the input-selector truth tables (input_selectors/input_selector_study_cff.py)
#     -> input_selectors/analyze_input_selectors.py (end-to-end muon efficiency);
#   * timing clones of the selectors with alternative models (forest_timing_cff.py)
#     -> TimeReport in the job log.
#
# Usage: run_cmssw_validation.sh --cmssw-src DIR --out DIR [--events N] [--threads N]
#                                [--relval-zmm DIR] [--relval-bs DIR] [--alt-models DIR]
#   --cmssw-src   src/ of the CMSSW area with the deployed selectors
#   --out         output directory (job configurations, logs, NANO and DQM files)
#   --events      events per sample and chain (default 250)
#   --threads     threads per cmsRun job (default 12; the four jobs run in parallel)
#   --relval-zmm, --relval-bs
#                 directories of GEN-SIM-DIGI-RAW RelVal files (default: the official
#                 CMSSW_20_0_0_pre1 PU200 ZMM / BsToMuMu RelVals used for the v3 validation)
#   --alt-models  search-path directory (relative to a CMSSW_SEARCH_PATH entry) holding
#                 IO_pixelPath_<tag>.bin, IO_seedsPath_<tag>.bin, OI_pixelPath_<tag>.bin,
#                 OI_seedsPath_<tag>.bin for tag = the directory name (e.g. MuonHPTiming/v2),
#                 timed next to the deployed models; omit to time the deployed models only.
set -e
usage() { sed -n '/^# Usage/,/^set -e/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }
SRC=; OUT=; NEV=250; NTH=12; ALTDIR=
ZP=/eos/cms/store/relval/CMSSW_20_0_0_pre1/RelValZMM_14/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU-v1/2590000
BP=/eos/cms/store/relval/CMSSW_20_0_0_pre1/RelValBsToMuMu_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_BPH81_D121_RegeneratedGS_PU_20260420_144751-v1/2590000
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage ;;
    --cmssw-src|--out|--events|--threads|--relval-zmm|--relval-bs|--alt-models)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; usage 1; } ;;&
    --cmssw-src) SRC=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --events) NEV=$2; shift 2 ;;
    --threads) NTH=$2; shift 2 ;;
    --relval-zmm) ZP=$2; shift 2 ;;
    --relval-bs) BP=$2; shift 2 ;;
    --alt-models) ALTDIR=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; usage 1 ;;
  esac
done
[ -n "$SRC" ] && [ -n "$OUT" ] || { echo "--cmssw-src and --out are required" >&2; usage 1; }
SRC=$(readlink -f "$SRC"); OUT=$(readlink -f "$OUT")
PROD=$(dirname $(dirname $(readlink -f "$0")))
mkdir -p "$OUT"
cd "$SRC" && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval $(scram runtime -sh) 2>/dev/null
cd "$OUT"
ZF=$(ls $ZP | sed -n '2,4p' | sed "s#^#file:$ZP/#" | paste -sd,)
BF=$(ls $BP | sed -n '1,3p' | sed "s#^#file:$BP/#" | paste -sd,)
LOGCFG="process.MessageLogger.cerr.noLineBreaks = True; process.MessageLogger.cerr.MuonIOTracksForestSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1)); process.MessageLogger.cerr.MuonOITracksForestSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1))"
for s in ZMM Bs; do
  [ $s = ZMM ] && IN=$ZF || IN=$BF
  for chain in pixel seeds; do
    if [ $chain = pixel ]; then MOD=phase2MuonPixelTracksSelector; IO=hltPhase2MuonPixelTracksHighPurityForest; P=pixelPath
    else MOD=phase2MuonSeedsSelector; IO=hltPhase2MuonIOTrackSelectionHighPurityForest; P=seedsPath; fi
    ALT="{}"
    if [ -n "$ALTDIR" ]; then TAG=$(basename $ALTDIR); ALT="{'$TAG': {'IO': '$ALTDIR/IO_${P}_$TAG.bin', 'OI': '$ALTDIR/OI_${P}_$TAG.bin'}}"; fi
    cmsDriver.py step2 -s L1P2GT,HLT:75e33,NANO:@MUHLTTraining,VALIDATION:@hltValidation \
      --conditions auto:phase2_realistic_T35 --datatier NANOAODSIM,DQMIO --eventcontent NANOAODSIM,DQMIO \
      --geometry ExtendedRun4D121 --era Phase2C22I13M9 --process HLTX --procModifiers $MOD \
      "--inputCommands=keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT" \
      --filein $IN -n $NEV --nThreads $NTH --no_exec --python_filename final_${chain}_${s}_cfg.py --fileout file:final_${chain}_${s}.root \
      --customise_commands "import sys; sys.path.insert(0, '$PROD/input_selectors'); sys.path.insert(0, '$PROD/features_validation'); from input_selector_study_cff import customise_$chain; from forest_timing_cff import customise_timing; process = customise_$chain(process); process = customise_timing(process, '$chain', $ALT); process.$IO.dumpFeatures = True; process.hltPhase2L3OIMuonTrackSelectionHighPurity.dumpFeatures = True; $LOGCFG" \
      > driver_${chain}_${s}.log 2>&1
    nohup cmsRun final_${chain}_${s}_cfg.py > final_${chain}_${s}.log 2>&1 &
  done
done
wait
echo "done: $OUT"
