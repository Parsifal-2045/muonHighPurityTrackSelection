# Selector for muon tracks in the Pixel detector of CMS
This repository contains code for the exploration and development of ML-based high purity selections for muon tracks in the Pixel detector of the CMS experiment. The project aims to be deployed for the Phase-2 High Level Trigger reconstruction, taking advantage of the improved PixelTracks reconstruction to effectively track muons without running a separate dedicated tracking iteration.
Development based on CA extension for Phase 2 and previous integration of Alpaka-based PixelTracks in muons sequences, as shown in [22/07 HLT Upgrade meeting](https://indico.cern.ch/event/1570468/#4-single-iteration-io-muon-rec). NB the results shown there were preliminary; further developments used as baseline for this work: CA extension implemented in CMSSW (PR) and PixelTracks usage for muon tracking (PR).

This work aims to reduce the fake tracks (i.e. not matched to a signal muon as per validation selection) present at the tracking level for HLT muon reconstruction, with the aim to replace the cut-based MVA selection typically applied to general tracks (meaning PixelTracks extended to the Outer Tracker via ckf).
Development steps:

- [x] Extend CMSSW ntuple producer to include MC truth information (using the same simToReco and recoToSim associators used by the standard validation sequences) ([commit](https://github.com/Parsifal-2045/cmssw/commit/04dd7388b7e0e2b6c0dc053f85071fb3d856b857) on private CMSSW branch)
- [x] Extract ntuples with variables of interest (mostly track quality parameters) and MC truth information (e.g. matched, duplicate, simulated pT, ...) ([commit](https://github.com/Parsifal-2045/cmssw/commit/f42d8e5b6ccfaa854c9a196621dfc4d7140c86e3) on private CMSSW branch)
- [x] Visual inspection of extracted variables to figure out which ones could be used to discriminate good tracks from fakes ([link](https://lferragi.web.cern.ch/plots/muon_hlt_phase2/muonPixelTracksSelector/) to plots and ntuples)
- [x] Implementation and training of ML models: residual DNNs and gradient-boosted forests (XGBoost), trained at Phase-2 pileup conditions on a varied sample (particle-gun muons over the full HLT pT spectrum plus low-pT Bs→μμ)
- [x] Analysis of model results and fine tuning: feature-set pruning (44 → 33 IO features, 26 → 22 OI features) via gain/permutation importance, DNN architecture scan, F2-based threshold optimisation
- [x] Export of the models for CMSSW: ONNX (opset 13) for the DNNs and a custom compact binary format for the forests (2.5x smaller than the ONNX serialisation), each verified numerically against the Python inference (< 1e-5)
- [x] Integration in CMSSW (`RecoMuon/L3TrackFinder` on the [`20_1_X_muonTracking` branch](https://github.com/Parsifal-2045/cmssw/tree/20_1_X_muonTracking) of the private Parsifal-2045/cmssw fork) with stream producers for both inference backends, wired into the Phase-2 HLT muon sequences behind process modifiers
- [x] Bit-level cross-check of the C++ and Python feature extraction on a fixed sample (`production/features_validation/`, IO selectors)
- [x] Forests selected as the production backend for all four selectors; v2 campaign: event-level (evt10) splits, validation-only working-point and forest-size selection, budget-and-prune tree sizing, per-pT-bin set points

The first three steps had been completed on a sample of 10'000 ZMM events at 200 PU as a bootstrap; all subsequent developments use the production samples described below.

## Repository structure

Production code and models (the **XGBoost forests are the production architecture**; DNNs are kept as the alternative backend, currently not wired in CMSSW):

- `production/` — shared feature/evaluation module (`pixel_features.py`), the IO forest trainers (`pixel_xgb.py`, `seeds_xgb.py`), `requirements.txt`.
- `production/io/` — production models with their full training outputs: `pixel/` (pixel-track forest), `seeds/` (IO-track forest); `archive/` holds the superseded campaigns (v1 forests and DNNs). Each model directory carries the full training log (`train.log`), thresholds (`.txt` and machine-readable `thresholds.json`), metrics and feature-importance artifacts, and all exports (`model.json`, `model_xgb.onnx`, `model_compact.bin`).
- `production/oi/` — outside-in selectors: feature extraction matching the CMSSW module (`OI_features.py`), shared runner (`OI_xgb_common.py`), flavour trainers (`OI_pixel_xgb.py`, `OI_general_xgb.py`), production forests (`pixel/`, `general/`, with the same artifact set), and the full-feature round-1 runs that motivated the 26→22 pruning (`pixel_26f/`, `general_26f/`).
- `production/dnn/` — the (currently unwired) DNN trainers and architecture benchmark.
- `production/features_validation/` — Python/C++ feature-extraction cross-checks (per-track CSV dumps and comparison).
- `tests/` — development iterations and earlier studies (kept intact).

## Production models (`production/`)

Four track selectors cover the Phase-2 muon HLT chains. Inside-out (IO):

- **Pixel tracks** (`muon_pixel_tracks_*`): HP selection on `hltPhase2MuonPixelTracks`;
- **IO tracks / seeds** (`muon_general_tracks_*`): HP selection on `hltPhase2MuonIOTracks` in the MkFit-seeds IO sequence.

Outside-in (OI, `l3_tk_OI_*` tracks with L2 standalone-muon matching):

- **OI pixel chain**: trained on the pixel-selector production sample;
- **OI general chain**: trained on the seeds-selector production sample (the OI track content follows the HLT chain of the sample).

All four production models are XGBoost forests trained by the v2 pipeline (see below; DNN variants exist for the IO selectors and are described further down). Test-split metrics (frozen working points, event-level split = no same-event leakage):

| Model | Features | Trees | ROC-AUC | PR-AUC | Precision @F2 | Recall @F2 | Directory |
|---|---|---|---|---|---|---|---|
| Pixel IO forest | 33 | 3250 | 0.99956 | 0.99931 | 0.974 | 0.995 | `production/io/pixel/` |
| Seeds IO forest | 33 | 4050 | 0.99913 | 0.99753 | 0.961 | 0.989 | `production/io/seeds/` |
| OI pixel forest | 22 | 1200 | 0.99970 | 0.99703 | 0.968 | 0.989 | `production/oi/pixel/` |
| OI general forest | 22 | 850 | 0.99982 | 0.99884 | 0.984 | 0.994 | `production/oi/general/` |

Reference points on the same convention: the legacy OI DNNs currently deployed in CMSSW score on the same test splits (C++-convention features) at P/R = 0.823/0.948 (pixel) and 0.849/0.960 (general) — far below the forests, partly because of the feature-parity defects documented below. The previous IO forests (2000 trees, 33f) reached PR-AUC 0.99911 (pixel) / 0.99673 (seeds) on the old track-level splits; a same-split comparison on the new evt10 test set is biased upward in their favour (their training saw tracks from those events) and not meaningful.

**Fake rejection** at the deployed per-pT-bin working points (test split): pixel 98.4% fakes rejected at 99.5% muon efficiency, seeds 99.1% at 98.9%, OI pixel 99.86% at 98.9%, OI general 99.83% at 99.4%. Evaluated per pT bin against the alternative of a uniform high recall target (99% in every bin), the per-bin F2 set points cost nothing elsewhere and avoid the rejection collapse that a recall target forces in the pT < 2 GeV bin (e.g. 0.36 rejection for the seeds forest, where the score distributions overlap too much to reach 99% recall cheaply). A Lagrangian per-bin threshold reallocation maximizing total fake rejection at fixed overall signal efficiency reproduces the deployed set points within ≤0.1 pp rejection for all four models — the deployed working points sit on the Pareto frontier, so further rejection must come from the model (features/training), not from thresholding.

Every output directory carries the full training log (`train.log`), thresholds (`.txt` and machine-readable `thresholds.json`), metrics/plots, gain and low-pT permutation feature importances, and all exports (`model.json`, `model_xgb.onnx`, `model_compact.bin`).

## Forest training pipeline (v2)

- **Data**: 24 ROOT files per flavour (~217 GB, 20 particle-gun samples covering the HLT pT spectrum + 4 Bs→μμ samples, Phase-2 pileup). Signal = track matched to a signal muon by the validation associator; background = unmatched ("fake") tracks. ~9.8M (pixel) / ~20.4M (seeds) tracks.
- **Split — "evt10"**: event-level, `ev % 10` → train 60% / val 10% / test 30%. Every track of an event stays in one split (no cross-leakage), and each sample contributes the same composition (replaces the old track-level stratified split, which let correlated tracks of one event leak between train and test and therefore overestimated the deployed metrics).
- **Weights**: signal boosted 5x, kinematic 1/pT reweighting capped at 20x (normalised per class).
- **Trees**: histogram XGBoost, depth 6, η = 0.1, subsample/colsample 0.8, GPU-trained. The forest is fitted to a **budget of 5000 boosting rounds with early stopping disabled** and then pruned to the smallest tree prefix whose validation F2 (at validation-chosen per-bin working points) is within **prune_tol = 0.001** of the best prefix — the same scheme as the HP forest in the PixelTracking chain. The 2000-tree production models had saturated their budget (best_iteration = 1999); the 5000-tree budget curves showed pixel saturating only beyond ~3000, and pruning kept 3250 (pixel), 4050 (seeds), 1200 (OI pixel), 850 (OI general) trees.
- **Working points — per-pT-bin set points**: one F2-optimal threshold per pT bin (`[0,2,5,10,50,200]` GeV, first bin separating pT < 2 GeV), derived **on the validation split only** (bins with < 100 signal tracks fall back to the global F2 threshold); the test split is evaluated exactly once with frozen working points. The previous pipeline picked thresholds on the test set.
- **Feature importances and pruning**: XGBoost gain ranking over the full test set plus permutation importance (PR-AUC drop) on the low-pT subset. IO pruning 44 → 33 (previous campaign): 2 zero-gain features, 6 marginal χ²/error features, 3 fully redundant ones. OI pruning 26 → 22 (round-1 `production/oi/*_26f/` importance): `ptErr`, `sigmaPtOverPt`, `relUncertaintyProduct` (all duplicate `qoverpErr`'s pT-uncertainty information) and `chi2` (subsumed by `normalizedChi2`/`chi2PerHit`); the round-2 re-trainings cost no measurable performance (PR-AUC within ±0.0001).
- **Verification**: every export is replayed in Python against XGBoost predict and required to agree < 1e-5 (`model_xgb.onnx` via ONNXRuntime, `model_compact.bin` via a numpy re-implementation of the CMSSW traversal). The compact .bin header is `int32 nNodes, int32 nTrees, fp32 baseLogit` followed by node feature/threshold/child arrays.

## OI feature extraction and the CMSSW parity fix

The OI selectors extract 26 features: 12 log-compressed track/branches + 5 plain + 6 derived (impact parameters, χ² terms, hit efficiencies) + 2 standalone-matching features (`l2_mu_vtx_hasMatch`, `l2_mu_vtx_matchingScore`) + the soft `is_low_pt`. The training-side extraction (`production/oi/OI_features.py`) mirrors the C++ extractor (`MuonOITracksDNNSelector.cc`) feature-for-feature — which the legacy OI DNN training (`tests/OI_*_model.py`) did **not**:

1. the track-to-standalone Δφ is wrapped (`reco::deltaPhi` convention) while the legacy python used the raw difference;
2. the matching score is (χ²η + χ²φ + χ²pT + χ²dz)/9 while the legacy python used the unnormalised sum;
3. the no-match imputation of the score is 10.0 in raw space while the legacy python imputed log10(10) ≈ 1.0.

The OI models therefore went through the forest campaign with C++-faithful extraction; reproduce them with `production/oi/OI_pixel_xgb.py` / `production/oi/OI_general_xgb.py` (shared runner `production/oi/OI_xgb_common.py`).

## CMSSW integration

All four selectors are implemented on the [`20_1_X_muonTracking` branch](https://github.com/Parsifal-2045/cmssw/tree/20_1_X_muonTracking) of the private Parsifal-2045/cmssw fork, under `RecoMuon/L3TrackFinder`. Given a track and its matching inputs, the feature vectors are built by two shared headers — `interface/IOTrackSelectorFeatures.h` (`muonhp::IOTrackFeatures`, 33 named fields) and `interface/OITrackSelectorFeatures.h` (`muonhp::OITrackFeatures`, 22 named fields) — whose `toArray()` maps the named struct to the canonical training order; both the forest and the DNN plugin of each family call the same extraction function (no index bookkeeping, no feature-drop logic in the plugins: the extractors emit exactly the 33/22 production features).

Forests are the deployed backend for all four chains (modifier-dependent):

- `MuonIOTracksForestSelector` — IO forest inference (custom compact `.bin`, serial traversal + sigmoid). Supports the pT-binned working points via `ptBinEdges` / `decisionThresholds` (threshold i applies to tracks with pT in [edges[i], edges[i+1]), last bin open-ended; empty vectors fall back to the single `decisionThreshold`). The deployed `.bin` files (`pixel_track_selector_forest.bin`, `seeds_track_selector_forest.bin`) are the v2 models; the previous 2000-tree models are kept alongside as `*_33f_2000t.bin`. The corresponding cfis carry the per-bin F2 set points from the trainings' `thresholds.json`.
- `MuonOITracksForestSelector` — OI forest inference, sharing the `MuonOITracksDNNSelector` extraction (22-feature production set; the 26-feature layout's ptErr, χ², σpT/pT and relUncertaintyProduct were pruned in round 2). Deployed `.bin` files: `OI_pixel_track_selector_forest.bin` (1200 trees), `OI_general_track_selector_forest.bin` (850 trees).
- cfis: `hltPhase2MuonPixelTracksHighPurityForest_cfi.py`, `hltPhase2MuonIOTrackSelectionHighPurityForest_cfi.py`, `hltPhase2L3OIMuonTrackSelectionHighPurityForest_cfi.py`. Sequence wiring: the process modifiers `phase2MuonPixelTracksSelector` / `phase2MuonSeedsSelector` swap the forests into `HLTPhase2MuonPixelTracksFromL1TkSequence` (IO pixel), `HLTPhase2L3MuonsIOSequence` (IO seeds) and the OI selection (`hltPhase2L3OIMuonTrackSelectionHighPurity` replacement, both chains — the legacy DNN replacements in `hltPhase2L3OIMuonTrackSelectionHighPurity_cfi.py` are kept but commented out).
- DNN backends (`MuonIOTracksDNNSelector`, ONNXRuntime) remain available but are not wired: known integration drift — the DNN cfis still declare `nFeatures=44` and reference the old 44-feature v6 ONNX models while the C++ extractor emits the pruned 33 features, and the newest trained DNN (36 inputs) matches neither. Reviving them requires retraining the DNN on the 33-feature set (or a dedicated extraction variant).
- A smoke config (`RecoMuon/L3TrackFinder/test/forestSelectorsSmoke_cfg.py`) constructs all four forest producers (config-schema, plugin registration, `.bin` loading); the full ML path is validated per-track by the `features_validation` CSV cross-check for the IO selectors — rerun it (and an equivalent OI check) with `dumpFeatures = True` whenever the extraction changes.

## Possible improvements

**Models**
- Forest–DNN ensembling or distillation, if a DNN is wanted back in CMSSW after retraining on the deployed 33-feature set.
- Multi-process training sample (tt̄, DY→μμ, Z→μμ beyond muon guns + Bs→μμ); PU-robustness check (train at 200 PU, test at 140/200).
- Per-η (or η×pT) working points — the machinery (bin edges + threshold vectors) is now in place on both the python and the C++ side.
- Probability calibration (Platt/isotonic) if calibrated scores are needed; otherwise publish an efficiency-vs-threshold table alongside the F2 points.
- The `duplicate` flag (currently unused) as an auxiliary target or third class.
- The C++ OI `hasMatch` feature is ~always 1 in events containing any standalone muon (bestScore < 25 over the best candidate); a track-level compatibility window would make it informative (chiefly for the pixel-chain OI model, where its gain is negligible).

**Pipeline and reproducibility**
- Consolidate scripts: a branch-prefix-parametrised feature module would remove ~1000 lines of duplication between the pixel and seeds trainers; the OI runner already demonstrates the shared-runner pattern.
- Commit the tuning/benchmark scripts referenced in comments (`pixel_xgb_optimize.py`, the DNN size-scan producer of `pixel_dnn_bench_results.json`), pin `requirements.txt`, and emit a per-run manifest (git hash, input-file list + checksums, package versions).
- Speed up or skip the low-pT permutation importance (currently the pipeline's longest CPU-only stage; batched prediction or a config flag would do) and vectorise the compact-.bin verification loop.
- Redo the 44→33 IO pruning with the nested-split protocol (it was selected on the same data that measured it).
- Fix the misnamed `impact3D` (log₁₀(dxy²+dz²) of the *squared* 3D impact parameter).
