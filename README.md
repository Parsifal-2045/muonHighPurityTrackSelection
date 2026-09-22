# Selector for muon tracks in the Pixel detector of CMS
This repository contains code for the exploration and development of an ML-based high purity selection for muon tracks in the Pixel detector of the CMS experiment. The project aims to be deployed for the Phase-2 High Level Trigger reconstruction, taking advantage of the improved PixelTracks reconstruction to effectively track muons without running a separate dedicated tracking iteration.
Development based on CA extension for Phase 2 and previous integration of Alpaka-based PixelTracks in muons sequences, as shown in [22/07 HLT Upgrade meeting](https://indico.cern.ch/event/1570468/#4-single-iteration-io-muon-rec). NB the results shown there were preliminary, much more recent results profiting of improvements in PixelTracking (such as [better tuning and duplicates treatment](https://github.com/cms-sw/cmssw/pull/51085) and a [new HP DNN for tracks](https://github.com/cms-sw/cmssw/pull/51042)) were shown in the [09/06 Muon POG meeting](https://indico.cern.ch/event/1690203/#32-dnn-for-high-purity-selecti).

This work aims to reduce the fake tracks (i.e. not matched to a signal muon as per validation selection) present at the tracking level for HLT muon reconstruction, with the aim to replace the cut-based MVA selection typically applied to L3 tracks (meaning PixelTracks extended to the Outer Tracker via ckf).
Development steps:

- [x] Extend CMSSW ntuple producer to include MC truth information (using the same simToReco and recoToSim associators used by the standard validation sequences) ([commit](https://github.com/Parsifal-2045/cmssw/commit/04dd7388b7e0e2b6c0dc053f85071fb3d856b857) on private CMSSW branch)
- [x] Extract ntuples with variables of interest (mostly track quality parameters) and MC truth information (e.g. matched, duplicate, simulated pT, ...) ([commit](https://github.com/Parsifal-2045/cmssw/commit/f42d8e5b6ccfaa854c9a196621dfc4d7140c86e3) on private CMSSW branch)
- [x] Visual inspection of extracted variables to figure out which ones could be used to discriminate good tracks from fakes ([link](https://lferragi.web.cern.ch/plots/muon_hlt_phase2/muonPixelTracksSelector/) to plots and ntuples)
- [x] Implementation and training of ML models: a residual DNN and a gradient-boosted forest (XGBoost), trained at Phase-2 pileup conditions on a varied sample (particle-gun muons over the full HLT pT spectrum plus low-pT Bs→μμ)
- [x] Analysis of model results and fine tuning: feature-set pruning (44 → 33 features) via gain/permutation importance, DNN architecture scan, F2-based threshold optimisation
- [x] Export of the models for CMSSW: ONNX (opset 13) for the DNNs and a custom compact binary format for the forests (2.5x smaller than the ONNX serialisation), each verified numerically against the Python inference
- [x] Integration in CMSSW (`RecoMuon/L3TrackFinder`) with stream producers for both inference backends, wired into the Phase-2 HLT muon sequences behind process modifiers
- [x] Bit-level cross-check of the C++ and Python feature extraction on a fixed sample (`production/features_analysis/`)
- [ ] Training on a multi-process sample (tt̄, DY/Z→μμ) to validate robustness for non-isolated muons
- [ ] Per-pT-region working points to recover the low-pT (≲5 GeV) efficiency

The first three steps had been completed on a sample of 10'000 ZMM events at 200 PU as a bootstrap; all subsequent developments use the production samples described below.

## Production models (`production/`)

Two track flavours are selected, both in the muon inside-out (IO) reconstruction chain:

- **Pixel tracks** (`muon_pixel_tracks_*`): HP selection applied to `hltPhase2MuonPixelTracks`, providing clean tracks for IO seeding and for the standalone-muon-less approach;
- **IO tracks / seeds** (`muon_general_tracks_*`): HP selection applied to `hltPhase2MuonIOTracks` in the MkFit-seeds IO sequence.

For each flavour two model families coexist, trained on identical data, splits and sample weights (`pixel_features.py` guarantees byte-identical features for the pixel case):

- **DNN**: residual MLP (hidden 144, 3 residual blocks, BatchNorm + PReLU + dropout 0.1, head 144→64→32→1), trained with torchrun/DDP using a pT-stratified differentiable F2 loss (low-pT stratum weighted 5x) plus a focal term, OneCycleLR, gradient clipping, patience 50 on validation PR-AUC, and an EMA-decay copy for model selection;
- **XGBoost**: histogram trees, depth 6, η = 0.1, 2000 boosting rounds with early stopping (100 rounds, on aucpr), subsample/colsample 0.8.

Current candidate productions (test-set metrics; thresholds are F2-optimal working points on the *weighted-training* score scales, not calibrated probabilities):

| Model | Features | PR-AUC | ROC-AUC | Precision @F2 | Recall @F2 | Exports |
|---|---|---|---|---|---|---|
| Pixel DNN (`pixel_dnn_output/`) | 36 | 0.99805 | 0.99881 | 0.959 | 0.991 | `model_standard.onnx`, `model_fast.onnx` (BatchNorm folded) |
| Pixel XGB (`pixel_xgb_output_33f/`) | 33 | **0.99911** | **0.99946** | 0.971 | 0.994 | `model_xgb.onnx`, `model.json`, **`model_compact.bin` → deployed** |
| Seeds XGB (`seeds_xgb_output_33f/`) | 33 | 0.99673 | 0.99895 | 0.953 | 0.987 | same, **`model_compact.bin` → deployed** |
| Pixel XGB, 1000 rounds (`pixel_xgb_output/`) | 44 | 0.99883 | 0.99929 | — | — | superseded by the 33-feature, 2000-round model |

The boosted forest is the current production choice: best PR-AUC on both flavours with a smaller, faster inference backend. The DNN remains competitive (a DNN size scan in `pixel_dnn_bench_results.json` shows even the smallest "nano" variant reaches PR-AUC 0.99723 at 8.2 µs/track single-sample CPU, vs 0.99868 at 25.8 µs for the largest), and shows slightly better low-pT recall. Older 44-feature DNN productions (`pixel_output/`, `tuned_pixel_output/`, `seeds_output/`; PR-AUC 0.99420–0.99808) are kept for reference; earlier development iterations live under `tests/` (including a separate outside-in selector line with standalone-muon matching features, `MuonOITracksDNNSelector` in CMSSW).

## Feature set

44 engineered features are built from the ntuple branches (`pixel_features.py::build_dataset`), then pruned to 33 for production based on gain/permutation importance:

- **Track kinematics and errors**: p, pT, η, φ, log-compressed error terms (ptErr, etaErr, phiErr, qoverpErr, lambdaErr, ...);
- **Hit content**: nPixelHits, nTrkLays, nFoundHits;
- **Derived quality features**: impact-parameter significances (sip2D, sipZ), |dxy|/pT, ptErr/p, σ(pT)/pT, |dz|/|dxy|, |η|, 3D impact (log₁₀(dxy²+dz²));
- **L1-track-muon (L1TkMu) matching**: best-match ΔR², |ΔpT|/pT, χ²-like pT compatibility, an ad-hoc matching score, number of loosely compatible L1 candidates, second-best ΔR², hasMatch flag;
- **L1 stub information for the matched L1TkMu**: stub counts (total/barrel/endcap), best stub quality, and the η/φ/depth region of the best stub — these dominate the forest's gain ranking (stub multiplicity is the single most powerful feature);
- **Soft low-pT indicator**: a clipped sigmoid of (pT − 5 GeV), giving the model a smooth handle on the low-pT regime where most of the discrimination loss is concentrated.

Sample weighting is shared by both models: signal boosted 5x, plus a kinematic 1/pT reweighting (capped at 20x, normalised within signal and background separately) to fight pT-spectrum imbalance.

## Training pipeline

- **Data**: 24 ROOT files per flavour (~217 GB): 20 particle-gun samples (low/medium/high/TeV-pT single muons) + 4 Bs→μμ samples, at Phase-2 pileup. Signal = track matched to a signal muon by the validation associator; background = unmatched ("fake") tracks.
- **Splits**: 64/16/20 train/validation/test, stratified by label (outer split additionally by input file), seed 42.
- **Thresholding**: a single global threshold is chosen as the F₂ (β=2) optimum on the precision-recall curve; per-pT-bin performance breakdowns are produced for reference.
- **Exports and verification**: DNN → opset-13 ONNX in two variants (`model_standard.onnx` with the StandardScaler folded into the first Linear; `model_fast.onnx` additionally with all Linear+BatchNorm pairs fused); XGBoost → ONNX via onnxmltools plus `model_compact.bin`, a custom little-endian format (node/feature/split/child arrays + base logit) consumed directly by the CMSSW forest module. Both exports are replayed in Python against the native inference and required to agree to ≲10⁻⁵ before being written.

## CMSSW integration

Both backends are stream EDProducers in `RecoMuon/L3TrackFinder/plugins`, developed on the [`20_1_X_muonTracking` branch](https://github.com/Parsifal-2045/cmssw/tree/20_1_X_muonTracking) of the private Parsifal-2045/cmssw fork:

- `MuonIOTracksForestSelector` — loads the compact `.bin` once per process (GlobalCache), does a serial tree traversal + sigmoid per track; used by the *active* HLT sequences for both flavours. ~2.5x smaller model than ONNX and measured ~2x faster at the typical 3–12 tracks/event occupancy;
- `MuonIOTracksDNNSelector` — ONNXRuntime-based (GlobalCache `cms::Ort::ONNXRuntime`), sigmoid inside the graph.

Both share one C++ 33-feature extractor whose semantics (log-epsilons, imputation sentinels, L1 matching windows ΔR<0.3/χ²pT<9, loose window ΔR<0.5/χ²pT<25, stub tie-breaking) mirror `build_dataset()` line by line; parity is validated with `production/features_analysis/` (per-track CSV dumps from CMSSW with `dumpFeatures=True` compared to the Python pipeline at the 10⁻⁶ level). The modules emit a filtered `reco::TrackCollection` plus a `scores` vector<float>.

Sequence wiring is done through process modifiers `phase2MuonPixelTracksSelector` and `phase2MuonSeedsSelector` (`Configuration/ProcessModifiers`), which swap the forest selectors into `HLTPhase2MuonPixelTracksFromL1TkSequence` and `HLTPhase2L3MuonsIOSequence` respectively; models and thresholds are taken from `RecoMuon/L3TrackFinder/data/` (deployed thresholds equal the F₂ points: 0.627851665019989 pixel, 0.6882812976837158 seeds; data files verified identical by checksum to the production exports).

Known integration drift (to fix): the DNN integration path is currently broken by a three-way feature-count mismatch. The C++ extractor shared by both selector plugins (`MuonIOTracksDNNSelector.cc`) always emits the pruned 33-feature set, and its runtime guard would throw on the first event; the two DNN cfis (`hltPhase2MuonPixelTracksHighPurity_cfi.py`, `hltPhase2MuonIOTrackSelectionHighPurity_cfi.py`) still declare `nFeatures=44`, carry stale F2 thresholds and reference the old 44-feature v6 ONNX models; and the newest trained DNN (`pixel_dnn_output/`, 36 inputs — it drops 8 features, not the forest's 11) matches neither. Until a DNN is retrained on exactly the 33-feature set the C++ extracts (or a dedicated C++ extraction variant is added for the DNN path) and the cfis updated accordingly, the DNN cfis are effectively dead config — the modifier-wired sequences use only the forest variants.

## Possible improvements

**Models**
- Per-pT (or per-η) working points: the 0–5 GeV bin is the only weak region (recall ~97% vs ≥99.5% above, PR-AUC ~0.984 vs ~0.998); per-bin thresholds on the existing score would likely beat a global F2 point at zero retraining cost.
- DNN–forest ensembling or distillation: the two families disagree enough to be complementary — a stacked combiner, or a compact DNN distilled from the forest, could recover forest-level PR-AUC at sub-10 µs.
- Retrain the DNN on the pruned 33-feature set so both backends share the single CMSSW extractor (also fixes the DNN cfi drift above).
- New features to evaluate: beamspot-relative impact parameters (dxy/dz are currently computed w.r.t. the origin — with matching care on the Python side), per-layer pixel hit patterns, hit-fit χ² contributions, stub geometry distributions beyond the best stub (e.g. per-depth stub counts), L1-object isolation sums, and the unused `duplicate` flag as an auxiliary training signal or a third class.
- Probability calibration (Platt/isotonic) if calibrated scores are wanted for threshold setting; otherwise publish an efficiency-vs-threshold table rather than a single F2 point.

**Data and evaluation**
- Train/validate/test splits are track-level, not event-level: tracks from the same event can appear in train and test, so quoted metrics are mildly optimistic. Group by (file, event) and re-measure; add k-fold for the forest to get error bars.
- Threshold and DNN checkpoint/EMA selection are done on the test set, which has also been reused across experiments — move all selection to the validation split and keep a frozen confirmation sample.
- The 44→33 feature pruning was selected on the same data used for evaluation; redo with a nested split.
- Extend the training mix beyond muon guns + Bs→μμ (tt̄, DY→μμ, Z→μμ at 200 PU) and check PU robustness (train @200, test @140/200).

**Pipeline and reproducibility**
- Consolidate the four scripts: a prefix-parametrised feature module would let `seeds_*.py` drop ~1000 lines of duplicated code, and `pixel_model.py` currently re-implements `compute_sample_weights` inline.
- Commit the tuning/benchmark scripts referenced in comments (`pixel_xgb_optimize.py`, the DNN size-scan producer of `pixel_dnn_bench_results.json`), pin `requirements.txt`, and emit a per-run manifest (git hash, input file list + checksums, package versions, full hyperparameters) next to each output directory.
- Assert the DNN ONNX parity on real test rows (currently checked on one random dummy input), same as the XGB path does.
- Key the dataset cache by input-file hash (currently written to one fixed, unversioned location) or move it under the output dir, to avoid stale-cache reuse across data-dir changes.
- Fix the misnamed `impact3D` feature (it is log₁₀(dxy²+dz²), the log of the *squared* 3D impact parameter) or document the convention.
- Refresh the CMSSW side: correct the DNN-plugin header doc (still shows the 44-feature order), clean up dead/unreferenced model files in `RecoMuon/L3TrackFinder/data/`, and add a provenance note (source run + checksum) next to each deployed model.
