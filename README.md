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
- [x] v3 campaign: shared and reproducible training pipeline, float64 feature convention shared with CMSSW (fixes a training/inference mismatch of the IO selectors), validation-only hyperparameter scan (deep forests), retrained and redeployed models, exact CMSSW/Python parity on real events, fixes to the L1TkMuon seed selector feeding the IO seeds model
- [ ] v4: training n-tuples re-produced with the current HLT configuration (including the fixed seed selector), retraining with the same chain (How to run the training chain) and a larger out-of-sample check of the forest depth

The first three steps had been completed on a sample of 10'000 ZMM events at 200 PU as a bootstrap; all subsequent developments use the production samples described below.

**Contents:** [Repository structure](#repository-structure) · **[How to run the training chain](#how-to-run-the-training-chain)** · [Production models](#production-models-v3-production) · [Forest training pipeline](#forest-training-pipeline-v3-productionforest_pipelinepy) · [Feature parity between training and CMSSW](#feature-parity-between-training-and-cmssw) · [CMSSW integration](#cmssw-integration) · [L1TkMuon input selectors](#l1tkmuon-input-selectors) · [Decisions and rationale](#decisions-and-rationale) · [Possible improvements](#possible-improvements)

## Repository structure

Production code and models (the **XGBoost forests are the production architecture**; DNNs are kept as the alternative backend, currently not wired in CMSSW):

- `production/forest_pipeline.py` — the shared training pipeline of the four forests (features and cache, split, training, size selection, working points, test evaluation, verified exports, provenance records). `production/pixel_xgb.py`, `production/seeds_xgb.py`, `production/oi/OI_pixel_xgb.py`, `production/oi/OI_general_xgb.py` only hold the flavour configuration.
- `production/pixel_features.py` — IO feature extraction (parametrised by the n-tuple branch prefix: pixel and seeds selectors share it), the production feature ABI (`io_production_features()`), evaluation helpers. `production/oi/OI_features.py` — the OI counterpart (`OI_PRODUCTION_FEATURES`).
- `production/onnx_utils.py` — ONNX Runtime sessions with explicit thread pools, pinned TorchScript exporter for the DNNs.
- `production/io/{pixel,seeds}/`, `production/oi/{pixel,general}/` — the production (v3) models with their full records: `train.log`, `thresholds.json`/`.txt`, `manifest.json` (git commit, packages, GPU, input files, cache key, verification results), `cmssw_cfi_snippet.py`, feature importances, plots and the exports (`model.json`, `model_compact.bin`, `model_xgb.onnx`). Superseded models: `production/io/archive/` (v1 forests, DNNs, v2 forests `*_v2`) and `production/oi/archive/` (v2 OI forests); round-1 OI feature studies in `production/oi/*_26f/`.
- `production/tuning/` — validation-only hyperparameter scan (`forest_scan.py`, results in `tuning/results/`, summary `summarize_scan.py`).
- `production/check_consistency.py` — checks training records, exports and (with `--cmssw-src`) the CMSSW deployment against each other. `production/deploy_cmssw.py` — installs the models in a CMSSW area (data files, cfi working points, data README). `production/make_summary.py` — result tables of the production models against the previous generation (`production/results/`).
- `production/features_validation/` — CMSSW/Python cross-check on real events (`validate_cmssw.py`, timing customisation `forest_timing_cff.py`); results in `features_validation/results/`. The 44-feature CSVs of the previous check are kept for reference.
- `production/input_selectors/` — efficiency study of the L1TkMuon input selectors (cmsRun customisation `input_selector_study_cff.py`, analysis `analyze_input_selectors.py`, results in `input_selectors/results/`).
- `production/dnn/` — the (currently unwired) DNN trainers and architecture benchmark.
- `tests/` — development iterations and earlier studies (kept intact).

## How to run the training chain

Every input is given on the command line, with no environment variables to set and no machine-specific paths in the scripts (`python <script> --help` lists all options). Commands are shown from `production/`; the scripts work from any directory.

### 1. Setup

- Python ≥ 3.9 with the packages of `production/requirements.txt` (e.g. `python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt`). A CUDA GPU is used when available, the CPU otherwise (announced in the log).
- The v3 models were trained with xgboost 2.1.4, numpy 2.0.2, awkward 2.8.12, uproot 5.6.9, scikit-learn 1.6.1, onnx 1.19.1, onnxruntime 1.19.2, onnxmltools 1.16.0 (recorded in every `manifest.json`). With the same versions and inputs a re-run reproduces `model.json`, `model_compact.bin`, `model_xgb.onnx` and the working points bit for bit.
- The CMSSW steps (6 and 7) need a CMSSW area with the `RecoMuon/L3TrackFinder` integration (see CMSSW integration), built and set up with `cmsenv`.

### 2. Inputs

The training n-tuples are the `NANO:@MUHLTTraining` output of each HLT chain, one directory per chain: every `*.root` file of the directory is read, one file per sample. File names carry no meaning to the pipeline (the split is by event number within each file). The IO and OI models of a chain read the same files (different branches).

| Flag | Script | Value |
|---|---|---|
| `--data-dir DIR` (required) | `pixel_xgb.py`, `oi/OI_pixel_xgb.py`, `dnn/pixel_model.py` | n-tuples of the pixel-track chain (process modifier `phase2MuonPixelTracksSelector`) |
| `--data-dir DIR` (required) | `seeds_xgb.py`, `oi/OI_general_xgb.py`, `dnn/seeds_model.py` | n-tuples of the seeds chain (process modifier `phase2MuonSeedsSelector`) |
| `--data-dir DIR` (required) | `tuning/forest_scan.py` | n-tuples of the scanned flavour's chain |
| `--pixel-data-dir DIR`, `--seeds-data-dir DIR` (required) | `make_summary.py` | n-tuples of both chains |
| `--cache-dir DIR` | forest trainings, `tuning/forest_scan.py`, `make_summary.py` | feature cache (default `/tmp/$USER/muonhp_feature_cache`; 1.9 / 3.8 / 0.6 / 0.8 GB for IO pixel / IO seeds / OI pixel / OI general) |
| `--cmssw-src DIR`, `--out DIR` (required), `--relval-zmm DIR`, `--relval-bs DIR` | `features_validation/run_cmssw_validation.sh` | CMSSW `src/`, output directory, GEN-SIM-DIGI-RAW RelVal directories (default: the CMSSW_20_0_0_pre1 PU200 RelVals on EOS used for v3) |
| `--data-dir DIR` (required) | `tests/*_model.py`, `io/archive/test/*_model.py` | n-tuples of the earlier studies |

The commands below use two shell variables for the n-tuple directories:
```bash
PIXEL=/path/to/pixel-chain/ntuples
SEEDS=/path/to/seeds-chain/ntuples
```

### 3. Start a new model version (only when retraining for deployment)

Skip this step to reproduce the deployed v3 models. A retraining meant for CMSSW gets a new version: the new models are compared with the deployed ones, and they get new file names (cms-data files are immutable). Archive the deployed models:
```bash
cp -a io/pixel io/archive/pixel_xgb_output_33f_v3; cp -a io/seeds io/archive/seeds_xgb_output_33f_v3
cp -a oi/pixel oi/archive/pixel_22f_v3;            cp -a oi/general oi/archive/general_22f_v3
```
then set `model_version="v4"` and `ref_version="v3"` in `forest_pipeline.DEFAULTS`. This is the only place holding the version: it fills the deployed file names (`cmssw_bin`) and the reference model of every report (`ref_model_path`).

### 4. Train the four forests

```bash
python pixel_xgb.py         --data-dir $PIXEL --force   # IO pixel   -> io/pixel/
python seeds_xgb.py         --data-dir $SEEDS --force   # IO seeds   -> io/seeds/
python oi/OI_pixel_xgb.py   --data-dir $PIXEL --force   # OI pixel   -> oi/pixel/
python oi/OI_general_xgb.py --data-dir $SEEDS --force   # OI general -> oi/general/
python check_consistency.py
```
On one H100 the trainings take 13 / 19 / 5 / 5 min, plus a one-off feature extraction for new n-tuples (later runs read the cache). Each writes its model directory: model, exports, working points, `train.log` (ending with the comparison to the reference model on the same test split), `thresholds.json`, `manifest.json` and `cmssw_cfi_snippet.py`. `check_consistency.py` re-exports every model from `model.json` and checks tree counts, checksums, the feature ABI, the exports and that the logs are free of warnings.

Other options: `--force` replaces an existing model directory (without it the training refuses to overwrite), `--output-dir DIR` writes elsewhere, `--set key=value` overrides a configuration value (e.g. `--set max_depth=8`), `--device cpu` forces the CPU. Do not edit `pixel_features.py` or `oi/OI_features.py` while a training runs: the job stops (see the feature cache).

### 5. Re-check the hyperparameters (optional)

Recommended when the samples change substantially. Per flavour (`io_pixel`, `io_seeds`, `oi_pixel`, `oi_general`), on the production validation fold and on the confirmation fold:
```bash
python tuning/forest_scan.py io_pixel --data-dir $PIXEL --configs base d10 d12 d12_lr02
python tuning/forest_scan.py io_pixel --data-dir $PIXEL --configs base d10 d12 d12_lr02 --val-fold 9
python tuning/summarize_scan.py > results/tuning_summary.md
```
`python tuning/forest_scan.py --list` shows the configurations. Change `DEFAULTS` (IO) or `OI_BASE` (OI) only for a configuration ahead on both folds at comparable inference cost, then retrain (step 4).

### 6. Deploy to CMSSW

```bash
python deploy_cmssw.py --cmssw-src $CMSSW_BASE/src
(cd $CMSSW_BASE/src && scram b)
python check_consistency.py --cmssw-src $CMSSW_BASE/src
```
`deploy_cmssw.py` copies the `.bin` files, writes the working points into the cfis, removes the previous model files and writes `RecoMuon/L3TrackFinder/data/README.md`; `check_consistency.py --cmssw-src` also checks the deployed files and cfi values.

### 7. Validate in CMSSW

L1 + HLT + training NANO on 250 ZMM and 250 BsToMuMu RelVal events per chain (four parallel jobs), then the track-by-track parity, the out-of-sample model comparison and the input-selector efficiencies (`OUT` = any scratch directory; replace `v4` by the new version):
```bash
features_validation/run_cmssw_validation.sh --cmssw-src $CMSSW_BASE/src --out $OUT
for s in ZMM Bs; do
  python features_validation/validate_cmssw.py --chain pixel --nano $OUT/final_pixel_$s.root --log $OUT/final_pixel_$s.log \
      --io-model io/pixel --oi-model oi/pixel --json features_validation/results/parity_v4_pixel_$s.json
  python features_validation/validate_cmssw.py --chain seeds --nano $OUT/final_seeds_$s.root --log $OUT/final_seeds_$s.log \
      --io-model io/seeds --oi-model oi/general --json features_validation/results/parity_v4_seeds_$s.json
done
python features_validation/compare_models_relval.py --family IO --prefix muon_pixel_tracks \
    --nano $OUT/final_pixel_{ZMM,Bs}.root --models io/archive/pixel_xgb_output_33f_v3 io/pixel \
    --json features_validation/results/relval_model_comparison_io_pixel.json
python features_validation/compare_models_relval.py --family IO --prefix muon_general_tracks \
    --nano $OUT/final_seeds_{ZMM,Bs}.root --models io/archive/seeds_xgb_output_33f_v3 io/seeds \
    --json features_validation/results/relval_model_comparison_io_seeds.json
python input_selectors/analyze_input_selectors.py --chain pixel --nano $OUT/final_pixel_{ZMM,Bs}.root \
    --label ZMM BsToMuMu --io-model io/pixel --json input_selectors/results/pixel_chain_v4.json
python input_selectors/analyze_input_selectors.py --chain seeds --nano $OUT/final_seeds_{ZMM,Bs}.root \
    --label ZMM BsToMuMu --io-model io/seeds --json input_selectors/results/seeds_chain_v4.json
```
`--models` lists the reference (the models archived in step 3) first. Expected: bit-identical features and 0 differing decisions for all four selectors (any difference means the training and CMSSW extractions diverged). The per-module times are in the TimeReport of `$OUT/final_*.log`. Other options of the validation script: `--events N`, `--threads N`, `--relval-zmm DIR`, `--relval-bs DIR`, and `--alt-models DIR` to time other models next to the deployed ones (see its `--help`).

### 8. Summarise the results

```bash
python make_summary.py --pixel-data-dir $PIXEL --seeds-data-dir $SEEDS --json results/summary_v4_vs_v3.json
```
prints the result tables of this README (production vs reference models, per working-point bin); update the result sections below with them.

## Production models (v3, `production/`)

Four track selectors cover the Phase-2 muon HLT chains. Inside-out (IO): **pixel tracks** (`hltPhase2MuonPixelTracks`, pixel-track chain, n-tuple `muon_pixel_tracks_*`) and **IO tracks from LST seeds** (`hltPhase2MuonIOTracks`, seeds chain, `muon_general_tracks_*`). Outside-in (OI, `hltPhase2L3OIMuCtfWithMaterialTracks`, `l3_tk_OI_*`, with L2 standalone-muon matching): **OI pixel** (trained on the pixel-chain sample) and **OI general** (seeds-chain sample).

Test split (event-level, never used for any choice), deployed per-pT-bin working points; v2 = previous production models, evaluated on the same test tracks with their own deployed working points (and the v3 feature convention, i.e. as they would run in CMSSW now). Node visits = mean number of internal tree nodes visited per track by the CMSSW traversal (the inference cost); "fakes kept" = unmatched test tracks accepted.

| Model | Version | Trees | Nodes | .bin [MB] | Node visits / track | ROC-AUC | PR-AUC | Precision | Recall | F2 | Fake rejection | Fakes kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IO pixel | v2 | 3250 | 378,584 | 4.9 | 19,480 | 0.99956 | 0.99931 | 0.9743 | 0.9947 | 0.9906 | 0.9845 | 27,392 |
| IO pixel | **v3** | 800 | 1,197,968 | 15.6 | **9,251** | 0.99977 | 0.99963 | **0.9822** | **0.9959** | **0.9932** | 0.9893 | **18,827** |
| IO seeds | v2 | 4050 | 475,524 | 6.2 | 24,280 | 0.99913 | 0.99753 | 0.9613 | 0.9888 | 0.9832 | 0.9914 | 40,571 |
| IO seeds | **v3** | 1250 | 1,919,368 | 25.0 | **14,669** | 0.99952 | 0.99880 | **0.9803** | **0.9930** | **0.9905** | 0.9957 | **20,348** |
| OI pixel | v2 | 1200 | 119,678 | 1.6 | 7,026 | 0.99970 | 0.99703 | 0.9680 | 0.9893 | 0.9849 | 0.9986 | 1,981 |
| OI pixel | **v3** | 300 | 166,972 | 2.2 | **2,524** | 0.99971 | 0.99722 | 0.9710 | 0.9891 | 0.9854 | 0.9987 | 1,791 |
| OI general | v2 | 850 | 88,912 | 1.2 | 5,025 | 0.99982 | 0.99884 | 0.9838 | 0.9935 | 0.9916 | 0.9983 | 2,858 |
| OI general | **v3** | 200 | 137,918 | 1.8 | **1,700** | 0.99983 | 0.99888 | 0.9828 | 0.9939 | 0.9916 | 0.9982 | 3,028 |

IO: at higher muon efficiency, the fakes kept drop by 31% (pixel) and 50% (seeds), with half / 60% of the node visits per track (the deep forests are larger, so the CMSSW wall time does not drop accordingly: see Behaviour in CMSSW). OI: performance unchanged within the statistical precision of the (smaller) OI samples, with a third of the node visits. These are training-distribution (test split) numbers; on independent RelVal events the v2/v3 differences are within the statistical reach of the check (see the out-of-sample check).

**v3 per working-point bin** (test split; threshold = deployed cfi value, pT of the track; `production/results/summary_v3_vs_v2.json`):

| Model | pT bin [GeV] | Threshold | Signal | Fakes | Precision | Recall | F2 | Fake rejection | Fakes kept |
|---|---|---|---|---|---|---|---|---|---|
| IO pixel | 0–2 | 0.107 | 4,597 | 136,378 | 0.8718 | 0.9215 | 0.9111 | 0.9954 | 623 |
| IO pixel | 2–5 | 0.213 | 309,523 | 1,594,114 | 0.9471 | 0.9879 | 0.9794 | 0.9893 | 17,088 |
| IO pixel | 5–10 | 0.502 | 256,340 | 34,239 | 0.9960 | 0.9995 | 0.9988 | 0.9700 | 1,028 |
| IO pixel | 10–50 | 0.544 | 114,254 | 1,326 | 0.9993 | 1.0000 | 0.9998 | 0.9382 | 82 |
| IO pixel | 50–200 | 0.863 | 145,634 | 164 | 1.0000 | 1.0000 | 1.0000 | 0.9695 | 5 |
| IO pixel | > 200 | 0.213 (global) | 214,517 | 1 | 1.0000 | 1.0000 | 1.0000 | 0 | 1 |
| IO seeds | 0–2 | 0.036 | 4,379 | 1,397,119 | 0.8960 | 0.7787 | 0.7996 | 0.9997 | 396 |
| IO seeds | 2–5 | 0.166 | 288,598 | 3,228,099 | 0.9418 | 0.9807 | 0.9727 | 0.9946 | 17,496 |
| IO seeds | 5–10 | 0.347 | 261,967 | 82,737 | 0.9923 | 0.9984 | 0.9972 | 0.9756 | 2,021 |
| IO seeds | 10–50 | 0.471 | 112,626 | 6,346 | 0.9972 | 0.9990 | 0.9986 | 0.9504 | 315 |
| IO seeds | 50–200 | 0.560 | 143,095 | 579 | 0.9993 | 0.9998 | 0.9997 | 0.8152 | 107 |
| IO seeds | > 200 | 0.856 | 209,597 | 115 | 0.9999 | 1.0000 | 1.0000 | 0.8870 | 13 |
| OI pixel | 0–2 | 0.101 | 5,059 | 425,161 | 0.9299 | 0.9854 | 0.9737 | 0.9991 | 376 |
| OI pixel | 2–5 | 0.274 | 21,343 | 656,104 | 0.9632 | 0.9864 | 0.9816 | 0.9988 | 805 |
| OI pixel | 5–10 | 0.277 | 8,820 | 207,041 | 0.9665 | 0.9817 | 0.9787 | 0.9986 | 300 |
| OI pixel | 10–50 | 0.334 | 4,991 | 111,384 | 0.9725 | 0.9864 | 0.9836 | 0.9988 | 139 |
| OI pixel | 50–200 | 0.344 | 6,297 | 18,471 | 0.9832 | 0.9943 | 0.9920 | 0.9942 | 107 |
| OI pixel | > 200 | 0.504 | 14,046 | 6,269 | 0.9955 | 0.9979 | 0.9974 | 0.9898 | 64 |
| OI general | 0–2 | 0.477 | 5,793 | 482,514 | 0.9456 | 0.9786 | 0.9718 | 0.9993 | 326 |
| OI general | 2–5 | 0.462 | 42,782 | 806,050 | 0.9653 | 0.9890 | 0.9842 | 0.9981 | 1,520 |
| OI general | 5–10 | 0.323 | 27,976 | 245,318 | 0.9765 | 0.9934 | 0.9899 | 0.9973 | 668 |
| OI general | 10–50 | 0.647 | 15,438 | 136,727 | 0.9923 | 0.9918 | 0.9919 | 0.9991 | 119 |
| OI general | 50–200 | 0.657 | 21,644 | 26,634 | 0.9928 | 0.9952 | 0.9947 | 0.9941 | 156 |
| OI general | > 200 | 0.569 | 60,905 | 9,597 | 0.9961 | 0.9990 | 0.9984 | 0.9751 | 239 |

The IO fakes concentrate at 2–5 GeV (5–11 fakes per muon track at the input, > 85% of the fakes kept); above 10 GeV the IO inputs are almost pure (≤ 0.06 fakes per muon track) and the working points keep essentially every muon. The IO pixel > 200 GeV bin has no fake in the validation split and uses the global F2 threshold (0.213). The IO seeds < 2 GeV bin (0.3% of its tracks are muons) sits at recall 0.78 / fake rejection 0.9997: there the fakes rise steeply with the recall (test split: 396 fakes kept at the deployed point, 1.4k at recall 0.80, 14k at 0.85, 52k at 0.90), so the F2 optimum gives up a fifth of these muons.

## Forest training pipeline (v3, `production/forest_pipeline.py`)

- **Data** (v3: pixel-chain n-tuples produced in May 2026, seeds-chain n-tuples in April–May 2026): 24 ROOT files per HLT chain (~222 GB, 20 particle-gun samples covering the HLT pT spectrum + 4 Bs→μμ samples, Phase-2 pileup). Signal = track matched to a muon TrackingParticle by the validation associator; background = unmatched ("fake") tracks. ~9.8M (pixel), ~20.4M (seeds), ~5.4M / ~6.9M (OI) tracks.
- **Feature cache**: content-addressed (`--cache-dir`, default `/tmp/$USER/muonhp_feature_cache`): the key covers the extraction code (source of `build_dataset` and every function/constant it uses), the branch list, the input-file manifest (path, size, mtime) and the numpy/awkward/uproot versions, so a stale cache is never used. A job stops if the feature code changes on disk while it runs: the key is computed from the source files, which then no longer describe the running code.
- **Feature ABI**: the kept features are asserted name by name against the order of the CMSSW extractors (`io_production_features()`, `OI_PRODUCTION_FEATURES` = `toArray()` of `muonhp::IOTrackFeatures` / `muonhp::OITrackFeatures`); a dropped feature that does not exist is an error.
- **Split — evt10**: `ev % 10` → train 60% / val 10% / test 30%, event-level (no same-event leakage), same composition per sample.
- **Weights**: signal ×5, kinematic 1/pT weight capped at 20 (normalised per class).
- **Trees**: histogram XGBoost, subsample/colsample 0.8, **IO: depth 12, η = 0.2; OI: depth 10, η = 0.1** (validation scan below), 5000-tree budget, no early stopping. The quantile cuts are computed on the host (`QuantileDMatrix`), then trees are grown on the GPU: the GPU sketch of weighted data is not reproducible across processes (the v2 models cannot be re-created), the host sketch is — a re-run reproduces every output bit for bit (checked across processes and GPUs).
- **Size selection**: smallest tree prefix whose validation F2 (at validation-chosen per-bin working points) is within 0.001 of the best prefix. v3 keeps 800 / 1250 / 300 / 200 trees (IO pixel / IO seeds / OI pixel / OI general).
- **Working points**: one F2-optimal threshold per pT bin (`[0,2,5,10,50,200]` GeV), validation only; bins with < 100 signal tracks or no background fall back to the global F2 point (IO pixel > 200 GeV: no fakes in the validation split). The test split is evaluated once.
- **Exports and verification**: `model.json`; `model_compact.bin` (CMSSW), replayed with a numpy re-implementation of the CMSSW traversal on 300k random test tracks plus every test track within 0.02 of its working point (asserts |Δscore| < 1e-5 and **0 working-point decision flips**; the exporter also refuses models whose nodes send missing values left, which the NaN → right traversal could not reproduce); `model_xgb.onnx`, replayed through ONNX Runtime on 100k test tracks (+ single-thread latency).
- **Records**: `train.log`, `thresholds.json` (working points, test metrics, size curve, feature ABI, .bin md5), `manifest.json` (git commit and dirty flag, package versions, GPU, input files, cache key, verification results, output checksums), `cmssw_cfi_snippet.py`. `thresholds.json`, `manifest.json` and `train.log` are version-controlled.
- **Logs**: warnings are shown once and must not occur (`check_consistency.py` fails on any warning/error line). ONNX Runtime sessions use explicit thread pools: the default pool pinned threads to all 192 physical cores and failed inside the job's 94-CPU cpuset, printing 144 `pthread_setaffinity_np failed` errors per session in the v2 logs. The DNN trainers pin the TorchScript ONNX exporter (opset 13, `dynamo=False`; torch ≥ 2.9 switches the default) and verify every export with ONNX Runtime.

### Feature sets

- **IO, 33 features** (`io_production_features(prefix)`, same list for the pixel and seeds chains, CMSSW `muonhp::IOTrackFeatures`): 18 track features (`p`, `pt`, `etaErr`, `phiErr`, `dzErr`, `qoverpErr`, `lambdaErr`, `nPixelHits`, `nTrkLays`, `nFoundHits`, `impact3D`, `sigmaPtOverPt`, `sip2D`, `sipZ`, `dxyOverPt`, `ptErrOverP`, `dzOverDxy`, `absEta`; momenta, errors and impact parameters log-compressed), 14 L1TkMuon features (stub counts `L1TkMu_nStubs`/`_Endcap`/`_Barrel`, `stubQual_max`, the η/φ/depth region of the best stub, and the track-to-L1TkMuon matching `hasMatch`, `dR2min`, `dPtNorm`, `chi2Pt`, `matchingScore`, `nCompatible`, `secondBest_dR2`) and the soft low-pT indicator `is_low_pt` (logistic in pT around 5 GeV). Dropped from 44: zero gain (`nLostHits`, `hitEfficiency`), near-zero gain (`normalizedChi2`, `chi2PerHit`, `chi2`, `impactSignificance`, `dxyErr`, `dszErr`), redundant (`eta` with `absEta`; `ptErr`, `relUncertaintyProduct` with `qoverpErr`).
- **OI, 22 features** (`OI_PRODUCTION_FEATURES`, `muonhp::OITrackFeatures`): 19 track features (`p`, `pt`, `normalizedChi2`, `etaErr`, `phiErr`, `dszErr`, `dxyErr`, `dzErr`, `qoverpErr`, `lambdaErr`, `eta`, `nPixelHits`, `nTrkLays`, `nFoundHits`, `nLostHits`, `impact3D`, `impactSignificance`, `chi2PerHit`, `hitEfficiency`), the L2 standalone-muon matching (`l2_mu_vtx_hasMatch`, `l2_mu_vtx_matchingScore`) and `is_low_pt`. Dropped from 26: `ptErr`, `sigmaPtOverPt`, `relUncertaintyProduct` (pT-uncertainty information of `qoverpErr`), `chi2` (with `normalizedChi2`/`chi2PerHit`).
- **What the v3 forests use** (gain, `feature_importance.txt` of each model): the IO selectors are driven by the L1TkMuon muon-station stubs (`L1TkMu_nStubs` and `L1TkMu_nStubs_Endcap` rank first and second on both chains, with 4.6–11x the gain of the third feature), followed by `is_low_pt`, the track pT, the best stub quality and the L1TkMuon ambiguity (`nCompatible`, `secondBest_dR2`); the OI selectors by the hit pattern (`nTrkLays`, `hitEfficiency`, `nPixelHits`: the OI general forest puts 8x more gain on `nPixelHits` than on any other feature) and the 3D impact parameter.

### Hyperparameter scan (validation only, `production/tuning/`)

Each configuration is trained on the non-test folds except the validation fold and scored by the production criterion (validation F2 at validation-chosen per-bin working points) for every tree prefix, together with the CMSSW inference cost of that prefix, so that configurations are compared at **equal inference cost**. 18 configurations per flavour (depth 6–14, learning rate 0.05–0.2, `min_child_weight`, L2, `colsample`, 512 bins, signal boost, kinematic weights); the chosen configurations were confirmed on a second validation fold (fold 9). Validation F2 within an inference budget:

| Flavour | Configuration | F2 @ ≤ 5k visits | F2 @ ≤ 15k visits | Chosen size (trees / visits / val F2) | Fold 9 (chosen size) |
|---|---|---|---|---|---|
| IO pixel | v2 config (depth 6, η 0.1) | 0.98732 | 0.98982 | 3150 / 18,881 / 0.99032 | 0.98934 |
| IO pixel | **v3 (depth 12, η 0.2)** | 0.99219 | 0.99347 | 800 / 9,268 / 0.99304 | 0.99218 |
| IO seeds | v2 config (depth 6, η 0.1) | 0.97652 | 0.98111 | 4000 / 23,982 / 0.98303 | 0.98200 |
| IO seeds | **v3 (depth 12, η 0.2)** | 0.98748 | 0.99061 | 1250 / 14,684 / 0.99061 | 0.99006 |
| OI pixel | v2 config (depth 6, η 0.1) | 0.98570 | 0.98683 | 1050 / 6,150 / 0.98611 | 0.98424 |
| OI pixel | **v3 (depth 10, η 0.1)** | 0.98712 | 0.98724 | 300 / 2,521 / 0.98646 | 0.98462 |
| OI general | v2 config (depth 6, η 0.1) | 0.99123 | 0.99214 | 850 / 5,026 / 0.99135 | 0.99066 |
| OI general | **v3 (depth 10, η 0.1)** | 0.99226 | 0.99236 | 200 / 1,704 / 0.99149 | 0.99079 |

Depth is the dominant factor (the depth-6 forests were under-fitted within their budget: their validation F2 still rose at 5000 trees); lower learning rates, `min_child_weight`, L2 and `colsample` changes do not help; 512 bins and removing the signal boost change F2 by ≤ 0.0003 (noise level). For the OI selectors all configurations agree within ±0.0005 (their validation samples hold ~20k/60k signal tracks); depth 10 is marginally but consistently ahead of depth 6 on both folds at a third of the cost (depth 12 / η 0.2 collapses to ~100 trees for no gain). Full tables: `production/results/tuning_summary.md`.

## Feature parity between training and CMSSW

The training features are computed from the n-tuple branches (float32), the CMSSW features from `reco::Track`/`l1t::TrackerMuon` objects. Both now follow one **numeric convention**: every raw input rounded to float32 (as the n-tuple stores it), all arithmetic in float64 (including an exact `reco::deltaPhi` equivalent in Python), each feature rounded to float32 once (`pixel_features.as_float64`, `muonhp::asStored` in the C++ headers).

**Bug found and fixed (v2 IO selectors, both chains)**: the imputed constants of the L1-matching features were evaluated in float64 in the training (`log10(1 + 1e-6)` = 4.342943e-07 for `L1TkMu_secondBest_dR2` when there is no second compatible L1TkMuon) and in float in CMSSW (`1.0f + 1e-6f` rounds: 4.141753e-07). The forest put a split exactly between the two values, used by 177 nodes of the deployed pixel model: every track without a second compatible L1TkMuon was scored on the wrong side of all of them in CMSSW. On 100 ZMM events, 11/402 (pixel) and 11/684 (seeds) IO selection decisions differed from the validated model, with score shifts up to 0.58. The OI selectors were not affected (raw-space imputation).

**Validation on real events** (`production/features_validation/run_cmssw_validation.sh`: L1 + HLT + training NANO on 250 ZMM and 250 BsToMuMu RelVal events at 200 PU per chain, both chains; `validate_cmssw.py` joins the per-track C++ dump with the training extraction on the same events): for the four deployed selectors the CMSSW features are **bit-identical** to the training features (7,627 tracks, every value) and **0 selection decisions differ**; scores agree to ≤ 1.2e-7 (float summation order). Results: `production/features_validation/results/parity_v3_*.json`. The same check on the v2 deployment (before the fix, 100 ZMM events) found 11/402 (IO pixel) and 11/684 (IO seeds) differing decisions.

## CMSSW integration

All four selectors live in `RecoMuon/L3TrackFinder` on the [`20_1_X_muonTracking` branch](https://github.com/Parsifal-2045/cmssw/tree/20_1_X_muonTracking) of the private Parsifal-2045/cmssw fork.

- **Models**: `RecoMuon/L3TrackFinder/data/IO/muonHP_IO_{pixelPath,seedsPath}_forest_v3.bin` and `data/OI/muonHP_OI_{pixelPath,seedsPath}_forest_v3.bin` (family / HLT chain / version), described in `data/README.md` (selector, chain, input tracks, cfi module, trees, test performance, training commit, md5). No other model files remain in `data/`. For the upstream PR they belong in cms-data (`RecoMuon-L3TrackFinder`).
- **Features**: `interface/IOTrackSelectorFeatures.h` (33, `muonhp::IOTrackFeatures`) and `interface/OITrackSelectorFeatures.h` (22, `muonhp::OITrackFeatures`), float64 convention above.
- **Inference**: `interface/CompactForest.h` (shared by both plugins): loads the `.bin` once per process (`GlobalCache`), re-lays it out for cache locality and validates it against the extractor (feature indices inside the feature vector, children after parents, exact file size) — loading a model trained on another feature set now fails with a clear error instead of reading out of bounds; `BinnedWorkingPoints` holds the per-pT-bin thresholds.
- **Plugins**: `MuonIOTracksForestSelector` (pixel and seeds IO), `MuonOITracksForestSelector` (both OI chains): extract the features of all tracks, score them in one batched call, output the selected tracks and the scores of all input tracks; `modelPath` is a required `FileInPath`; `nFeatures` is checked once against the extractor; inputs are read with `iEvent.get` (a missing product is an error); debugging output via MessageLogger (the IO plugin printed one line per event to stdout); `dumpFeatures` prints one `MUONHP_FEATURES,<label>,<run>,<lumi>,<event>,<track>,<features>,<score>` line per track for the cross-check.
- **cfis**: `hltPhase2MuonPixelTracksHighPurityForest_cfi.py`, `hltPhase2MuonIOTrackSelectionHighPurityForest_cfi.py`, `hltPhase2L3OIMuonTrackSelectionHighPurityForest_cfi.py`; model path and working points are written from the trainings' `thresholds.json` by `production/deploy_cmssw.py` and checked by `production/check_consistency.py --cmssw-src`.
- **Wiring**: the process modifiers `phase2MuonPixelTracksSelector` / `phase2MuonSeedsSelector` (and `ngtScouting`, pixel chain) swap the forests into `HLTPhase2MuonPixelTracksFromL1TkSequence` (IO pixel), `HLTPhase2L3MuonsIOSequence` (IO seeds) and `hltPhase2L3OIMuonTrackSelectionHighPurity` (OI). Fixed: the seeds-chain n-tuple table, HLT track validation and associator still referenced `hltPhase2MuonIOTrackSelectionHighPurity`, removed with the DNN selector (now `...HighPurityForest`).
- **Tests**: `test/forestSelectorsSmoke_cfg.py` constructs the four selectors and loads/validates their forests; the per-track behaviour on real events is checked by `production/features_validation/validate_cmssw.py`.

### Behaviour in CMSSW

**Inference code.** `CompactForest` re-lays every tree out depth-first as packed 8-byte nodes at load time (the file format is unchanged) and the plugins score all tracks of an event in one call, tree-major (a tree's nodes are reused across the event's tracks while in cache). Scores are bit-identical to the plain traversal (checked on 7,256 tracks between runs). Single-thread cost per track on real feature vectors (`features_validation/forest_bench.cc`):

| Model | File layout | Packed | Packed + batched (8 tracks) |
|---|---|---|---|
| IO pixel v2 (3250 trees, depth 6) | 122 µs | 136 µs | 81 µs |
| IO pixel v3 (800 trees, depth 12) | 145 µs | 91 µs | 65 µs |
| IO seeds v2 (4050 trees, depth 6) | 156 µs | 172 µs | 96 µs |
| IO seeds v3 (1250 trees, depth 12) | 231 µs | 167 µs | 103 µs |
| OI pixel v3 (300 trees, depth 10) | 24 µs | 17 µs | 15 µs |

**Time per event in the HLT jobs** (TimeReport, same events, 12-thread jobs sharing the node; ms/event, ZMM / BsToMuMu; the full event takes ~2.0 s):

| Selector | v2, previous code | v3, previous code | v2, new code | **v3, new code (deployed)** |
|---|---|---|---|---|
| IO pixel | 1.05 / 1.03 | 2.14 / 2.21 | 0.69 / 0.68 | **1.10 / 1.18** |
| IO seeds | 1.56 / 1.98 | 4.57 / 5.10 | 1.01 / 1.12 | **2.16 / 2.45** |
| OI (pixel chain) | 0.11 / 0.12 | 0.13 / 0.15 | 0.10 / 0.11 | **0.12 / 0.13** |
| OI (seeds chain) | 0.08 / 0.10 | 0.10 / 0.11 | 0.08 / 0.09 | **0.09 / 0.10** |

The deep IO forests visit fewer nodes but have 3–4x more of them (16/25 MB), so they stay memory-bound: in the busy multi-threaded job the deployed v3 IO selectors cost about as much as the v2 ones did with the previous code (+0.05–0.15 ms pixel, +0.5–0.6 ms seeds per event, ≤ 0.03% of the event time) and ~1.6–2.2x the v2 models on the new code. Results: `features_validation/results/timing_forest_selectors.json`.

**Out-of-sample check (RelVal).** On the same 500 RelVal events per chain (tracks with MC truth from the training NANO), every model separates signal from fakes much worse than on the training test split (IO ROC-AUC 0.96–0.975 vs 0.999+), including the BsToMuMu RelVal against the training's own Bs→μμ files (0.965 vs 0.998): the April–May 2026 training n-tuples are not representative of the current HLT reconstruction on these samples. Between models the differences are within the model-to-model variance of this small sample (retraining v2's own configuration moves the fakes kept at fixed efficiency by ~30% for the seeds selector): at v2's efficiency, fakes/event are v2 0.51 / v3 0.46 (pixel) and v2 0.48 / v3 0.63 (seeds). At their deployed working points the v3 IO selectors sit at higher muon efficiency (pixel 0.981 vs 0.972, seeds 0.934 vs 0.916) with more fakes (0.68 vs 0.54, 0.95 vs 0.50 per event): the per-bin set points are derived in the training domain and shift outside it. Results: `features_validation/results/relval_model_comparison_io_*.json`; `compare_models_relval.py` reruns it on any NANO.

## L1TkMuon input selectors

The HP forests only see the tracks the L1TkMuon-based selectors keep: `MuonTracksSelectorFromL1TkMuon` (`hltPhase2MuonPixelTracks`, best pixel track per L1TkMuon by a dR/dz/curvature score) and `MuonSeedsSelectorFromL1TkMuon` (`hltPhase2MuonIOTrackSeeds`, LST seeds around each L1TkMuon, arbitrated per L1TkMuon). `production/input_selectors/` measures, with MC truth on the selector inputs, the fraction of muons (TrackingParticles, pT > 2 GeV, |η| < 2.4, reconstructed in the input collection and having their own L1TkMuon) that each selection keeps.

**Pixel-track selector: no losses.** Every reachable muon keeps its pixel track (403/403 ZMM, 371/371 BsToMuMu, 250 events each); keeping the 2 or 3 best tracks per L1TkMuon (`nTracksToKeep`) recovers nothing and adds 2–3 pixel tracks (0.2–0.3 fakes after the HP forest) per event, so `nTracksToKeep = 1` is right. Apparent losses with a loose dR < 0.3 cone are muons without their own L1TkMuon (the nearest candidate belongs to another particle: 2–14 cm apart in z). Cleanup: `trackMinPt`/`trackMaxEta` were read but never applied (0.06% of the tracks, of which 584 genuine muons in the training sample, would fall below 0.9 GeV); removed from the plugin and the cfi.

**Seed selector: two bugs fixed.**
1. *Seed direction at the wrong place.* The seed pT/η/φ were taken from `seed.startingState()`, which sits on the outermost seed hit (up to R ≈ 1 m for LST seeds), while the L1TkMuon direction is a vertex quantity: at 2–3 GeV the bending rotates φ by ~0.2–0.3 rad, the genuine seed loses the dR-driven arbitration to unrelated seeds (and leaves the 0.4 cone below ~1.6 GeV). The seed is now taken at its closest approach to the beam line (`TSCBLBuilderNoMaterial`, new `beamSpot` parameter).
2. *Arbitration gated on the L1TkMuon pT.* Seeds above `maxPtForCompatibilityCheck` (25 GeV) are kept unconditionally, but the arbitration of the others only ran for L1TkMuons below 25 GeV: the genuine seeds of a higher-pT L1TkMuon measured just below 25 GeV were dropped. The arbitration now runs for every L1TkMuon (it can only add seeds; rare in these samples).

Muons kept (reachable = reconstructed by LST, own L1TkMuon; 250 events per sample):

| | ZMM seeds (2–5 GeV) | ZMM IO tracks | BsToMuMu seeds (2–5 GeV) | BsToMuMu IO tracks | Seeds / fakes per event (ZMM, Bs) |
|---|---|---|---|---|---|
| Before | 0.971 (0.739) | 0.968 | 0.924 (0.866) | 0.924 | 7.05 / 5.43, 7.95 / 6.48 |
| Threshold fix | 0.971 (0.739) | 0.968 | 0.924 (0.866) | 0.924 | 7.06 / 5.44, 7.95 / 6.48 |
| **Both fixes** | **0.998 (1.000)** | **0.995** | **1.000 (1.000)** | **1.000** | 7.09 / 5.38, 7.88 / 6.27 |

Keeping 3–4 seeds per L1TkMuon (instead of 2) recovered only half of the losses before the fix and is unnecessary after it. End to end (IO tracks after the HP forest, all fixes, v3 models): pixel chain 0.980 (ZMM) / 0.987 (Bs), seeds chain 0.949 / 0.956; the remaining losses are 2–5 GeV muons rejected by the HP forests (mostly non-prompt muons of the RelVal events, see the out-of-sample check). The IO seeds forest was trained on n-tuples produced with the old seed selector. Results: `production/input_selectors/results/`.

## OI feature extraction and the CMSSW parity fix (v2 campaign)

The OI selectors extract 26 features (22 kept): 12 log-compressed track/branches + 5 plain + 6 derived + 2 standalone-matching features (`l2_mu_vtx_hasMatch`, `l2_mu_vtx_matchingScore`) + the soft `is_low_pt`. The training-side extraction (`production/oi/OI_features.py`) mirrors the C++ extractor feature-for-feature, which the legacy OI DNN training (`tests/OI_*_model.py`) did **not**: wrapped track-to-standalone Δφ, matching score (χ²η + χ²φ + χ²pT + χ²dz)/9, no-match imputation 10.0 in raw space.

## Decisions and rationale

**Training protocol**
- *Muon guns with pileup, plus Bs→μμ.* Particle-gun muons over the full HLT pT spectrum keep the selection independent of any specific physics process; pileup mixing provides the non-isolated environment and its fakes; Bs→μμ adds mildly displaced muons. Physics samples such as tt̄ or DY are deliberately not part of the training.
- *Event-level split, test used once.* Tracks of one event share pileup, vertex and duplicates of the same muon, so a track-level split leaks; evt10 assigns whole events. Forest size, working points and hyperparameters are all chosen on validation; the test split only reports.
- *F2 per-pT-bin working points.* F2 weighs recall 4x over precision: a muon track lost by the HP selection is not recovered downstream, while a kept fake costs rate and time. The fake/muon ratio at the input varies by more than three orders of magnitude over pT (IO: 5–11 at 2–5 GeV, ≤ 0.06 above 10 GeV), so a single threshold is too loose at low pT or too tight at high pT. Bins with < 100 validation muons or no background use the global F2 threshold rather than a noisy per-bin optimum.
- *Tree budget + validation prefix selection, no early stopping.* The full 5000-tree size curve is scored at validation-chosen working points (the production criterion), and the smallest prefix within 0.001 of the best is kept. 0.001 is below the fold-to-fold spread of the validation F2 (0.0006–0.0018 between folds 3 and 9 at the chosen sizes), so the extra trees would buy no measurable performance while costing inference time; the chosen sizes (200–1250 trees) are far below the budget.
- *Deep forests* (IO: depth 12 / η 0.2, OI: depth 10 / η 0.1): best validation F2 at equal CMSSW inference cost on two folds (scan above).
- *Host-side quantile sketch.* The GPU sketch of weighted data differs between processes (the v2 models could not be re-created from their own configuration); the `QuantileDMatrix` sketch is computed on the host and the trees are still grown on the GPU, so every output is reproducible bit for bit.

**Training/CMSSW agreement**
- *float64 feature convention.* Inputs rounded to float32 (as the n-tuples store them), arithmetic in float64, one rounding per feature: numpy computes in float64 by default, and making the C++ follow it is exact and simple, while float32 arithmetic would need op-by-op agreement between numpy and libm. The convention removed the v2 IO mismatch (177 nodes split between the float and double values of an imputed constant).
- *One feature ABI.* The feature order is asserted name by name against the C++ extractors at every training, and the plugins validate every `.bin` against the extractor when they load it.
- *Compact `.bin` format, fp32.* No ONNX Runtime in the HLT path, 2.5x smaller than the ONNX serialisation, traversal replayed in Python at every training (0 decision flips) and on real events; fp16 leaf values accumulate ~0.2 score error over thousands of trees. NaN goes right (the exporter refuses default-left nodes, so the traversal needs no missing-value flags). ONNX is kept as a verified secondary export.
- *Packed, batched inference.* The deep v3 forests have 3–4x more nodes than v2; the packed depth-first layout and tree-major batching keep their HLT cost at the level of the v2 models on the previous code (Behaviour in CMSSW).

**Model choice**
- *Keep v3 despite the RelVal check.* On the test split v3 keeps 31% (pixel) / 50% (seeds) fewer fakes at higher recall, on 2.8M / 5.7M test tracks. On the 500 RelVal events all models, whatever their depth, rank signal and fakes equally within the reach of the sample. Fakes/event at v2's efficiency:

  | IO selector | v2 | depth 6 (v2 configuration, retrained) | depth 8 | depth 10 | v3 (depth 12) |
  |---|---|---|---|---|---|
  | pixel | 0.51 | 0.46 | 0.47 | 0.44 | 0.46 |
  | seeds | 0.48 | 0.61 | 0.57 | 0.59 | 0.63 |

  The v2 configuration retrained is as far from v2 as v3 is. The large drop of every model on RelVal (ROC-AUC 0.96–0.975 vs 0.999+) is a training-sample issue that no depth choice fixes: re-evaluate the depth on the new n-tuples with a larger out-of-sample check.
- *IO vs OI configurations.* The OI validation F2 is flat across the scan (±0.0005); depth 10 / η 0.1 is marginally but consistently ahead of depth 6 on both folds at a third of its inference cost, and depth 12 / η 0.2 collapses to ~100 trees for no gain.

**Inputs of the HP forests**
- *`nTracksToKeep = 1`* for the pixel-track selector: every reachable muon is already kept, and 2–3 tracks per L1TkMuon only add fakes.
- *Two seeds per L1TkMuon* for the seed selector: after the direction and arbitration fixes it keeps 0.998–1.000 of the reachable muons; more seeds per L1TkMuon only recovered half of the losses before the fixes and are unnecessary after them.

**Deployment**
- *Model files* in `RecoMuon/L3TrackFinder/data/IO` and `/OI`, named by family, HLT chain and version (`muonHP_<IO|OI>_<pixelPath|seedsPath>_forest_<version>.bin`). A deployed retraining always gets a new version (`model_version`): files in cms-data are immutable and a changed model under an old name would silently change results. `deploy_cmssw.py` writes the cfi working points from `thresholds.json` (never by hand), and `check_consistency.py --cmssw-src` verifies files, checksums and cfi values.
- *No machine-specific paths in code.* Every input is a command-line flag (How to run the training chain) rather than a path in the code or an environment variable; the records (`manifest.json`, `train.log`) keep the absolute input paths as provenance.
- *DNN exports* pin the TorchScript ONNX exporter (`dynamo=False`, opset 13): torch ≥ 2.9 switches the default to the dynamo exporter.

## Possible improvements

**Models**
- **Training samples (largest open item).** On current RelVal events all models separate genuine muons from fakes much worse than on their own test split (IO ROC-AUC 0.96–0.975 vs 0.999+). The drop is as large for the BsToMuMu RelVal against the training's own Bs→μμ files (0.965 vs 0.998), i.e. for the same process, so it points to the reconstruction having changed since the April–May 2026 n-tuples rather than to the sample composition: re-produce the gun + Bs→μμ n-tuples with the current HLT configuration (including the fixed seed selector, whose recovered low-pT muons are absent from the v3 seeds training sample), retrain (one command per flavour) and re-derive the working points, then choose the forest depth on an out-of-sample check with more than the 500 events used here. If the gap persists with current n-tuples, compare RelVal and training tracks feature by feature.
- The IO HP forests reject a sizeable fraction of the low-pT (2–5 GeV) muons of the RelVal events (mostly non-prompt), more than of the gun/Bs→μμ training muons: same remedy.
- Per-η (or η×pT) working points (the machinery — bin edges + thresholds — exists on both sides).
- Probability calibration (Platt/isotonic) if calibrated scores are needed; otherwise publish an efficiency-vs-threshold table alongside the F2 points.
- The `duplicate` flag (currently unused) as an auxiliary target or third class.
- The C++ OI `hasMatch` feature is ~always 1 in events containing any standalone muon (bestScore < 25 over the best candidate); a track-level compatibility window would make it informative.
- The v3 IO `.bin` files are 16/25 MB (deep trees): if file size matters for cms-data, `max_leaves` or a node budget in the size selection would shrink them.

**Pipeline**
- Redo the 44→33 IO pruning with the nested-split protocol of the scan (it was selected on the data that measured it).
- Fix the misnamed `impact3D` (log₁₀(dxy²+dz²) of the *squared* 3D impact parameter; renaming changes the feature ABI on both sides).
- Revive the DNN backend only after retraining on the deployed 33/22-feature sets and the float64 convention.
