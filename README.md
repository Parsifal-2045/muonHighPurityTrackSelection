# Selector for muon tracks in the Pixel detector of CMS
Machine-learning high-purity selections for muon tracks at the Phase-2 CMS High Level Trigger. The muon HLT reconstruction builds its inner tracks from pixel tracks, or from tracks seeded by LST, matched to L1 tracker muons, without a dedicated tracking iteration. The selectors of this repository reject the fake tracks among them (tracks not matched to a simulated muon), in place of the cut-based selection applied to general tracks, and do the same for the outside-in tracks.

The repository holds the training chain of the four selectors deployed in CMSSW (XGBoost forests), the models with their training records, and the tools that deploy them to a CMSSW area and validate them there.

**Contents:** [Repository structure](#repository-structure) · **[How to run the training chain](#how-to-run-the-training-chain)** · [Deployed models (v1)](#deployed-models-v1-production) · [Validation in CMSSW (v1)](#validation-in-cmssw-v1) · [Forest training pipeline](#forest-training-pipeline-productionforest_pipelinepy) · [Feature parity between training and CMSSW](#feature-parity-between-training-and-cmssw) · [CMSSW integration](#cmssw-integration) · [Decisions and rationale](#decisions-and-rationale)

## Repository structure

Production code and models (the **XGBoost forests are the production architecture**; DNNs are kept as the alternative backend, currently not wired in CMSSW):

- `production/forest_pipeline.py` — the shared training pipeline of the four forests (features and cache, split, training, size selection, working points, test evaluation, verified exports, provenance records). `production/pixel_xgb.py`, `production/seeds_xgb.py` (with `production/io_xgb_common.py`: features, recipe and working points of the two IO forests), `production/oi/OI_pixel_xgb.py`, `production/oi/OI_general_xgb.py` only hold the flavour configuration.
- `production/pixel_features.py` — IO feature extraction (parametrised by the n-tuple branch prefix: pixel and seeds selectors share it), the production feature ABI (`io_production_features()`), evaluation helpers. `production/chamber_match.py` — the matching of a track to the muon chambers (the candidates of every muon station and their raw quantities), used by the IO extraction. `production/oi/OI_features.py` — the OI counterpart (`OI_PRODUCTION_FEATURES`).
- `production/onnx_utils.py` — ONNX Runtime sessions with explicit thread pools, the number of CPUs the process may use, pinned TorchScript exporter for the DNNs.
- `production/io/{pixel,seeds}/`, `production/oi/{pixel,general}/` — the deployed (v1) models with their records. Version-controlled: `model_compact.bin` (the file deployed to CMSSW), `train.log`, `thresholds.json`/`.txt`, `manifest.json` (git commit, packages, device, input files, cache key, verification results), `cmssw_cfi_snippet.py` and `feature_importance.txt`. A training also writes `model.json`, `model_xgb.onnx` and the plots, which are not version-controlled: `check_consistency.py` and the tools that score a model (`validate_cmssw.py`, `compare_models_relval.py`, `analyze_input_selectors.py`) read `model.json`, so they need a model directory as a training leaves it. `production/io/archive/` and `production/oi/archive/` hold the reference generations the comparisons are made against, dev5 and dev6 (`model_compact.bin`, working points, feature importances).
- `production/tuning/` — quick validation-only hyperparameter scan (`forest_scan.py` writes its results to `tuning/results/`, `summarize_scan.py` tabulates them).
- `production/check_consistency.py` — checks training records, exports and (with `--cmssw-src`) the CMSSW deployment against each other. `production/deploy_cmssw.py` — installs the models in a CMSSW area (data files, cfi working points, data README). `production/make_summary.py` — result tables of the production models from their records (`production/results/`).
- `production/features_validation/` — CMSSW/Python cross-check on real events (`validate_cmssw.py`, timing customisation `forest_timing_cff.py`), comparison of model generations on independent events (`compare_models_relval.py`), L3 muons at the end of the chain (`l3_muon_tables_cff.py`, `count_l3_muons.py`); results in `features_validation/results/`.
- `production/input_selectors/` — efficiency study of the L1TkMuon input selectors (cmsRun customisation `input_selector_study_cff.py`, analysis `analyze_input_selectors.py`, results in `input_selectors/results/`).
- `production/dnn/` — the (currently unwired) DNN trainers and architecture benchmark.
- `tests/` — exploratory scripts, not part of the production chain.

## How to run the training chain

Every input is given on the command line, with no environment variables to set and no machine-specific paths in the scripts (`python <script> --help` lists all options). Commands are shown from `production/`; the scripts work from any directory.

### 1. Setup

- Python ≥ 3.9 with the packages of `production/requirements.txt` (e.g. `python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt`). A CUDA GPU is used when available, the CPU otherwise (announced in the log; `--device cpu` forces it).
- The v1 models were trained on CPU with xgboost 2.1.4, numpy 2.0.2, awkward 2.8.12, uproot 5.6.9, scikit-learn 1.6.1, onnx 1.19.1, onnxruntime 1.19.2, onnxmltools 1.16.0 (recorded in every `manifest.json`). A re-run with the same versions and inputs is not guaranteed to reproduce the models bit for bit. They were trained with the code of commit `3814083` before it was committed: the manifests, the cfi comments and the CMSSW data README therefore name its parent `e622949`, with `git_dirty: true` in the manifests.
- The CMSSW steps (6 and 7) need a CMSSW area with the `RecoMuon/L3TrackFinder` integration (see CMSSW integration), built and set up with `cmsenv`.

### 2. Inputs

The training n-tuples are the `NANO:@MUHLTTraining` output of each HLT chain, one directory per chain: every `*.root` file of the directory is read, one file per sample. File names carry no meaning to the pipeline. The IO and OI models of a chain read the same files (different branches). The IO selectors need the tables of the muon local reconstruction (DT, CSC and GEM segments, GEM and RPC rec-hits), filled, with the global coordinates of the RPC rec-hits (`rpcRecHit_g_eta`, `rpcRecHit_g_phi`).

| Flag | Script | Value |
|---|---|---|
| `--data-dir DIR` (required) | `pixel_xgb.py`, `oi/OI_pixel_xgb.py`, `dnn/pixel_model.py` | n-tuples of the pixel-track chain (process modifier `phase2MuonPixelTracksSelector`) |
| `--data-dir DIR` (required) | `seeds_xgb.py`, `oi/OI_general_xgb.py`, `dnn/seeds_model.py` | n-tuples of the seeds chain (process modifier `phase2MuonSeedsSelector`) |
| `--wp-files FILE [FILE ...]` | `pixel_xgb.py`, `seeds_xgb.py` | n-tuples of independent events of the same chain (e.g. RelVal samples), on which the working points are pinned (step 4) |
| `--data-dir DIR` (required) | `tuning/forest_scan.py` | n-tuples of the scanned flavour's chain |
| `--cache-dir DIR` | forest trainings, `tuning/forest_scan.py` | feature cache (default `/tmp/$USER/muonhp_feature_cache`; 2.1 / 3.7 / 0.4 / 0.4 GB for IO pixel / IO seeds / OI pixel / OI general) |
| `--cmssw-src DIR`, `--out DIR` (required), `--relval-zmm DIR`, `--relval-bs DIR` | `features_validation/run_cmssw_validation.sh` | CMSSW `src/`, output directory, GEN-SIM-DIGI-RAW RelVal directories (default: the CMSSW_20_0_0_pre1 PU200 RelVals on EOS) |
| `--data-dir DIR` (required) | `tests/*_model.py`, `io/archive/test/*_model.py` | n-tuples for the exploratory scripts |

The commands below use two shell variables for the n-tuple directories:
```bash
PIXEL=/path/to/pixel-chain/ntuples
SEEDS=/path/to/seeds-chain/ntuples
```

Before training on a new production, check that the L1 tracker muons are there: the fraction of generated muons with an L1 tracker muon within ΔR < 0.1, per |η| region (Forest training pipeline, Data).

### 3. Start a new model version (only when retraining for deployment)

Skip this step to reproduce the deployed v1 models. A retraining meant for CMSSW gets a new version: the new models are compared with the deployed ones, and they get new file names (cms-data files are immutable). Archive the deployed models:
```bash
cp -a io/pixel io/archive/pixel_xgb_output_46f_v1; cp -a io/seeds io/archive/seeds_xgb_output_46f_v1
cp -a oi/pixel oi/archive/pixel_22f_v1;            cp -a oi/general oi/archive/general_22f_v1
```
then set `model_version` to the new label and `ref_version` to the deployed one (`"v1"`) in `forest_pipeline.DEFAULTS`. This is the only place holding the version: it fills the deployed file names (`cmssw_bin`) and the reference model of the OI reports (`ref_model_path`).

### 4. Train the four forests

```bash
python pixel_xgb.py         --data-dir $PIXEL --force --wp-files $WP_PIXEL   # IO pixel   -> io/pixel/
python seeds_xgb.py         --data-dir $SEEDS --force --wp-files $WP_SEEDS   # IO seeds   -> io/seeds/
python oi/OI_pixel_xgb.py   --data-dir $PIXEL --force                        # OI pixel   -> oi/pixel/
python oi/OI_general_xgb.py --data-dir $SEEDS --force                        # OI general -> oi/general/
python check_consistency.py
```
`$WP_PIXEL` and `$WP_SEEDS` are the n-tuples of independent events of each chain on which the IO working points are pinned (Forest training pipeline, Working points); without `--wp-files` the working points come from the validation split of the training sample. They must not be the events used to validate the result. For v1 they are 14,500 pre3 D128 RelVal events per chain (ZMM, BsToMuMu, TTbar at 200 PU), with the L1 tracker muons re-emulated (step 7).

The v1 trainings took 24 / 24 / 9 / 9 min (IO on 36 CPU threads, OI on 24), plus a one-off feature extraction for new n-tuples (about 35 min per IO chain on 9 processes, on a node busy with other jobs; later runs read the cache). Each writes its model directory: model, exports, working points, `train.log`, `thresholds.json`, `manifest.json` and `cmssw_cfi_snippet.py`. `check_consistency.py` re-exports every model from `model.json` and checks tree counts, checksums, the feature ABI, the exports and that the logs are free of warnings.

Other options: `--force` replaces an existing model directory (without it the training refuses to overwrite), `--output-dir DIR` writes elsewhere, `--set key=value` overrides a configuration value (e.g. `--set max_depth=8`, `--set nthread=44`, `--set feature_workers=14`), `--device cpu` forces the CPU. The feature extraction of the IO chains takes about 10 GB per process. Do not edit `pixel_features.py`, `chamber_match.py` or `oi/OI_features.py` while a training runs: the job stops (see the feature cache).

### 5. Re-check the hyperparameters (optional)

Recommended when the samples or the features change substantially. Train the flavour with the alternative values into a scratch directory and compare the test split of its records and the fakes kept on independent events (step 7):
```bash
python pixel_xgb.py --data-dir $PIXEL --output-dir $SCRATCH/io_pixel_d10 --set max_depth=10 --no-permutation-importance
python features_validation/compare_models_relval.py --family IO --prefix muon_pixel_tracks \
    --nano $RELVAL_NANO --models io/pixel $SCRATCH/io_pixel_d10
```
Decide on the independent events, at equal recall (thresholds giving every model the same fraction of muons kept; the working points of scratch trainings come from the training sample and are not at equal recall), and with more than one seed: two trainings that differ only by the seed differ by about 7% in fakes kept. The fake rejection of the copy-aware test split is not enough: it keeps improving with more trees and with importance-pruned feature lists that keep more fakes on independent events (Hyperparameters). Change `io_xgb_common.RECIPE` (IO) or `OI_BASE` (OI) only for a configuration clearly ahead there, at an acceptable inference cost, then retrain (step 4). `tuning/forest_scan.py` (`--list` shows its configurations) is a quicker scan on event-level validation folds: forests deeper than about 6 partly memorise the recurring pileup tracks of such folds, so use it for shallow forests only.

### 6. Deploy to CMSSW

```bash
python deploy_cmssw.py --cmssw-src $CMSSW_BASE/src
(cd $CMSSW_BASE/src && scram b)
python check_consistency.py --cmssw-src $CMSSW_BASE/src
```
`deploy_cmssw.py` copies the `.bin` files, writes the working points into the cfis, removes the previous model files and writes `RecoMuon/L3TrackFinder/data/README.md`; `check_consistency.py --cmssw-src` also checks the deployed files and cfi values.

### 7. Validate in CMSSW

Validate in the release family the training n-tuples were produced with (CMSSW_20_1_0_pre3 or later for v1: the L1 muon emulation differs between releases). `ZMM_D128` and `BS_D128` are the `GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D128_RegeneratedGS_PU_20260903_192421_RV19-v1/2830000` directories of `RelValZMM_14` and `RelValBsToMuMu_14TeV` under `/eos/cms/store/relval/CMSSW_20_1_0_pre3/`; the script reads the first three files of each.

**L1 tracker muons of the pre3 RelVals.** The jobs run `L1P2GT,HLT` on the stored L1 objects, and the files made with the L1 emulation of CMSSW_20_1_0_pre3 store an L1 tracker muon for only about 30% of the barrel muons ([cms-sw/cmssw#52096](https://github.com/cms-sw/cmssw/issues/52096); Forest training pipeline, Data). On such files the stub and tracker-muon producers have to run again at the start of the job, in an area where `L1Trigger/Phase2L1GMT` has the workaround built:
```python
from L1Trigger.Phase2L1GMT.gmtStubs_cfi import gmtStubs
from L1Trigger.Phase2L1GMT.gmtTkMuons_cfi import gmtTkMuons
process.l1tStubsGmt = gmtStubs.clone()
process.l1tTkMuonsGmt = gmtTkMuons.clone(srcStubs="l1tStubsGmt:tps")   # as in SimL1Emulator_cff
process.l1MuonReEmulation = cms.Path(process.l1tStubsGmt + process.l1tTkMuonsGmt)
process.schedule.insert(0, process.l1MuonReEmulation)
```
`run_cmssw_validation.sh` does not add these lines: append them to the `final_<chain>_<sample>_cfg.py` files it writes and run those again (`cmsRun final_<chain>_<sample>_cfg.py`). Every RelVal number of this README comes from jobs with the L1 tracker muons re-emulated this way; with them 97% of the ZMM muons above 10 GeV have an L1 tracker muon in the barrel and 96% in the endcaps (28% in the barrel with the stored objects).

L1 + HLT + training NANO on 250 ZMM and 250 BsToMuMu RelVal events per chain (four parallel jobs), then the track-by-track parity, the out-of-sample model comparison and the input-selector efficiencies (`OUT` = any scratch directory; replace `v1` by the new version and the reference models by those to compare with):
```bash
features_validation/run_cmssw_validation.sh --cmssw-src $CMSSW_BASE/src --out $OUT \
    --geometry ExtendedRun4D128 --era Phase2C26I13M9 --relval-zmm $ZMM_D128 --relval-bs $BS_D128
for s in ZMM Bs; do
  python features_validation/validate_cmssw.py --chain pixel --nano $OUT/final_pixel_$s.root --log $OUT/final_pixel_$s.log \
      --io-model io/pixel --oi-model oi/pixel --json features_validation/results/parity_v1_pixel_$s.json
  python features_validation/validate_cmssw.py --chain seeds --nano $OUT/final_seeds_$s.root --log $OUT/final_seeds_$s.log \
      --io-model io/seeds --oi-model oi/general --json features_validation/results/parity_v1_seeds_$s.json
done
python features_validation/compare_models_relval.py --family IO --prefix muon_pixel_tracks \
    --nano $OUT/final_pixel_{ZMM,Bs}.root \
    --models io/archive/pixel_xgb_output_33f_dev5 io/archive/pixel_xgb_output_46f_dev6 io/pixel \
    --json features_validation/results/relval_model_comparison_io_pixel_ZMMBs.json
python features_validation/compare_models_relval.py --family IO --prefix muon_general_tracks \
    --nano $OUT/final_seeds_{ZMM,Bs}.root \
    --models io/archive/seeds_xgb_output_33f_dev5 io/archive/seeds_xgb_output_46f_dev6 io/seeds \
    --json features_validation/results/relval_model_comparison_io_seeds_ZMMBs.json
python input_selectors/analyze_input_selectors.py --chain pixel --nano $OUT/final_pixel_{ZMM,Bs}.root \
    --label ZMM BsToMuMu --io-model io/pixel --json input_selectors/results/pixel_chain_v1.json
python input_selectors/analyze_input_selectors.py --chain seeds --nano $OUT/final_seeds_{ZMM,Bs}.root \
    --label ZMM BsToMuMu --io-model io/seeds --json input_selectors/results/seeds_chain_v1.json
```
`--models` lists the references first; every model is scored with its own feature list and at its own working points (`--family OI` does the same for the OI forests, on the `l3_tk_OI_*` tracks of the n-tuples). Expected: bit-identical features and 0 differing decisions for all four selectors (any difference means the training and CMSSW extractions diverged). The per-module times are in the TimeReport of `$OUT/final_*.log`. Other options of the validation script: `--events N`, `--threads N`, `--relval-zmm DIR`, `--relval-bs DIR`, and `--alt-models DIR` to time other models next to the deployed ones (see its `--help`).

**L3 muons at the end of the chain.** The selectors act on tracks; what reaches the trigger are the L3 muons after the muon identification. To count them, add `features_validation/l3_muon_tables_cff.py` to a job of the same kind (`process = customise_l3_muon_tables(process)`; with `pass_through=True` the selectors keep every track, the chain without a high-purity selection) and run
```bash
python features_validation/count_l3_muons.py --sample "v1 pixel" $OUT/v1_pixel_{ZMM,Bs}.root \
    --sample "no selection, pixel" $OUT/passall_pixel_{ZMM,Bs}.root --json features_validation/results/l3_muons_v1.json
```
on the same events processed with each configuration (fake muons per event, muons per event and efficiency for the generated muons, before and after the identification). Only the full chain gives these numbers: with fewer fake tracks, fewer genuine muons lose their segments to a fake in the arbitration of the muon identification and more standalone muons are passed to the outside-in step.

### 8. Summarise the results

```bash
python make_summary.py --json results/summary_v1.json
```
prints the result tables of this README from the training records of the four production models (per model and per working-point bin); update the result sections below with them.

## Deployed models (v1, `production/`)

Four track selectors cover the Phase-2 muon HLT chains. Inside-out (IO): **pixel tracks** (`hltPhase2MuonPixelTracks`, pixel-track chain, n-tuple `muon_pixel_tracks_*`) and **IO tracks from LST seeds** (`hltPhase2MuonIOTracks`, seeds chain, `muon_general_tracks_*`). Outside-in (OI, `hltPhase2L3OIMuCtfWithMaterialTracks`, `l3_tk_OI_*`, with L2 standalone-muon matching): **OI pixel** (trained on the pixel-chain sample) and **OI general** (seeds-chain sample).

The aim is a track-level stand-in for the muon identification: keep every genuine muon track and remove as many fakes as possible. The IO selectors combine track quantities with the matching of the track to the muon chambers (Feature sets). v1 (trained 2026-10-10) is evaluated in three ways: on the test split of the training samples, built so that a recurring pileup track never sits in both training and test (Forest training pipeline); on the tracks of independent RelVal events; and on the L3 muons at the end of the HLT chain run on the same events. Two earlier generations are the references: **dev5** (33 track and L1 features, no chamber matching) and **dev6** (the v1 features and recipe, trained and pinned on samples in which the L1 emulation lost most of the barrel muons; Forest training pipeline, Data).

**Working points**: one threshold per pT bin, keeping a fixed fraction of the muons of the bin, the same targets for the four selectors.

| pT bin [GeV] | 0–3 | 3–5 | 5–10 | 10–50 | 50–200 | > 200 |
|---|---|---|---|---|---|---|
| Muons kept (target) | 98% | 99% | 99.5% | 99.5% | 99.5% | 99.5% |
| IO thresholds set on | RelVal | RelVal | RelVal | RelVal | validation split | validation split |

For the IO selectors the target refers to the muons of the signal event, and the thresholds of the four bins below 50 GeV are pinned on 14,500 independent RelVal events per chain (7,000 ZMM, 7,000 BsToMuMu and 500 TTbar pre3 D128 events at 200 PU with the L1 tracker muons re-emulated, disjoint from the validation events below); the two bins above 50 GeV, which those events do not populate enough, and the OI selectors use the validation split of the training sample (gun and Bs muons). The files of both RelVal sets are listed in `production/features_validation/relval_d128_tuning_files.txt`.

**On the tracks of independent RelVal events** (validation events: 5,000 pre3 D128 ZMM + BsToMuMu events and 2,500 TTbar events at 200 PU per chain, L1 tracker muons re-emulated, tracks with MC truth; `features_validation/results/relval_model_comparison_*.json`). Fakes per event before and after the selector and fraction of muons kept, each model at its own working points, in the two samples (ZMM+BsToMuMu / TTbar):

| Selector | Input fakes / event | Fakes kept / event, v1 | dev6 | dev5 | Fake rejection, v1 | Muons kept, v1 | dev6 | dev5 |
|---|---|---|---|---|---|---|---|---|
| IO pixel | 2.69 / 2.79 | 0.26 / 0.30 | 0.27 / 0.31 | 1.30 / 1.42 | 0.90 / 0.89 | 0.991 / 0.987 | 0.989 / 0.986 | 0.997 / 0.998 |
| IO seeds | 5.82 / 5.94 | 0.34 / 0.39 | 0.38 / 0.43 | 3.08 / 3.43 | 0.94 / 0.93 | 0.992 / 0.990 | 0.990 / 0.981 | 0.996 / 0.998 |
| OI pixel | 2.11 / 2.21 | 0.130 / 0.165 | 0.164 / 0.193 | 0.100 / 0.124 | 0.94 / 0.93 | 0.994 / 0.989 | 0.994 / 0.989 | 0.984 / 0.949 |
| OI general | 2.02 / 2.13 | 0.115 / 0.134 | 0.190 / 0.222 | 0.108 / 0.136 | 0.94 / 0.94 | 0.990 / 0.979 | 0.990 / 1.000 | 0.977 / 0.950 |

(IO: muons of the signal event. OI: all muon tracks, on the OI tracks of the chain run with v1, which hold 487 / 175 muon tracks in the pixel chain and 397 / 141 in the seeds chain.)

- **IO against dev5**: 5 (pixel) and 9 (seeds) times fewer fakes, for 0.4–1.1 points fewer muons.
- **IO against dev6**: 4–10% fewer fakes and more muons. The difference is in the barrel: for |η| < 0.9, v1 keeps 0.993 / 0.995 (pixel) and 0.993 / 0.995 (seeds) of the signal-event muons, dev6 0.984 / 0.990 and 0.986 / 0.976. At equal recall the two generations keep the same fakes within the noise of a retraining (Hyperparameters).
- **Per pT bin** (signal-event muons kept by the IO selectors on ZMM+BsToMuMu, five bins up to 200 GeV): 0.978 / 0.990 / 0.995 / 0.993 / 0.992 (pixel) and 0.977 / 0.984 / 0.995 / 0.996 / 0.998 (seeds).
- **Non-isolated muons**: on TTbar the IO selectors keep 0.982 / 1.000 / 0.986 (pixel) and 0.973 / 1.000 / 0.986 (seeds) of the signal-event muons in the three bins between 3 and 50 GeV. The shortfall is outside the barrel: for |η| > 1.2 they keep 0.981 (pixel) and 0.982 (seeds), and the pixel selector 0.975 for 0.9 < |η| < 1.2, against 0.995 in the barrel. The pinning events are almost all ZMM and BsToMuMu.
- **Above 200 GeV, pixel chain**: in the 22,000 RelVal events (validation and pinning together) 23 signal-event muon tracks have a reconstructed pT above 200 GeV. For 19 of them the pixel-track pT is off by more than 30%, and v1 keeps 14 of those (dev6 18): the five lost are endcap tracks (|η| 1.8–2.2) of muons with a true pT of 38–69 GeV. The 4 well-measured ones are kept, and a single fake track sits in that bin. The threshold there (0.9998) is set on gun muons, whose reconstructed pT is mostly right; the seeds chain keeps 7 of 7.
- **OI**: on ZMM+BsToMuMu v1 keeps 20–40% fewer fakes than dev6 for the same muons (on TTbar, OI general keeps 138 of 141 muon tracks where dev6 keeps all), and more muons than dev5, whose low-pT target was 96% instead of 98%, for up to 30% more fakes. The OI forests of v1 were trained on n-tuples made with the dev6 IO selectors upstream; the OI tracks of the chain are the same with either (2.11 / 2.21 and 2.02 / 2.13 input fakes per event with v1, 2.10 / 2.19 and 1.99 / 2.09 with dev6).

**On the L3 muons at the end of the chain** (the same validation events through the full HLT chain, `features_validation/count_l3_muons.py`, `features_validation/results/l3_muons_v1.json`): fake L3 muons per event and efficiency for the generated muons of the signal event after the muon identification, without any high-purity selection, with dev5, with dev6 and with v1. The efficiency counts every generated muon with pT > 2 GeV and |η| < 2.4, including those too soft to reach the muon stations or to have an L1 tracker muon: with v1 on ZMM+BsToMuMu it is 0.75 for |η| < 0.9 and 0.93–0.94 for |η| > 1.2.

| Chain, sample | Selection | Fake L3 muons / event | Efficiency |
|---|---|---|---|
| Pixel chain, ZMM+BsToMuMu | none / dev5 / dev6 / **v1** | 1.01 / 0.60 / 0.24 / **0.22** | 0.819 / 0.822 / 0.827 / **0.828** |
| Pixel chain, TTbar | none / dev5 / dev6 / **v1** | 1.03 / 0.65 / 0.29 / **0.28** | 0.840 / 0.846 / 0.852 / **0.855** |
| Seeds chain, ZMM+BsToMuMu | none / dev5 / dev6 / **v1** | 1.28 / 0.85 / 0.28 / **0.25** | 0.830 / 0.831 / 0.837 / **0.839** |
| Seeds chain, TTbar | none / dev5 / dev6 / **v1** | 1.28 / 0.90 / 0.33 / **0.30** | 0.859 / 0.866 / 0.868 / **0.870** |

With v1 the chain delivers about 4–5 times fewer fake muons than without a high-purity selection (2.3–3.4 times fewer than with dev5, 2–10% fewer than with dev6) and about one point more efficiency: a fake track can take the segments of a genuine muon in the arbitration of the muon identification, and it stops the standalone muon it is matched to from being passed to the outside-in step (with v1 five to six times more outside-in tracks are built per event than without a selection: 2.2 against 0.4 in the pixel chain on ZMM+BsToMuMu). The reduction of the fakes is smaller at the end of the chain than on the tracks because the muon identification, like the IO selectors, asks for compatible segments: it removes half of the fake muons made from IO tracks when there is no selection and a tenth of those v1 keeps. Of the fake muons left with v1 on ZMM+BsToMuMu, 0.19 (pixel chain) and 0.22 (seeds chain) per event come from IO tracks and 0.03 from OI tracks.

**On the test split of the training samples** (`production/results/summary_v1.json`):

| Model | Features | Trees | Nodes | .bin [MB] | ROC-AUC | PR-AUC | Precision | Recall | Fake rejection |
|---|---|---|---|---|---|---|---|---|---|
| IO pixel | 46 | 1100 | 278,238 | 3.6 | 0.99498 | 0.99020 | 0.8781 | 0.9901 | 0.9287 |
| IO seeds | 46 | 400 | 552,616 | 7.2 | 0.99764 | 0.99211 | 0.8621 | 0.9889 | 0.9585 |
| OI pixel | 22 | 200 | 5,958 | 0.08 | 0.99151 | 0.86929 | 0.3433 | 0.9856 | 0.9200 |
| OI general | 22 | 200 | 5,938 | 0.08 | 0.99254 | 0.88157 | 0.3591 | 0.9840 | 0.9280 |

Per working-point bin (test split; for IO the recall of the signal-event muons, on which the targets are defined):

| Model | pT bin [GeV] | Threshold | Tracks | Muons | Recall | Fake rejection |
|---|---|---|---|---|---|---|
| IO pixel | [0, 3) / [3, 5) / [5, 10) | 0.261 / 0.586 / 0.938 | 850,856 / 322,031 / 146,682 | 54,791 / 118,145 / 124,170 | 0.957 / 0.993 / 0.998 | 0.936 / 0.898 / 0.953 |
| IO pixel | [10, 50) / [50, 200) / ≥ 200 | 0.9990 / 0.9998 / 0.9998 | 57,623 / 72,507 / 105,061 | 56,620 / 72,466 / 105,002 | 0.991 / 0.995 / 0.996 | 0.993 / 0.878 / 0.983 |
| IO seeds | [0, 3) / [3, 5) / [5, 10) | 0.089 / 0.461 / 0.928 | 1,800,804 / 496,443 / 169,804 | 73,321 / 123,050 / 129,955 | 0.958 / 0.990 / 0.997 | 0.959 / 0.955 / 0.963 |
| IO seeds | [10, 50) / [50, 200) / ≥ 200 | 0.9982 / 0.9987 / 0.9995 | 60,092 / 72,125 / 108,701 | 56,402 / 71,688 / 108,654 | 0.991 / 0.995 / 0.995 | 0.998 / 0.998 / 0.957 |
| OI pixel | [0, 3) / [3, 5) / [5, 10) | 0.056 / 0.045 / 0.027 | 500,787 / 203,595 / 129,901 | 9,200 / 6,852 / 6,479 | 0.976 / 0.972 / 0.995 | 0.916 / 0.920 / 0.907 |
| OI pixel | [10, 50) / [50, 200) / ≥ 200 | 0.053 / 0.108 / 0.658 | 73,118 / 15,782 / 12,637 | 3,375 / 3,792 / 8,376 | 0.988 / 0.995 / 0.995 | 0.964 / 0.960 / 0.980 |
| OI general | [0, 3) / [3, 5) / [5, 10) | 0.064 / 0.032 / 0.071 | 488,426 / 194,087 / 123,148 | 8,461 / 5,431 / 5,259 | 0.965 / 0.977 / 0.989 | 0.923 / 0.909 / 0.958 |
| OI general | [10, 50) / [50, 200) / ≥ 200 | 0.032 / 0.138 / 0.815 | 71,443 / 15,444 / 15,364 | 2,614 / 2,950 / 11,028 | 0.994 / 0.997 / 0.994 | 0.951 / 0.971 / 0.988 |

The IO thresholds below 50 GeV are those pinned on RelVal muons, so the recall of the gun and Bs muons of the test split differs from the targets there (0.957 below 3 GeV): the two muon populations are not distributed alike, which is why the thresholds are pinned. The OI thresholds come from the validation split, and on the test split the OI recall is 1–2 points under the targets below 5 GeV (0.976 / 0.972 for OI pixel, 0.965 / 0.977 for OI general against 0.98 / 0.99); on the RelVal tracks of those bins it is 0.94–1.00, with few muons. For the IO selectors the fake rejection of the test split is two to four points above that on RelVal events.

## Validation in CMSSW (v1)

The deployed selectors in a CMSSW_20_1_0_pre3 area with the `L1Trigger/Phase2L1GMT` workaround, on the validation RelVal events (L1 tracker muons re-emulated + HLT + training NANO, 2,500 pre3 D128 events at 200 PU per sample and chain; `features_validation/results/parity_v1_*.json`, `l3_muons_v1.json`).

**CMSSW vs training code.** For the four selectors the features computed in CMSSW are bit-identical to the training extraction on the same events and no selection decision differs (83,532 IO and 32,703 OI tracks of the six jobs; scores agree to 1.8e-7). 12 further NANO tracks had no dump line in the job logs and are not compared.

**Cost.** Time per event of the selector modules (TimeReport, 8-thread jobs), ZMM / BsToMuMu / TTbar:

| Selector | v1 [ms / event] | dev6 [ms / event] | dev5 [ms / event] |
|---|---|---|---|
| IO pixel (v1: 1100 trees, depth 8) | 1.20 / 1.25 / 1.09 | 1.55 / 1.60 / 1.44 | 0.22 / 0.22 / 0.18 |
| IO seeds (v1: 400 trees, depth 12) | 1.39 / 1.52 / 1.28 | 2.11 / 2.25 / 1.90 | 0.32 / 0.34 / 0.29 |
| OI, pixel chain (v1: 200 trees, depth 4) | 0.023 / 0.025 / 0.022 | 0.051 / 0.058 / 0.054 | 0.031 / 0.036 / 0.033 |
| OI, seeds chain (v1: 200 trees, depth 4) | 0.022 / 0.024 / 0.022 | 0.092 / 0.103 / 0.097 | 0.021 / 0.022 / 0.022 |

(`features_validation/results/timing_v1.json`.) The three generations did not run at the same time: the node was shared with 6 to 12 other jobs and trainings and its load differed (the full event took 3.3–3.8 s in the v1 jobs, 4.2–4.7 s in the dev6 jobs, 3.5–4.0 s in the dev5 jobs), so compare the columns loosely. dev6 has 1300 (IO pixel), 600 (IO seeds), 700 and 1600 (OI) trees.

The IO selectors spend about 0.5 ms per event collecting the segments and rec-hits of the muon detectors in global coordinates and matching them to the tracks (measured with a 100-tree forest); the rest is the forest.

**Muons kept along the chain** (`input_selectors/results/*_chain_v1.json`; TrackingParticle muons with pT > 2 GeV, |η| < 2.4, reconstructed in the selector input and having their own L1TkMuon: 3,870 / 3,710 / 1,588 in the pixel chain for ZMM / BsToMuMu / TTbar; in parentheses 2–5 GeV; fakes per event after the step):

| | ZMM | BsToMuMu | TTbar | Fakes / event (ZMM, BsToMuMu, TTbar) |
|---|---|---|---|---|
| Pixel-track selector | 1.000 | 0.998 (0.997) | 0.989 (0.995) | 2.64, 2.73, 2.79 |
| + IO pixel forest | 0.992 (0.955) | 0.983 (0.980) | 0.976 (0.974) | 0.24, 0.27, 0.30 |
| Seed selector → IO tracks | 0.999 (0.995) | 0.996 (0.995) | 0.993 (0.987) | 5.37, 6.26, 5.94 |
| + IO seeds forest | 0.995 (0.959) | 0.983 (0.978) | 0.981 (0.962) | 0.32, 0.37, 0.39 |

These fractions are per muon and do not include what the outside-in step recovers (the L3 muon table above does). The ZMM 2–5 GeV muons are few and mostly non-prompt. On TTbar the pixel-track selector loses 18 of 1,588 muons with one track per L1TkMuon (`nTracksToKeep = 1`, the value in the cfi and in every number of this document); with two it loses one. Applying the v1 forest to the two-track input of the same events (the selector emulated on the stored tracks, not a new run of the chain) gives 0.992 / 0.984 / 0.987 of the muons kept (ZMM / BsToMuMu / TTbar, against 0.992 / 0.983 / 0.976) and 0.28 / 0.33 / 0.37 fakes per event after the forest (against 0.24 / 0.27 / 0.30). The L3 muons after the identification have not been counted with two tracks.


## Forest training pipeline (`production/forest_pipeline.py`)

- **Data** (v1): 28 ROOT files per HLT chain (113 GB for the pixel chain, 120 GB for the seeds chain; 20 particle-gun files covering the HLT pT spectrum + 8 Bs→μμ files, 1.28M events, D128 geometry, 200 pileup mixed from a 900k-event MinBias pool), produced with the HLT configuration of the deployment, with the tables of the muon local reconstruction. Signal = track matched to a muon TrackingParticle by the validation associator; background = unmatched ("fake") tracks. 5.1M (IO pixel), 9.1M (IO seeds), 3.1M / 3.0M (OI pixel / general) tracks. The muon HLT starts from the L1 tracker muons, and the L1 emulation of CMSSW_20_1_0_pre3 finds one for only about 30% of the muons with |η| < 0.8 ([cms-sw/cmssw#52096](https://github.com/cms-sw/cmssw/issues/52096)): the barrel stubs handed to the track-plus-stub matching of `L1Trigger/Phase2L1GMT` carry z and a slope in the fields the matching reads as pseudorapidity. The v1 samples were produced with a workaround (those stubs built as in CMSSW_20_0_0_patch1): an L1 tracker muon is within ΔR < 0.1 of 96–98% of the gun muons above 5 GeV for |η| < 0.8 and of 95–97% at higher |η| (31–34% and 79–96% in a production without the workaround, on which dev6 was trained). Measure this fraction per |η| region before using a sample; RelVal files made with that release store the deficient L1 tracker muons (step 7 of the instructions).
- **Feature cache**: content-addressed (`--cache-dir`, default `/tmp/$USER/muonhp_feature_cache`): the key covers the extraction code (source of `build_dataset` and every function/constant it uses), its arguments, the branch list, the input-file manifest (path, size, mtime) and the numpy/awkward/uproot versions, so a stale cache is never used. A job stops if the feature code changes on disk while it runs: the key is computed from the source files, which then no longer describe the running code. The extraction runs one process per input file (`feature_workers`).
- **Feature ABI**: the kept features are asserted name by name against the order of the CMSSW extractors (`io_production_features()`, `OI_PRODUCTION_FEATURES` = `toArray()` of `muonhp::IOTrackFeatures` / `muonhp::OITrackFeatures`); a dropped feature that does not exist is an error.
- **Split — copy-aware**: the pileup is mixed from a finite MinBias pool, so the same pileup track recurs in many events (0.5% / 0.6% of the IO pixel / seeds tracks are bit-identical copies of another track, many more are near-copies: every pileup event is used about 290 times). Exact copies are removed (`dedup_copies`), and tracks are grouped by three quantities rounded to 0.01 (`copy_group_features`: log₁₀ pT, |η| and log₁₀ of the dz uncertainty for IO; log₁₀ pT, η and pixel hits for OI); whole groups go to train 60% / val 10% / test 30% (`split_mode="copygroup"`), so a recurring track never sits in both training and test. An event-level split (`split_mode="evt10"`, also available) is not independent when tracks recur: forests memorise them and the test metrics overestimate the performance. Each training prints the fraction of copies.
- **Weights**: signal ×5 (normalised per class); no kinematic 1/pT weighting.
- **Trees**: histogram XGBoost, η = 0.1, subsample/colsample 0.8, no early stopping. IO: **depth 8** (pixel, 1500-tree budget) and **depth 12** (seeds, 400-tree budget; Hyperparameters), `min_child_weight` 5 (`io_xgb_common.RECIPE`). OI: depth 4, 2000-tree budget, Gaussian jitter of 0.3 σ on the continuous training features (`train_jitter`, against memorising recurring tracks). The quantile cuts are computed on the host (`QuantileDMatrix`), then trees are grown on the GPU or the CPU; the thread count is that of the CPUs the process may use (cpuset and container quota; `nthread`).
- **Size selection**: smallest tree prefix whose validation fake rejection at the validation-chosen working points is within a tolerance of the best prefix (`prune_tol`: 0.002 for IO, 0.005 for OI). v1 keeps 1100 / 400 / 200 / 200 trees (IO pixel / IO seeds / OI pixel / OI general); for IO seeds this is the whole budget, set where independent events stop improving.
- **Working points**: one threshold per pT bin (`[0,3,5,10,50,200]` GeV) keeping a fixed fraction of the muons of the bin (`wp_mode="recall"`, `wp_recall_targets`; 1.0 = open threshold). The fraction is defined on the validation muons of the bin: for IO, those of the signal event (`wp_signal_feature`: tracks of a generator-level muon; the pileup muons of the samples are signal for the training but not for the targets); bins with < 100 such muons use the global threshold (99% of all muons). With `--wp-files` the threshold of a bin is instead the same quantile of the muons of independent events (`wp_pinning` in `thresholds.json`), for the bins those events populate: at least 10 reference muons below the threshold (`wp_files_min_rejected`), since the precision of a quantile is set by the number of muons beyond it. The other bins keep the validation threshold. F2-optimal points remain available (`wp_mode="f2"`). The test split is evaluated once, at the final working points.
- **Exports and verification**: `model.json`; `model_compact.bin` (CMSSW), replayed with a numpy re-implementation of the CMSSW traversal on 300k random test tracks plus every test track within 0.02 of its working point (asserts |Δscore| < 1e-5 and **0 working-point decision flips**; the exporter also refuses models whose split nodes send missing values left, which the NaN → right traversal could not reproduce); `model_xgb.onnx`, replayed through ONNX Runtime on 100k test tracks (+ single-thread latency).
- **Records**: `train.log`, `thresholds.json` (working points and their pinning, test metrics, size curve, feature ABI, .bin md5), `manifest.json` (git commit and dirty flag, package versions, device, input files, cache key, verification results, output checksums), `cmssw_cfi_snippet.py`. `thresholds.json`, `manifest.json` and `train.log` are version-controlled.
- **Logs**: warnings are shown once and must not occur (`check_consistency.py` fails on any warning/error line). ONNX Runtime sessions use explicit thread pools sized to the CPUs the process may use. The DNN trainers pin the TorchScript ONNX exporter (opset 13, `dynamo=False`; torch ≥ 2.9 switches the default) and verify every export with ONNX Runtime.

### Feature sets

- **IO, 46 features** (`io_production_features(prefix)`, same list for the pixel and seeds chains, CMSSW `muonhp::IOTrackFeatures`):
  - 9 track quantities: `p`, `pt`, `dzErr`, `qoverpErr`, `etaErr`, `phiErr` (log-compressed), `sigmaPtOverPt` (relative pT uncertainty, log), `sip2D` (significance of the transverse impact parameter, log), `absEta`;
  - `L1TkMu_nStubs_Endcap`: the number of endcap muon stubs of the L1 tracker muon matched to the track (0 without a match);
  - 36 quantities of the track-to-chamber matching (`chamber_match.py`). For a track of charge q, transverse momentum pT and direction (η, φ) at its closest approach to the beam line, and a segment or rec-hit of a muon station, **bend** = q · Δφ(position) · pT [rad GeV] is the charge-signed azimuthal displacement of the object from the track direction, scaled by pT so that it is about the same at every momentum (negative on the side the charge bends to); **dEta** is its η minus the track's; **bendDiff**, for segments, is how much more the segment direction is bent than its position. The stations are ME0 (segments), the four DT/CSC stations (segments), GE1/1 and GE2/1 (rec-hits; the HLT builds no segments there) and the four RPC stations (rec-hits). The candidates of a station are the objects in a wide window (|dEta| < 0.3, bend on the charge side and below 8 rad GeV in size, rec-hits within one bunch crossing) whose direction, for segments, is also bent on the charge side; they are ordered by |dEta|, the coordinate the field does not bend. The features are `cm_bend_<station>` and `cm_dEta_<station>` of the nearest candidate in each of the 11 stations (22), `cm_bendDiff_<station>` for the 5 segment stations (5), and the second candidate where the nearest one is most often not the muon's own: `cm_bend2_`, `cm_dEta2_` in ME0, station 1, GE1/1 and GE2/1 (8; in the GEM stations it is mostly the muon's hit in the other layer) and `cm_bendDiff2_St1` (1). A station without a candidate has values outside the range of any candidate (9 for a bend, 0.5 for dEta).

  Where a muon's own segments lie in these quantities depends on the station and on |η|: the forest learns it from the raw values. Nothing is tabulated or fitted, and no second model scores the matches.
- **OI, 22 features** (`OI_PRODUCTION_FEATURES`, `muonhp::OITrackFeatures`): 19 track features (`p`, `pt`, `normalizedChi2`, `etaErr`, `phiErr`, `dszErr`, `dxyErr`, `dzErr`, `qoverpErr`, `lambdaErr`, `eta`, `nPixelHits`, `nTrkLays`, `nFoundHits`, `nLostHits`, `impact3D`, `impactSignificance`, `chi2PerHit`, `hitEfficiency`), the L2 standalone-muon matching (`l2_mu_vtx_hasMatch`, `l2_mu_vtx_matchingScore`) and `is_low_pt` (logistic in pT around 5 GeV). Dropped from 26: `ptErr`, `sigmaPtOverPt`, `relUncertaintyProduct` (pT-uncertainty information of `qoverpErr`), `chi2` (with `normalizedChi2`/`chi2PerHit`).
- **How the IO list was chosen.** `build_dataset()` computes 44 track and L1TkMuon quantities and, for every station, the two nearest candidates (54 quantities). The 9 track quantities and the single L1 one are those whose thresholds, set on the gun and Bs muons of the training sample, give the target recall on independent RelVal muons (with the full track list the recall on RelVal fell from 99% to 96–98%). The chamber quantities were built by structure and checked against alternatives on independent events: second candidates beyond those listed add nothing measurable, and lists pruned by importance keep more fakes (Hyperparameters).
- **What the v1 forests use** (gain, `feature_importance.txt` of each model): the IO selectors by pT and the L1 stub count first, then the bends and η differences of the stations (the outer RPC station, ME0, stations 2 and 3, GE2/1, the second RPC station), the curvature and η uncertainties and |η|; the OI selectors by the hit pattern (`nTrkLays`, `nPixelHits`, found hits, `hitEfficiency`), the 3D impact parameter and `is_low_pt`.

### Hyperparameters

The IO forests have to learn, from raw quantities, where the candidates of a muon lie in each station. Depth, forest size and feature list were scanned on the v1 samples with the pipeline's own training split and weights, and judged on the validation RelVal events at equal recall: per-pT-bin thresholds that keep 98 / 99 / 99.5% of the signal-event muons of all validation events, so that every variant is compared at the same muon efficiency. Fakes kept per event, ZMM+BsToMuMu / TTbar, 46 features:

| Chain | Depth | 400 trees | 1000 trees | 1500 trees |
|---|---|---|---|---|
| IO pixel | 6 | 0.261 / 0.293 | 0.260 / 0.294 | 0.259 / 0.296 |
| IO pixel | **8** | 0.264 / 0.308 | 0.265 / 0.308 | 0.262 / 0.304 |
| IO pixel | 10 | 0.256 / 0.299 | 0.274 / 0.319 | 0.278 / 0.319 |
| IO pixel | 12 | 0.257 / 0.301 | 0.249 / 0.290 | 0.249 / 0.290 |
| IO seeds | 8 | 0.440 / 0.485 | 0.416 / 0.458 | |
| IO seeds | 10 | 0.389 / 0.440 | 0.425 / 0.482 | |
| IO seeds | **12** | 0.378 / 0.424 | 0.486 / 0.519 | |
| IO seeds | 12, another seed | 0.378 / 0.427 | 0.435 / 0.481 | |
| IO seeds | 14 | 0.378 / 0.426 | 0.393 / 0.438 | |
| IO seeds | 14, another seed | 0.439 / 0.492 | 0.521 / 0.564 | |

- **Noise.** Three trainings that differ only by the random seed keep 0.264 / 0.282 / 0.245 fakes per event on ZMM+BsToMuMu (pixel, 800 trees) and 0.400 / 0.393 / 0.446 (seeds, 500 trees), about ±7%; the finite validation sample adds about 8% (pixel) and 15% (seeds) to one number (a handful of muons below 3 GeV set the threshold of the bin that holds most of the kept fakes). A difference below 10–15% between two rows is not significant.
- **IO pixel**: depth and size make no significant difference between 400 and 1500 trees and between depths 6 and 12. Depth 8 with a 1500-tree budget is kept; the size selection keeps 1100 trees.
- **IO seeds**: at 400 trees depths 10, 12 and 14 keep 0.38–0.39 fakes per event on ZMM+BsToMuMu (one of the two depth-14 trainings 0.44) and depth 8 keeps 0.44. Beyond 400 trees the depth-12 forests keep more fakes at equal recall on the RelVal events (13–29% more at 1000 trees, in the two seeds; depth 10 and one of the two depth-14 trainings do the same), while the fake rejection of the copy-aware validation and test splits keeps improving: with a 1000-tree budget the size selection keeps 900 trees, which give 0.467 / 0.509 fakes per event at equal recall against 0.378 / 0.424 for 400 trees. The evidence is moderate for one training and has the same sign in all the deep ones; the smaller forest is also faster. Depth 12 is kept, with a budget of 400 trees, which the size selection keeps entirely.
- **Feature list** (pixel at 800 trees, seeds at 500, one training each): adding the second candidate of DT/CSC stations 2–4 and of RPC1 (54 features), using all 98 built quantities, or dropping the 3 or 6 least important of the 46, changes the fakes kept by less than the noise (ratios to the 46 between 0.96 and 1.09). Lists pruned by importance from the 98 (70 to 40 features) do better on the test split and keep about 10–40% more fakes on the RelVal events.

η = 0.1, subsample and colsample 0.8, `min_child_weight` 5 throughout. The OI forests keep the depth-4 recipe of the pipeline defaults.


## Feature parity between training and CMSSW

The training features are computed from the n-tuple branches (float32), the CMSSW features from `reco::Track`/`l1t::TrackerMuon` objects and from the segments and rec-hits of the muon detectors. Both follow one **numeric convention**: every raw input rounded to float32 (as the n-tuple stores it), all arithmetic in float64 (including an exact `reco::deltaPhi` equivalent in Python), each feature rounded to float32 once (`pixel_features.as_float64`, `muonhp::asStored` in the C++ headers).

For the chamber matching the inputs are the global η and φ of the segments and hits. The n-tuple tables evaluate them through the ROOT interpreter, which does not fuse x·x + y·y into a multiply-add as compiled code does; `muonhp::chamberPerp`/`chamberEta` (`MuonChamberMatch.h`) keep the two roundings, so the selector sees the values of the tables to the last bit. Candidates at the same |dEta| are taken in the order of the tables (detector, then collection order). The RPC rec-hit table stores the CMSSW global η and φ of the hits, in the training and in the validation n-tuples.

The convention is checked on real events by `production/features_validation/validate_cmssw.py` (step 7 of the instructions), which joins a per-track dump of the CMSSW selectors with the training extraction on the same events and requires bit-identical features and no differing selection decision.

## CMSSW integration

All four selectors live in `RecoMuon/L3TrackFinder` on the [`20_1_X_muonTracking_v1` branch](https://github.com/Parsifal-2045/cmssw/tree/20_1_X_muonTracking_v1) of the private Parsifal-2045/cmssw fork, based on CMSSW_20_1_0_pre3. Its last commit holds only the model files, so that it can be dropped once cms-data serves them; the commit before it is the temporary `L1Trigger/Phase2L1GMT` workaround (step 7 of the instructions).

- **Models**: `RecoMuon/L3TrackFinder/data/IO/muonHP_IO_{pixelPath,seedsPath}_forest_v1.bin` and `data/OI/muonHP_OI_{pixelPath,seedsPath}_forest_v1.bin` (family / HLT chain / version), described in `data/README.md` (selector, chain, input tracks, cfi module, trees, test performance, training commit, md5). No other model files remain in `data/`. For the upstream PR they belong in cms-data (`RecoMuon-L3TrackFinder`).
- **Features**: `interface/IOTrackSelectorFeatures.h` (46, `muonhp::IOTrackFeatures`: named track quantities, the L1 stub count and a `muonhp::ChamberMatch`, the nearest and second candidate of every station with their `bend`, `dEta`, `bendDiff`), `interface/MuonChamberMatch.h` (the search window and the candidate search, `muonhp::matchChambers`) and `interface/OITrackSelectorFeatures.h` (22, `muonhp::OITrackFeatures`), float64 convention above.
- **Inference**: `interface/CompactForest.h` (shared by both plugins): loads the `.bin` once per process (`GlobalCache`), re-lays it out for cache locality and validates it against the extractor (feature indices inside the feature vector, children after parents, exact file size) — loading a model trained on another feature set fails with a clear error; `BinnedWorkingPoints` holds the per-pT-bin thresholds.
- **Plugins**: `MuonIOTracksForestSelector` (pixel and seeds IO), `MuonOITracksForestSelector` (both OI chains): extract the features of all tracks, score them in one batched call, output the selected tracks and the scores of all input tracks; `modelPath` is a required `FileInPath`; `nFeatures` is checked once against the extractor; inputs are read with `iEvent.get` (a missing product is an error); debugging output via MessageLogger; `dumpFeatures` prints one `MUONHP_FEATURES,<label>,<run>,<lumi>,<event>,<track>,<features>,<score>` line per track for the cross-check. The IO plugin also reads the DT, CSC and GEM segments, the GEM and RPC rec-hits (`dtSegments`, `cscSegments`, `gemSegments`, `gemRecHits`, `rpcRecHits`; defaults: the HLT collections) and the muon geometries, and builds once per event the list of segments and hits in global coordinates.
- **cfis**: `hltPhase2MuonPixelTracksHighPurityForest_cfi.py`, `hltPhase2MuonIOTrackSelectionHighPurityForest_cfi.py`, `hltPhase2L3OIMuonTrackSelectionHighPurityForest_cfi.py`; model path and working points are written from the trainings' `thresholds.json` by `production/deploy_cmssw.py` and checked by `production/check_consistency.py --cmssw-src`.
- **Wiring**: the process modifiers `phase2MuonPixelTracksSelector` / `phase2MuonSeedsSelector` (and `ngtScouting`, pixel chain) swap the forests into `HLTPhase2MuonPixelTracksFromL1TkSequence` (IO pixel), `HLTPhase2L3MuonsIOSequence` (IO seeds) and `hltPhase2L3OIMuonTrackSelectionHighPurity` (OI). The muon local reconstruction runs before the IO selectors in both chains.
- **Tests**: `test/forestSelectorsSmoke_cfg.py` constructs the four selectors and loads/validates their forests; the per-track behaviour on real events is checked by `production/features_validation/validate_cmssw.py`.

- **Inference cost**: `CompactForest` re-lays every tree out depth-first as packed 8-byte nodes at load time (the file format is unchanged) and the plugins score all tracks of an event in one call, tree-major (a tree's nodes are reused across the event's tracks while in cache); scores are bit-identical to the plain traversal. `features_validation/forest_bench.cc` measures the cost per track, the TimeReport of the validation jobs the cost per event.
- **Input selectors**: the HP forests only see the tracks the L1TkMuon-based selectors keep. `MuonTracksSelectorFromL1TkMuon` (`hltPhase2MuonPixelTracks`) keeps the best pixel track per L1TkMuon by a dR/dz/curvature score (`nTracksToKeep = 1` in `hltPhase2MuonPixelTracks_cfi.py`). `MuonSeedsSelectorFromL1TkMuon` (`hltPhase2MuonIOTrackSeeds`) keeps LST seeds around each L1TkMuon, arbitrated per L1TkMuon, with the seed direction taken at its closest approach to the beam line (`beamSpot` parameter). `production/input_selectors/` measures the fraction of muons each selection keeps.

## Decisions and rationale

**Training protocol**
- *Muon guns with pileup, plus Bs→μμ.* Particle-gun muons over the full HLT pT spectrum keep the selection independent of any specific physics process; pileup mixing provides the non-isolated environment and its fakes; Bs→μμ adds mildly displaced muons. Physics samples such as tt̄ or DY are deliberately not part of the training.
- *Copy-aware split, test used once.* Tracks of one event share pileup and duplicates, and with a recycled pileup pool the same track recurs across events, so neither a track-level nor an event-level split is independent; the copy-aware split keeps every recurring track on one side. Forest size is chosen on validation, the recipe on the test split of scratch trainings together with independent RelVal events; the test split of the production training only reports. The RelVal events never enter the training of a forest: one set pins the working points, a disjoint set validates the result.
- *Recall-target working points per pT bin, pinned on independent events.* A muon track lost by the HP selection is not recovered downstream, while a kept fake costs rate and time; the selector should act like the muon identification, keeping the muons. The recall is fixed bin by bin: 98% below 3 GeV (where the fakes are and the separation is worst), 99% at 3–5 GeV, 99.5% above, for all four selectors. For the IO selectors the fraction refers to the muons of the signal event, and the thresholds are set on the muons of independent RelVal events where these populate the bin: the gun and Bs muons of the training sample are not distributed like those of physics samples, and thresholds set on them give a recall that is off by up to a point there, in either direction.
- *Tree budget + validation prefix selection, no early stopping.* The full size curve is scored at the validation-chosen working points (the production criterion) and the smallest prefix within a tolerance of the best validation fake rejection is kept. The budget bounds what the selection can choose: for the deep forests of the seeds chain the validation split of the training sample, which shares the pileup pool with the training, keeps improving with trees that cost fakes on independent events, so the budget is set on those (400 trees).
- *Depth per chain.* The IO forests learn the matching bands of 11 stations from raw quantities and need depth (8 for the pixel tracks, where depths 6 to 12 are level on independent events; 12 for the IO tracks of the seeds chain, where depths 10 to 14 are level and depth 8 is not better); the OI forests, on track quantities, use depth 4.
- *Host-side quantile sketch.* The GPU sketch of weighted data differs between processes; the `QuantileDMatrix` sketch is computed on the host and the trees are grown on the GPU or the CPU.

**Muon-chamber matching**
- *Local reconstruction instead of L1 quantities.* The input tracks are already selected by their match to an L1 tracker muon, so the L1 matching quantities separate little; whether the track has compatible segments and hits in the muon stations separates much more. One L1 quantity is kept, the number of endcap stubs.
- *Raw quantities of the candidates, one forest.* The forest gets the bend and the η difference of the candidates and learns where a muon's own lie. The alternatives were calibration tables of the expected bend per station and η (hard-coded constants to maintain per geometry) and a second forest scoring every track–object pair (two models to train, deploy and keep consistent). A single forest on raw quantities needs neither and comes close once it is deep enough: at the same working points it keeps about 5% (pixel chain) and 20% (seeds chain) more fakes than the two-forest version.
- *Candidates by |dEta|, on the charge side.* The only choices of the matching are the wide search window and the order of the candidates: nearest in the coordinate the field does not bend, among the objects displaced on the side the charge bends to. The nearest candidate is the muon's own in 70–95% of the cases depending on the station; the second candidate is given where it is not (ME0, station 1 and the two GEM stations).
- *GEM and RPC rec-hits.* GE1/1 and GE2/1 have no segments at the HLT and the RPC stations complement the DT/CSC segments: their hits enter as stations of their own.

**Training/CMSSW agreement**
- *float64 feature convention.* Inputs rounded to float32 (as the n-tuples store them), arithmetic in float64, one rounding per feature: numpy computes in float64 by default, and making the C++ follow it is exact and simple, while float32 arithmetic would need op-by-op agreement between numpy and libm. A float/double mismatch of one imputed constant is enough to put tracks on the wrong side of forest splits.
- *One feature ABI.* The feature order is asserted name by name against the C++ extractors at every training, and the plugins validate every `.bin` against the extractor when they load it.
- *Compact `.bin` format, fp32.* No ONNX Runtime in the HLT path, 2.5x smaller than the ONNX serialisation, traversal replayed in Python at every training (0 decision flips) and on real events; fp16 leaf values accumulate ~0.2 score error over thousands of trees. NaN goes right (the exporter refuses default-left split nodes, so the traversal needs no missing-value flags). ONNX is kept as a verified secondary export.
- *Packed, batched inference.* Deep forests have hundreds of thousands of nodes and are memory-bound; the packed depth-first layout and tree-major batching cut their cost per track.

**Inputs of the HP forests**
- *`nTracksToKeep = 1`* for the pixel-track selector: the best track per L1TkMuon keeps every reachable muon on ZMM and all but 0.2% on BsToMuMu, and more tracks per L1TkMuon add fakes. On TTbar, where the muons are not isolated, it loses 1.1% of them to a track closer to the L1TkMuon; a second track per L1TkMuon recovers them for 0.04–0.07 more fakes per event after the forest (Validation in CMSSW). The forest is trained and its working points are pinned on the one-track input, and the two-track option has not been run through the full chain: it is not adopted.
- *Two seeds per L1TkMuon* for the seed selector: with the seed direction taken at the beam line it keeps practically every reachable muon, so more seeds per L1TkMuon are unnecessary.

**Deployment**
- *Model files* in `RecoMuon/L3TrackFinder/data/IO` and `/OI`, named by family, HLT chain and version (`muonHP_<IO|OI>_<pixelPath|seedsPath>_forest_<version>.bin`). A deployed retraining always gets a new version (`model_version`): files in cms-data are immutable and a changed model under an old name would silently change results. `deploy_cmssw.py` writes the cfi working points from `thresholds.json` (never by hand), and `check_consistency.py --cmssw-src` verifies files, checksums and cfi values.
- *No machine-specific paths in code.* Every input is a command-line flag (How to run the training chain) rather than a path in the code or an environment variable; the records (`manifest.json`, `train.log`) keep the absolute input paths as provenance.
- *DNN exports* pin the TorchScript ONNX exporter (`dynamo=False`, opset 13): torch ≥ 2.9 switches the default to the dynamo exporter.
