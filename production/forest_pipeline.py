"""
forest_pipeline.py - Shared training pipeline of the muon high-purity
XGBoost forests deployed in CMSSW (IO pixel, IO seeds, OI pixel, OI general).

The flavour scripts (pixel_xgb.py, seeds_xgb.py, oi/OI_pixel_xgb.py,
oi/OI_general_xgb.py) hold configuration only; the pipeline lives here:

  1. features: n-tuple -> build_dataset() -> content-addressed cache (the key
     covers the extraction code, its constants, the branch list and the input
     file manifest, so a stale cache can never be picked up);
  2. pruning to the production feature ABI, asserted name-by-name against the
     order of the CMSSW extractor (toArray() of the C++ feature structs);
  3. sample weights: signal boost x capped 1/pT kinematic weight (per class);
  4. event-level "evt10" split: ev % 10 -> train 60 / val 10 / test 30;
  5. XGBoost training on a fixed tree budget. Quantile cuts are computed on
     the host (QuantileDMatrix): the GPU sketch of weighted data is not
     reproducible across processes, the host sketch is, so a re-run with the
     same inputs reproduces the model bit for bit;
  6. forest size chosen on validation: smallest tree prefix whose validation
     F2 (at validation-chosen per-pT-bin working points) is within prune_tol
     of the best prefix;
  7. per-pT-bin F2 working points, chosen on validation and frozen;
  8. a single evaluation of the frozen model + working points on test;
  9. exports: model.json, model_compact.bin (CMSSW) and model_xgb.onnx, each
     replayed against XGBoost. The .bin replay re-implements the CMSSW
     traversal (float32, sequential accumulation) and is run on a random
     test sample plus every test track whose score is within
     VERIFY_THR_WINDOW of its working point, so the working-point decision
     flips between CMSSW and Python are counted over the full test split;
 10. records: train.log, thresholds.txt/json, manifest.json (code, data and
     package provenance), cmssw_cfi_snippet.py (the values for the CMSSW cfi),
     feature importances and plots.

Input n-tuples (--data-dir): the NANO:@MUHLTTraining files of the flavour's
HLT chain - pixel-track chain for IO pixel / OI pixel, seeds chain for
IO seeds / OI general.

Usage (via a flavour script):
    python pixel_xgb.py --data-dir DIR [--cache-dir DIR] [--output-dir DIR] [--force]
                        [--device cpu|cuda] [--set key=json_value ...] [--no-permutation-importance]
"""

import argparse
import gc
import getpass
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import warnings

import numpy as np
import uproot
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import onnx_utils
import pixel_features as pf

PROD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROD_DIR)

# Modules whose attributes count as feature-extraction code for the cache key
_FEATURE_MODULES = ("pixel_features", "OI_features")
# Their files at startup: the cache key hashes the source read from disk, which
# must be the code this process runs (see feature_code_fingerprint).
_FEATURE_FILES = {p: os.stat(p).st_mtime_ns for p in (
    os.path.join(PROD_DIR, "pixel_features.py"), os.path.join(PROD_DIR, "oi", "OI_features.py"))}

# .bin verification: random test rows replayed through the CMSSW traversal,
# plus every test row whose score lies within this window of its threshold.
VERIFY_N_RANDOM = 300_000
VERIFY_THR_WINDOW = 0.02
VERIFY_TOL = 1e-5

DEFAULTS = dict(
    # Sample weights
    signal_boost=pf.SIGNAL_BOOST,
    kin_weight_max=pf.KIN_WEIGHT_MAX,
    # XGBoost hyperparameters. Depth 12 / learning rate 0.2 (v3) from the
    # validation-only scan (tuning/forest_scan.py, tuning/results/): at equal
    # CMSSW inference cost it beats the v2 depth-6 / 0.1 forests by +0.003
    # (IO pixel) and +0.008 (IO seeds) in validation F2, confirmed on a second
    # validation fold, and the size selection keeps 4-6x fewer trees.
    max_depth=12,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    max_bin=256,
    seed=42,
    # Tree budget + size selection on validation (no early stopping)
    n_estimators=5000,
    prune_tol=0.001,       # val-F2 tolerance for the size selection
    prune_grid_step=50,    # prefix grid spacing [trees]
    prune_min_trees=100,   # smallest prefix considered
    device="cuda",         # "cuda" falls back to CPU (loudly) if unavailable
    # Event-level split
    split_mode="evt10",
    split_seed_offset=0,
    # pT-binned working points [GeV]: one F2 set point per bin, last bin
    # open-ended; bins with < min_bin_signal validation signal tracks (or
    # without background) fall back to the global F2 threshold.
    pt_threshold_edges=[0.0, 2.0, 5.0, 10.0, 50.0, 200.0],
    min_bin_signal=100,
    # Exports. fp16 .bin values accumulate ~0.2 score error over thousands of
    # trees: too lossy for a fixed-threshold selector, keep fp32.
    onnx_opset=pf.ONNX_OPSET,
    compact_bin_fp16=False,
    # Diagnostics
    permutation_importance=True,
    perm_repeats=5,
    ref_model_path=None,   # previous forest, re-scored on the same split
    verbose_eval=250,
    # Model generation: fills "{version}" in the flavours' cmssw_bin (deployed
    # file names, which must change with every deployed retraining: cms-data
    # files are immutable) and "{ref_version}" in their ref_model_path (the
    # previous production forests, archived under */archive/).
    model_version="v3",
    ref_version="v2",
)


def full_config(flavour_cfg, overrides=None):
    """DEFAULTS + flavour configuration + overrides, version placeholders filled."""
    cfg = dict(DEFAULTS)
    cfg.update(flavour_cfg)
    cfg.update(overrides or {})
    for k in ("cmssw_bin", "ref_model_path"):
        if cfg.get(k):
            cfg[k] = cfg[k].format(version=cfg["model_version"], ref_version=cfg["ref_version"])
    return cfg


# --------------------------------------------------------------------------- #
# Feature cache
# --------------------------------------------------------------------------- #
def _code_objects(code):
    yield code
    for c in code.co_consts:
        if inspect.iscode(c):
            yield from _code_objects(c)


def feature_code_fingerprint(build_fn):
    """Hash of the feature-extraction code: build_fn's source plus the source
    of every project function and the value of every constant it references
    (transitively, including module attributes such as pf.LOW_PT_CUT)."""
    # inspect.getsource reads the files on disk: after an edit they no longer
    # describe the loaded code, and the features of the old code would be
    # cached under the key of the new one.
    changed = [rel(p) for p, t in _FEATURE_FILES.items() if os.stat(p).st_mtime_ns != t]
    if changed:
        raise SystemExit(f"feature code changed on disk while this job was running ({', '.join(changed)}): "
                         "restart it")
    parts, seen = [], set()

    def visit(fn):
        if fn in seen:
            return
        seen.add(fn)
        parts.append(inspect.getsource(fn))
        g = fn.__globals__
        names = sorted({n for c in _code_objects(fn.__code__) for n in c.co_names})
        mods = [g[n] for n in names if n in g and inspect.ismodule(g[n])
                and g[n].__name__ in _FEATURE_MODULES]
        for n in names:
            for owner, obj in ([("", g[n])] if n in g else []) + [
                (m.__name__ + ".", getattr(m, n)) for m in mods if hasattr(m, n)
            ]:
                if inspect.isfunction(obj) and obj.__module__ in _FEATURE_MODULES:
                    visit(obj)
                elif isinstance(obj, (bool, int, float, str, list, tuple)):
                    parts.append(f"{owner}{n}={obj!r}")

    visit(build_fn)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def list_input_files(data_dir):
    return pf.get_files(data_dir)


# HLT chains of the training n-tuples (flavour config "data_chain")
CHAINS = {
    "pixel": "pixel-track chain (process modifier phase2MuonPixelTracksSelector)",
    "seeds": "seeds chain (process modifier phase2MuonSeedsSelector)",
}
DEFAULT_CACHE_DIR = f"/tmp/{getpass.getuser()}/muonhp_feature_cache"


def add_data_dir_arg(ap, chain=None, flag="--data-dir", required=True):
    """--data-dir style argument: directory with the training n-tuples of an HLT chain."""
    what = f"the {CHAINS[chain]}" if chain else "the flavour's HLT chain"
    ap.add_argument(flag, required=required, metavar="DIR",
                    help=f"directory with the training n-tuples (*.root) of {what}")


def add_cache_dir_arg(ap):
    ap.add_argument("--cache-dir", metavar="DIR", default=DEFAULT_CACHE_DIR,
                    help="feature cache directory (default: %(default)s)")


def resolve_data_dir(cfg):
    """The flavour's n-tuple directory (cfg["data_dir"], from --data-dir)."""
    path = cfg.get("data_dir")
    if not path:
        raise SystemExit(f"{cfg['name']}: no n-tuple directory given (--data-dir, {CHAINS[cfg['data_chain']]})")
    if not os.path.isdir(path):
        raise SystemExit(f"n-tuple directory {path} does not exist")
    return path


def rel(path):
    """Path relative to the repository when inside it (records and logs
    carry no machine-specific prefix for repository files)."""
    ap = os.path.abspath(path)
    return os.path.relpath(ap, REPO_DIR) if ap.startswith(REPO_DIR + os.sep) else path


def feature_cache_key(cfg, files):
    manifest = [(f, os.path.getsize(f), os.stat(f).st_mtime_ns) for f in files]
    payload = json.dumps(
        dict(
            code=feature_code_fingerprint(cfg["build"]),
            build_kwargs=cfg.get("build_kwargs", {}),
            branches=cfg["branches"],
            files=manifest,
            packages={m: importlib.metadata.version(m) for m in ("numpy", "awkward", "uproot")},
        ),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest(), manifest


def cache_dir(cfg):
    return cfg.get("cache_dir") or DEFAULT_CACHE_DIR


def load_features(cfg):
    """(X, y, fl, ev, names, files, cache_info) for the flavour; fl is the
    input-file index and ev the per-file event index of every track."""
    data_dir = resolve_data_dir(cfg)
    files = list_input_files(data_dir)
    if not files:
        raise RuntimeError(f"no input files in {data_dir}")
    key, manifest = feature_cache_key(cfg, files)
    path = os.path.join(cache_dir(cfg), f"{cfg['cache_tag']}_{key[:16]}.npz")
    info = dict(key=key, path=path, files=manifest)
    if os.path.exists(path):
        c = np.load(path, allow_pickle=False)
        if str(c["key"]) == key:
            print(f"Loading cached features from {path} (key {key[:16]})")
            names = [str(n) for n in c["names"]]
            print(f"  {c['X'].shape[0]} tracks x {len(names)} features, {len(files)} input files")
            info["hit"] = True
            return c["X"], c["y"], c["fl"], c["ev"], names, files, info
        print(f"Cache {path} has a different key; rebuilding.")
    info["hit"] = False

    print(f"Building features from {len(files)} input files (cache key {key[:16]}):")
    X_l, y_l, fl_l, ev_l, names = [], [], [], [], None
    total_ev = 0
    for i, f in enumerate(files):
        t0 = time.time()
        with uproot.open(f) as rf:
            a = rf[pf.MAIN_BRANCH].arrays(cfg["branches"])
        ne = len(a)
        total_ev += ne
        Xc, yc, lc, ec, fn = cfg["build"](
            a, np.full(ne, i), event_ids=np.arange(ne, dtype=np.int64), **cfg.get("build_kwargs", {})
        )
        if names is None:
            names = fn
        elif fn != names:
            raise RuntimeError(f"feature names differ between input files ({f})")
        X_l.append(Xc); y_l.append(yc); fl_l.append(lc); ev_l.append(ec)
        print(f"  [{i + 1:2d}/{len(files)}] {os.path.basename(f)}: {ne} events, "
              f"{len(yc)} tracks ({time.time() - t0:.0f}s)", flush=True)
        del a
        gc.collect()
    X = np.concatenate(X_l); y = np.concatenate(y_l)
    fl = np.concatenate(fl_l).astype(np.int32); ev = np.concatenate(ev_l)
    print(f"  Total: {total_ev} events, {X.shape[0]} tracks x {X.shape[1]} features")
    os.makedirs(cache_dir(cfg), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, X=X, y=y, fl=fl, ev=ev, names=np.array(names), files=np.array(files), key=np.array(key))
    os.replace(tmp, path)
    print(f"  Cached to {path} ({os.path.getsize(path) / 1e9:.2f} GB)")
    return X, y, fl, ev, names, files, info


def select_features(X, names, drop, expected):
    """Drop features by name; the result must equal `expected` (the CMSSW
    extractor order) exactly."""
    unknown = [f for f in drop if f not in names]
    if unknown:
        raise ValueError(f"drop_features not produced by build_dataset: {unknown}")
    keep = [i for i, n in enumerate(names) if n not in drop]
    kept = [names[i] for i in keep]
    if expected is not None and kept != list(expected):
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(kept, expected)) if a != b]
        raise ValueError(
            f"kept features do not match the production ABI ({len(kept)} vs {len(expected)} "
            f"features; first differences {diff[:5]})"
        )
    return X[:, keep], kept


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def xgb_params(cfg):
    return dict(
        objective="binary:logistic",
        eval_metric=["aucpr", "auc", "logloss"],
        tree_method="hist",
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        min_child_weight=cfg["min_child_weight"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        max_bin=cfg["max_bin"],
        seed=cfg["seed"],
    )


def train_forest(cfg, X_tr, y_tr, w_tr, X_va, y_va, w_va, n_rounds=None, verbose_eval=None):
    """Train on the host-sketched QuantileDMatrix (bit-reproducible on GPU).
    Returns (booster, device_used, evals_result)."""
    dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, weight=w_tr, max_bin=cfg["max_bin"])
    dval = xgb.QuantileDMatrix(X_va, label=y_va, weight=w_va, ref=dtrain, max_bin=cfg["max_bin"])
    n_rounds = n_rounds or cfg["n_estimators"]
    verbose_eval = cfg["verbose_eval"] if verbose_eval is None else verbose_eval
    device = cfg["device"]
    evals_result = {}
    try:
        bst = xgb.train(dict(xgb_params(cfg), device=device), dtrain, num_boost_round=n_rounds,
                        evals=[(dtrain, "train"), (dval, "val")], evals_result=evals_result,
                        verbose_eval=verbose_eval)
    except xgb.core.XGBoostError as e:
        if device == "cpu":
            raise
        print(f"\n  !!! device={device} failed ({e}); FALLING BACK TO CPU !!!\n")
        device, evals_result = "cpu", {}
        bst = xgb.train(dict(xgb_params(cfg), device="cpu"), dtrain, num_boost_round=n_rounds,
                        evals=[(dtrain, "train"), (dval, "val")], evals_result=evals_result,
                        verbose_eval=verbose_eval)
    return bst, device, evals_result


def base_logit(bst):
    """logit(base_score): XGBoost stores base_score as a probability for
    binary:logistic; the CMSSW kernel adds baseLogit in margin space."""
    bs = float(json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"])
    bs = min(max(bs, 1e-7), 1.0 - 1e-7)
    return float(np.log(bs / (1.0 - bs)))


def prefix_margins(bst, X, grid, step):
    """Margins of every tree prefix in `grid` (multiples of step + full size),
    accumulated block-wise in one pass over the forest."""
    n_trees = bst.num_boosted_rounds()
    b0 = base_logit(bst)
    d = xgb.DMatrix(X)
    run = np.full(X.shape[0], b0, dtype=np.float64)
    out = {}
    for t0 in range(0, n_trees, step):
        t1 = min(t0 + step, n_trees)
        run += bst.predict(d, iteration_range=(t0, t1), output_margin=True) - b0
        if t1 in grid:
            out[t1] = run.copy()
    assert set(out) == set(grid)
    return out


def val_f2_at_binned_wp(y_val, scores, pt_val, cfg):
    wp = pf.find_f2_threshold_binned(y_val, scores, pt_val, cfg["pt_threshold_edges"], cfg["min_bin_signal"])
    dec = pf.apply_binned_thresholds(scores, pt_val, cfg["pt_threshold_edges"], [b[2] for b in wp["bins"]])
    tn, fp, fn, tp = confusion_matrix(y_val, dec, labels=[0, 1]).ravel()
    return pf.calculate_metrics((tp, fp, fn, tn))[4]


def size_curve(bst, X_val, y_val, pt_val, cfg):
    """[(n_trees, val F2 at the val-chosen per-bin WPs)] on the prune grid."""
    n_fit = bst.num_boosted_rounds()
    step = cfg["prune_grid_step"]
    g0 = step * max(1, int(np.ceil(cfg["prune_min_trees"] / step)))
    grid = sorted(set(range(g0, n_fit, step)) | {n_fit})
    margins = prefix_margins(bst, X_val, grid, step)
    return [(n, val_f2_at_binned_wp(y_val, 1.0 / (1.0 + np.exp(-margins[n])), pt_val, cfg)) for n in grid]


def choose_size(curve, tol):
    best_n, best_f2 = max(curve, key=lambda c: (c[1], -c[0]))
    chosen = min(n for n, f2 in curve if f2 >= best_f2 - tol)
    return chosen, best_n, best_f2


# --------------------------------------------------------------------------- #
# Compact .bin export (CMSSW forest selectors) and its C++-faithful replay
# --------------------------------------------------------------------------- #
# Format (little endian):
#   int32 nNodes, int32 nTrees, float32 baseLogit,
#   int8  feat[nNodes]   (-1 = leaf),
#   float32 val[nNodes]  (threshold / leaf value),
#   int32 left[nNodes], right[nNodes],
#   int32 roots[nTrees]
# Traversal: while feat[node] >= 0: node = x[feat] < val ? left : right;
# margin += val[leaf]; score = sigmoid(margin). Missing values (NaN) fail
# every `<` and go right, which equals XGBoost's behaviour only if no node
# has default_left set: the exporter asserts it.
def _flatten_tree(node_json, feat, val, left, right):
    """Append one tree (BFS, sequential child allocation); returns its root."""
    root = len(feat)
    feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
    queue = [(node_json, root)]
    while queue:
        n, slot = queue.pop(0)
        if "leaf" in n:
            feat[slot] = -1
            val[slot] = float(n["leaf"])
            continue
        feat[slot] = int(n["split"][1:])  # "f12" -> 12
        val[slot] = float(n["split_condition"])
        children = {c["nodeid"]: c for c in n["children"]}
        l_idx = len(feat)
        feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
        r_idx = len(feat)
        feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
        left[slot], right[slot] = l_idx, r_idx
        queue.append((children[n["yes"]], l_idx))
        queue.append((children[n["no"]], r_idx))
    return root


def forest_arrays(bst, fp16=False):
    raw = json.loads(bst.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"]
    n_default_left = sum(sum(t["default_left"]) for t in raw)
    if n_default_left:
        raise ValueError(f"{n_default_left} nodes route missing values left; the compact .bin "
                         "(NaN -> right) cannot represent this model")
    feat, val, left, right, roots = [], [], [], [], []
    for t in bst.get_dump(dump_format="json"):
        roots.append(_flatten_tree(json.loads(t), feat, val, left, right))
    val = np.asarray(val, dtype=np.float32)
    if fp16:
        val = val.astype(np.float16).astype(np.float32)
    if max(feat) > 127:
        raise ValueError("feature index does not fit the int8 .bin field")
    return dict(
        feat=np.asarray(feat, dtype=np.int8), val=val,
        left=np.asarray(left, dtype=np.int32), right=np.asarray(right, dtype=np.int32),
        roots=np.asarray(roots, dtype=np.int32), base_logit=np.float32(base_logit(bst)),
    )


def write_compact_bin(forest, path):
    import struct

    with open(path, "wb") as f:
        f.write(struct.pack("<iif", len(forest["feat"]), len(forest["roots"]), float(forest["base_logit"])))
        for k in ("feat", "val", "left", "right", "roots"):
            f.write(forest[k].tobytes())


def read_compact_bin(path):
    import struct

    with open(path, "rb") as f:
        data = f.read()
    n_nodes, n_trees, bl = struct.unpack("<iif", data[:12])
    o = 12
    feat = np.frombuffer(data, np.int8, n_nodes, o); o += n_nodes
    val = np.frombuffer(data, np.float32, n_nodes, o); o += 4 * n_nodes
    left = np.frombuffer(data, np.int32, n_nodes, o); o += 4 * n_nodes
    right = np.frombuffer(data, np.int32, n_nodes, o); o += 4 * n_nodes
    roots = np.frombuffer(data, np.int32, n_trees, o); o += 4 * n_trees
    if o != len(data):
        raise ValueError(f"{path}: {len(data) - o} trailing bytes")
    return dict(feat=feat, val=val, left=left, right=right, roots=roots, base_logit=np.float32(bl))


def compact_forest_predict(forest, X):
    """Scores exactly as the CMSSW forest selectors compute them (float32,
    trees accumulated in file order), vectorised over rows."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    feat = forest["feat"].astype(np.int64)
    val, left, right = forest["val"], forest["left"], forest["right"]
    margin = np.full(n, forest["base_logit"], dtype=np.float32)
    for r in forest["roots"]:
        node = np.full(n, r, dtype=np.int64)
        f = np.full(n, feat[r])
        act = np.nonzero(f >= 0)[0]
        while act.size:
            nd = node[act]
            go_left = X[act, f[act]] < val[nd]
            node[act] = np.where(go_left, left[nd], right[nd])
            f[act] = feat[node[act]]
            act = act[f[act] >= 0]
        margin += val[node]
    return (1.0 / (1.0 + np.exp(-margin))).astype(np.float32)


def verify_compact_bin(path, bst, X_test, pt_test, cfg, bin_edges, bin_thrs, seed=0):
    """Replay the .bin on (random sample + all near-threshold rows) of the test
    split; asserts |score diff| < VERIFY_TOL and counts decision flips."""
    forest = read_compact_bin(path)
    if int(forest["feat"].max()) >= X_test.shape[1]:
        raise ValueError(".bin references a feature index outside the feature vector")
    s_xgb = bst.predict(xgb.DMatrix(X_test))
    thr = np.asarray(bin_thrs)[np.searchsorted(np.asarray(bin_edges), pt_test, side="right") - 1]
    near = np.nonzero(np.abs(s_xgb - thr) < VERIFY_THR_WINDOW)[0]
    rng = np.random.default_rng(seed)
    rand = rng.choice(len(X_test), size=min(VERIFY_N_RANDOM, len(X_test)), replace=False)
    rows = np.union1d(near, rand)
    t0 = time.time()
    s_bin = compact_forest_predict(forest, X_test[rows])
    diff = np.abs(s_bin - s_xgb[rows])
    flips = int(np.sum((s_bin >= thr[rows]) != (s_xgb[rows] >= thr[rows])))
    res = dict(rows_checked=int(len(rows)), near_threshold_rows=int(len(near)),
               max_abs_diff=float(diff.max()), decision_flips=flips, seconds=round(time.time() - t0, 1))
    print(f"  .bin replay (CMSSW traversal) on {len(rows)} test rows "
          f"({len(near)} within {VERIFY_THR_WINDOW} of their WP): max|diff|={res['max_abs_diff']:.2e}, "
          f"WP decision flips={flips} [{res['seconds']}s]")
    if res["max_abs_diff"] >= VERIFY_TOL or flips:
        raise AssertionError(f".bin replay disagrees with XGBoost: {res}")
    return res


# --------------------------------------------------------------------------- #
# ONNX export (TreeEnsembleClassifier, for ONNX Runtime consumers)
# --------------------------------------------------------------------------- #
def export_onnx(bst, feature_names, path, opset, graph_name):
    import onnx
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    model = convert_xgboost(bst, initial_types=[("input", FloatTensorType([None, len(feature_names)]))],
                            target_opset=opset)
    model.graph.name = graph_name  # the converter draws a random UUID; keep exports reproducible
    meta = {f"feature_{i}": n for i, n in enumerate(feature_names)}
    meta["feature_names"] = ",".join(feature_names)
    for k, v in meta.items():
        e = onnx.StringStringEntryProto(); e.key = k; e.value = v
        model.metadata_props.append(e)
    onnx.save_model(model, path)
    return path


def verify_onnx(path, bst, X_test, n=100_000):
    """ONNX (probabilities[:, 1]) vs XGBoost on n test rows + a 1-thread
    single-track latency benchmark (CMSSW-like)."""
    Xs = np.ascontiguousarray(X_test[:n], dtype=np.float32)
    sess = onnx_utils.make_session(path)
    name = sess.get_inputs()[0].name
    probs = next(o for o in sess.run(None, {name: Xs}) if o.ndim == 2 and o.shape[1] == 2)
    max_diff = float(np.abs(probs[:, 1] - bst.predict(xgb.DMatrix(Xs))).max())
    single = onnx_utils.make_session(path, intra_threads=1)
    x1 = Xs[:1]
    for _ in range(100):
        single.run(None, {name: x1})
    t0 = time.perf_counter()
    for _ in range(1000):
        single.run(None, {name: x1})
    us = (time.perf_counter() - t0) / 1000 * 1e6
    print(f"  ONNX vs XGBoost on {len(Xs)} test rows: max|diff|={max_diff:.2e}; "
          f"single-track latency (1 thread): {us:.0f} us")
    if max_diff >= VERIFY_TOL:
        raise AssertionError(f"ONNX/XGBoost mismatch: max|diff|={max_diff:.2e}")
    return dict(rows_checked=len(Xs), max_abs_diff=max_diff, single_track_us=round(us, 1))


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def decision_metrics(y_true, dec):
    tn, fp, fn, tp = confusion_matrix(y_true, dec, labels=[0, 1]).ravel()
    p, r, a, f1, f2 = pf.calculate_metrics((tp, fp, fn, tn))
    return dict(precision=p, recall=r, f2=f2, fake_rejection=float(tn / max(tn + fp, 1)),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def perbin_table(y_true, dec, pt, edges):
    rows = []
    for (lo, hi) in pf.pt_bins_from_edges(edges):
        m = (pt >= lo) & (pt < hi)
        if not m.any():
            continue
        d = decision_metrics(y_true[m], dec[m])
        d.update(lo=lo, hi=hi if np.isfinite(hi) else None, n=int(m.sum()), n_sig=int(y_true[m].sum()))
        rows.append(d)
    return rows


def print_perbin_table(rows, title):
    print(f"\n  {title}")
    print(f"  {'pT bin':>12s} {'N':>9s} {'N sig':>8s} {'Prec':>7s} {'Rec':>7s} {'F2':>7s} "
          f"{'FakeRej':>8s} {'FN':>6s} {'FP':>7s}")
    for r in rows:
        lab = f"{r['lo']:g}-{r['hi']:g}" if r["hi"] is not None else f">{r['lo']:g}"
        print(f"  {lab:>12s} {r['n']:>9d} {r['n_sig']:>8d} {r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['f2']:>7.4f} {r['fake_rejection']:>8.4f} {r['fn']:>6d} {r['fp']:>7d}")


def permutation_importance(bst, X, y, names, repeats, seed=0):
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, bst.predict(xgb.DMatrix(X)))
    imp = np.zeros((len(names), repeats), dtype=np.float32)
    Xp = X.copy()
    for fi, fn in enumerate(names):
        col = X[:, fi].copy()
        for r in range(repeats):
            Xp[:, fi] = rng.permutation(col)
            imp[fi, r] = base - average_precision_score(y, bst.predict(xgb.DMatrix(Xp)))
        Xp[:, fi] = col
        print(f"    [{fi:02d}] {fn:<45s} dPR-AUC={imp[fi].mean():+.5f} +/- {imp[fi].std():.5f}")
    return imp.mean(1), imp.std(1), base


def rescore_reference(ref_path, X_val, y_val, pt_val, X_test, y_test, pt_test, cfg):
    """Previous forest on the same split, with its own validation-derived WPs.
    NB: its training may have seen events of this test split (older split)."""
    ref = xgb.Booster()
    ref.load_model(ref_path)
    s_val = ref.predict(xgb.DMatrix(X_val))
    s_test = ref.predict(xgb.DMatrix(X_test))
    wp = pf.find_f2_threshold_binned(y_val, s_val, pt_val, cfg["pt_threshold_edges"], cfg["min_bin_signal"])
    dec = pf.apply_binned_thresholds(s_test, pt_test, cfg["pt_threshold_edges"], [b[2] for b in wp["bins"]])
    res = dict(path=rel(ref_path), trees=ref.num_boosted_rounds(),
               roc_auc=float(roc_auc_score(y_test, s_test)), pr_auc=float(average_precision_score(y_test, s_test)),
               perbin_wp=decision_metrics(y_test, dec))
    print(f"  Reference {rel(ref_path)} ({res['trees']} trees): ROC-AUC={res['roc_auc']:.6f} PR-AUC={res['pr_auc']:.6f} "
          f"@per-bin WPs P={res['perbin_wp']['precision']:.4f} R={res['perbin_wp']['recall']:.4f} "
          f"F2={res['perbin_wp']['f2']:.4f}")
    return res


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _git(*args):
    try:
        return subprocess.run(["git", "-C", PROD_DIR, *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        idx = int(vis.split(",")[0]) if vis and vis.split(",")[0].isdigit() else 0
        return out[idx] if idx < len(out) else None
    except Exception:
        return None


def _jsonable_cfg(cfg):
    return {k: v for k, v in cfg.items() if isinstance(v, (bool, int, float, str, list, dict, type(None)))}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run(flavour_cfg, overrides=None, force=False):
    try:
        return _run(flavour_cfg, overrides, force)
    finally:
        if isinstance(sys.stdout, pf.TeeStream):
            sys.stdout.close_log()


def _run(flavour_cfg, overrides, force):
    cfg = full_config(flavour_cfg, overrides)
    cfg["data_dir"] = resolve_data_dir(cfg)
    out = os.path.join(cfg["output_dir"], "")
    if os.path.isdir(out) and os.listdir(out):
        if not force:
            raise SystemExit(f"output directory {out} is not empty; use --force to replace it "
                             "(or --output-dir to write elsewhere)")
        shutil.rmtree(out)
    os.makedirs(out)
    pf.tee_log(out, display=rel(out + "train.log"))
    warnings.simplefilter("default")  # every distinct warning is shown once, in the log
    t_start = time.time()

    manifest = dict(
        flavour=cfg["name"], started=time.strftime("%Y-%m-%d %H:%M:%S"),
        host=socket.gethostname(), user=getpass.getuser(), python=platform.python_version(),
        git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain", "--", ".")),
        packages={m: importlib.metadata.version(m) for m in (
            "xgboost", "numpy", "scikit-learn", "awkward", "uproot", "onnx", "onnxruntime", "onnxmltools")},
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu=_gpu_name(),
        available_cpus=onnx_utils.available_cpus(), config=_jsonable_cfg(cfg),
    )
    for k in ("output_dir", "ref_model_path"):
        if manifest["config"].get(k):
            manifest["config"][k] = rel(manifest["config"][k])
    print("=" * 70 + f"\n{cfg['name']} forest  ->  {rel(out)}\n" + "=" * 70)
    print(f"  git {manifest['git_commit']} (dirty={manifest['git_dirty']})  xgboost {xgb.__version__}")

    # ---- Features -------------------------------------------------------- #
    X, y, fl, ev, names, files, cache_info = load_features(cfg)
    manifest["inputs"] = dict(data_dir=cfg["data_dir"], cache_key=cache_info["key"],
                              cache_path=cache_info["path"], cache_hit=cache_info["hit"],
                              files=[dict(path=p, size=s, mtime_ns=t) for p, s, t in cache_info["files"]])
    X, names = select_features(X, names, cfg["drop_features"], cfg.get("production_features"))
    print(f"\nFeatures: {len(names)} (dropped {len(cfg['drop_features'])}); order matches the CMSSW extractor")
    print(f"  {names}")
    pt_idx = names.index(cfg["pt_feature"])
    n_nonfinite = int((~np.isfinite(X)).any(axis=1).sum())
    if n_nonfinite:
        raise ValueError(f"{n_nonfinite} rows with non-finite features after extraction")

    # ---- Weights + split ------------------------------------------------- #
    pt_all = 10 ** X[:, pt_idx]
    weights = pf.compute_sample_weights(y, pt_all, signal_boost=cfg["signal_boost"],
                                        kin_weight_max=cfg["kin_weight_max"])
    print(f"\nSample weights: signal mean {weights[y == 1].mean():.3f}, background mean "
          f"{weights[y == 0].mean():.3f}; pT < {pf.LOW_PT_CUT:g} GeV tracks: "
          f"{100 * (pt_all < pf.LOW_PT_CUT).mean():.1f}%")
    assert cfg["split_mode"] == "evt10", f"unknown split_mode {cfg['split_mode']}"
    tr, va, te = pf.evt10_split(ev, seed_offset=cfg["split_seed_offset"])
    split_info = {}
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        n_evt = len(np.unique(fl[m].astype(np.int64) * (1 << 32) + ev[m]))
        split_info[nm] = dict(tracks=int(m.sum()), signal=int(y[m].sum()), events=n_evt)
        print(f"  {nm:5s}: {m.sum():>9d} tracks ({y[m].sum():>8d} signal, {100 * y[m].mean():4.1f}%) "
              f"in {n_evt} events")
    manifest["split"] = split_info
    X_tr, y_tr, w_tr = X[tr], y[tr], weights[tr]
    X_va, y_va, w_va, pt_va = X[va], y[va], weights[va], pt_all[va]
    X_te, y_te, fl_te, pt_te = X[te], y[te].astype(np.int32), fl[te], pt_all[te]
    del X, weights, tr, va, te
    gc.collect()

    # ---- Train ----------------------------------------------------------- #
    print("\n" + "=" * 70 + "\nXGBOOST TRAINING\n" + "=" * 70)
    print(f"  params: {dict(xgb_params(cfg), device=cfg['device'])}")
    print(f"  budget: {cfg['n_estimators']} trees (no early stopping; size chosen on validation)")
    t0 = time.time()
    bst, device, evals = train_forest(cfg, X_tr, y_tr, w_tr, X_va, y_va, w_va)
    manifest["device_used"] = device
    manifest["train_seconds"] = round(time.time() - t0, 1)
    print(f"  trained {bst.num_boosted_rounds()} trees on {device} in {manifest['train_seconds']}s")
    del X_tr, y_tr, w_tr
    gc.collect()

    # ---- Size selection (validation) ------------------------------------- #
    print("\n" + "=" * 70 + "\nFOREST SIZE (validation F2 at validation-chosen per-bin WPs)\n" + "=" * 70)
    curve = size_curve(bst, X_va, y_va, pt_va, cfg)
    chosen, best_n, best_f2 = choose_size(curve, cfg["prune_tol"])
    print("  " + " ".join(f"{n}:{f2:.5f}" for n, f2 in curve))
    print(f"  best prefix {best_n} (val F2 {best_f2:.5f}); smallest within {cfg['prune_tol']}: {chosen} trees")
    if best_n == curve[-1][0]:
        print(f"  NOTE: the curve peaks at the budget end ({best_n}); the chosen size is budget-limited")
    bst = bst[0:chosen]
    assert bst.num_boosted_rounds() == chosen

    # ---- Working points (validation, frozen) ----------------------------- #
    print("\n" + "=" * 70 + "\nWORKING POINTS (validation only)\n" + "=" * 70)
    s_val = bst.predict(xgb.DMatrix(X_va))
    wp = pf.find_f2_threshold_binned(y_va, s_val, pt_va, cfg["pt_threshold_edges"], cfg["min_bin_signal"])
    edges = list(cfg["pt_threshold_edges"])
    thrs = [float(b[2]) for b in wp["bins"]]
    for lo, hi, thr, n_sig, n_rows, fb in wp["bins"]:
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:g}"
        print(f"    pT [{lo:6.1f}, {hi_s:>4s}) GeV: thr={thr:.6f}  n_sig={n_sig}  n={n_rows}"
              + ("  (fallback -> global F2)" if fb else ""))
    val_curve_f2 = dict(curve)[chosen]

    # ---- Test (once) ----------------------------------------------------- #
    print("\n" + "=" * 70 + "\nTEST (frozen model + working points)\n" + "=" * 70)
    s_te = bst.predict(xgb.DMatrix(X_te))
    roc = float(roc_auc_score(y_te, s_te))
    prauc = float(average_precision_score(y_te, s_te))
    dec_g = s_te >= wp["global_f2"]
    dec_b = pf.apply_binned_thresholds(s_te, pt_te, edges, thrs)
    m_g, m_b = decision_metrics(y_te, dec_g), decision_metrics(y_te, dec_b)
    print(f"  ROC-AUC {roc:.6f}  PR-AUC {prauc:.6f}")
    for lab, m in (("global F2 WP", m_g), ("per-bin F2 WPs", m_b)):
        print(f"  @{lab:15s}: P={m['precision']:.4f} R={m['recall']:.4f} F2={m['f2']:.4f} "
              f"fake rejection={m['fake_rejection']:.4f}")
    print(f"\n{classification_report(y_te, dec_b.astype(int), digits=4)}")
    tab_wp = perbin_table(y_te, dec_b, pt_te, edges)
    print_perbin_table(tab_wp, "per working-point bin (per-bin WPs)")
    tab_std = perbin_table(y_te, dec_b, pt_te, [0.0, 5.0, 10.0, 50.0, 200.0])
    print_perbin_table(tab_std, "standard reporting bins (per-bin WPs)")
    print("\n  per input file (per-bin WPs):")
    for fi, fname in enumerate(files):
        fm = fl_te == fi
        if not fm.any():
            continue
        m = decision_metrics(y_te[fm], dec_b[fm])
        auc_f = roc_auc_score(y_te[fm], s_te[fm]) if len(np.unique(y_te[fm])) == 2 else float("nan")
        print(f"    {os.path.basename(fname):<22s} AUC={auc_f:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F2={m['f2']:.4f} [TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}]")
    pf.plot_roc_pr(y_te, s_te, out)
    pf.plot_confusion_matrix(y_te, None, 0, out, decisions=dec_b, title_suffix="(per-pT-bin F2 working points)")
    pf.plot_confusion_matrix(y_te, s_te, wp["global_f2"], out, title_suffix="(global F2 working point)",
                             filename="confusion_matrix_globalthr.png")
    pf.evaluate_pt_bins(y_te, s_te, pt_te, wp["global_f2"], out, decisions=dec_b)

    # ---- Exports + verification ------------------------------------------ #
    print("\n" + "=" * 70 + "\nEXPORTS\n" + "=" * 70)
    bst.save_model(out + "model.json")
    forest = forest_arrays(bst, fp16=cfg["compact_bin_fp16"])
    write_compact_bin(forest, out + "model_compact.bin")
    bin_md5 = _md5(out + "model_compact.bin")
    print(f"  model_compact.bin: {len(forest['feat'])} nodes, {len(forest['roots'])} trees, "
          f"baseLogit={float(forest['base_logit']):.8f}, "
          f"{os.path.getsize(out + 'model_compact.bin') / 1024:.0f} KB, md5 {bin_md5}")
    bin_check = verify_compact_bin(out + "model_compact.bin", bst, X_te, pt_te, cfg, edges, thrs)
    export_onnx(bst, names, out + "model_xgb.onnx", cfg["onnx_opset"],
                graph_name=f"muonHP_{cfg['cache_tag']}_forest")
    onnx_check = verify_onnx(out + "model_xgb.onnx", bst, X_te)

    # ---- Records ---------------------------------------------------------- #
    thresholds = dict(
        model="XGBoost", flavour=cfg["name"], split="evt10",
        n_features=len(names), feature_names=names,
        tree_budget=cfg["n_estimators"], chosen_trees=chosen, prune_tol=cfg["prune_tol"],
        best_prefix=best_n, best_prefix_val_f2=best_f2, val_f2_at_chosen=val_curve_f2,
        size_curve=[[n, f2] for n, f2 in curve],
        global_f1_threshold=float(wp["global_f1"]), global_f2_threshold=float(wp["global_f2"]),
        pt_bin_edges=edges, pt_bin_f2_thresholds=thrs,
        pt_bin_fallback=[bool(b[5]) for b in wp["bins"]],
        compact_bin=dict(md5=bin_md5, n_nodes=int(len(forest["feat"])), n_trees=int(len(forest["roots"])),
                         base_logit=float(forest["base_logit"]), cmssw_file=cfg.get("cmssw_bin")),
        test=dict(roc_auc=roc, pr_auc=prauc, global_wp=m_g, perbin_wp=m_b,
                  f2_global_wp=m_g["f2"], f2_perbin_wp=m_b["f2"],
                  precision_perbin_wp=m_b["precision"], recall_perbin_wp=m_b["recall"],
                  per_wp_bin=tab_wp),
    )
    with open(out + "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(out + "thresholds.txt", "w") as f:
        f.write(f"Flavour: {cfg['name']}\nModel: XGBoost ({chosen} trees, {len(names)} features)\n")
        f.write(f"F1_Threshold: {wp['global_f1']}\nF2_Threshold: {wp['global_f2']}\n")
        f.write(f"Test_ROC_AUC: {roc}\nTest_PR_AUC: {prauc}\n")
        f.write(f"Test_F2_global_WP: {m_g['f2']}\nTest_F2_perbin_WP: {m_b['f2']}\n")
        f.write(f"Test_perbin_WP: precision={m_b['precision']} recall={m_b['recall']} "
                f"fake_rejection={m_b['fake_rejection']}\n")
        f.write("Split: evt10 (event-level, train 60 / val 10 / test 30)\n")
        f.write(f"Tree_budget: {cfg['n_estimators']}\nChosen_trees: {chosen} (prune_tol={cfg['prune_tol']}, "
                f"val F2 {val_curve_f2:.6f}, best prefix {best_n} at {best_f2:.6f})\n")
        f.write("pT-bin F2 set points (validation-derived):\n")
        for (lo, hi, thr, n_sig, _, fb) in wp["bins"]:
            hi_s = "inf" if not np.isfinite(hi) else f"{hi:.1f}"
            f.write(f"  pT [{lo:.1f}, {hi_s}): thr={thr:.6f}  n_val_sig={n_sig}"
                    + ("  [fallback->global]\n" if fb else "\n"))
        f.write(f"compact .bin md5: {bin_md5}\n")
    write_cmssw_snippet(out + "cmssw_cfi_snippet.py", cfg, thresholds)

    # ---- Feature importance ---------------------------------------------- #
    print("\n" + "=" * 70 + "\nFEATURE IMPORTANCE\n" + "=" * 70)
    gain = bst.get_score(importance_type="gain")
    cover = bst.get_score(importance_type="cover")
    ranked = sorted(((names[int(k[1:])], v, cover.get(k, 0.0)) for k, v in gain.items()), key=lambda r: -r[1])
    unused = [n for i, n in enumerate(names) if f"f{i}" not in gain]
    with open(out + "feature_importance.txt", "w") as f:
        f.write(f"Test PR-AUC: {prauc:.6f}\nTest ROC-AUC: {roc:.6f}\n\n")
        f.write(f"{'Rank':<6}{'Feature':<45}{'Gain':>12}{'Cover':>12}\n" + "-" * 75 + "\n")
        for i, (n, g, c) in enumerate(ranked):
            f.write(f"{i + 1:<6}{n:<45}{g:>12.2f}{c:>12.1f}\n")
            print(f"  {i + 1:<4}{n:<45}{g:>12.2f}  cover={c:.1f}")
        if unused:
            f.write(f"\nUnused features (no split): {unused}\n")
            print(f"  unused features: {unused}")
    pf.plot_importance(np.array([r[1] for r in ranked]), np.zeros(len(ranked)), [r[0] for r in ranked],
                       "Feature importance (XGBoost gain) - full test", out + "feat_imp_all.png")
    lpm = pt_te < pf.LOW_PT_CUT
    if cfg["permutation_importance"] and lpm.sum() > 1000 and len(np.unique(y_te[lpm])) == 2:
        print(f"\n  permutation importance (PR-AUC drop), pT < {pf.LOW_PT_CUT:g} GeV test subset "
              f"({lpm.sum()} tracks):")
        t0 = time.time()
        bst.set_param({"device": device})
        il, sl, _ = permutation_importance(bst, X_te[lpm], y_te[lpm], names, cfg["perm_repeats"])
        pf.plot_importance(il, sl, names, "Feature importance (permutation) - low-pT", out + "feat_imp_lowpt.png")
        with open(out + "feature_importance.txt", "a") as f:
            f.write(f"\nPermutation importance (PR-AUC drop, pT < {pf.LOW_PT_CUT:g} GeV test subset, "
                    f"{cfg['perm_repeats']} repeats):\n")
            for i in np.argsort(-il):
                f.write(f"  {names[i]:<45}{il[i]:+.5f} +/- {sl[i]:.5f}\n")
        print(f"  [{time.time() - t0:.0f}s]")

    # ---- Reference forest ------------------------------------------------ #
    ref = None
    if cfg.get("ref_model_path") and os.path.isfile(cfg["ref_model_path"]):
        print("\n" + "=" * 70 + "\nREFERENCE FOREST (same split, own val-derived WPs)\n" + "=" * 70)
        ref = rescore_reference(cfg["ref_model_path"], X_va, y_va, pt_va, X_te, y_te, pt_te, cfg)

    manifest.update(
        verification=dict(compact_bin=bin_check, onnx=onnx_check),
        outputs={fn: _md5(out + fn) for fn in ("model.json", "model_compact.bin", "model_xgb.onnx",
                                                 "thresholds.json")},
        reference=ref, total_seconds=round(time.time() - t_start, 1),
        finished=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(out + "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone in {manifest['total_seconds']:.0f}s -> {rel(out)}")
    return thresholds, manifest


def write_cmssw_snippet(path, cfg, th):
    """Parameters of the CMSSW forest-selector cfi for this model."""
    lines = [
        f"# {cfg['name']} forest: {th['chosen_trees']} trees, {th['n_features']} features, "
        f".bin md5 {th['compact_bin']['md5']}",
        f"# generated by forest_pipeline.py; paste into {cfg.get('cmssw_cfi', 'the selector cfi')}",
        f"modelPath = cms.FileInPath('{cfg.get('cmssw_bin', 'RecoMuon/L3TrackFinder/data/<model>.bin')}'),",
        f"decisionThreshold = cms.double({th['global_f2_threshold']!r}),",
        f"ptBinEdges = cms.vdouble({', '.join(repr(float(e)) for e in th['pt_bin_edges'])}),",
        "decisionThresholds = cms.vdouble(",
    ]
    for (lo, hi), thr, fb in zip(pf.pt_bins_from_edges(th["pt_bin_edges"]), th["pt_bin_f2_thresholds"],
                                 th["pt_bin_fallback"]):
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:g}"
        lines.append(f"    {thr!r},  # pT [{lo:g}, {hi_s})" + (" - global F2 fallback" if fb else ""))
    lines += [")", f"nFeatures = cms.int32({th['n_features']}),"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main_cli(flavour_cfg):
    ap = argparse.ArgumentParser(description=f"Train the {flavour_cfg['name']} HP forest")
    add_data_dir_arg(ap, flavour_cfg["data_chain"])
    add_cache_dir_arg(ap)
    ap.add_argument("--output-dir", help="output directory (default: the production directory)")
    ap.add_argument("--force", action="store_true", help="replace a non-empty output directory")
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--no-permutation-importance", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override a config value, e.g. --set max_depth=8")
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in DEFAULTS and k not in flavour_cfg:
            ap.error(f"unknown config key {k}")
        over[k] = json.loads(v)
    over["data_dir"], over["cache_dir"] = a.data_dir, a.cache_dir
    if a.output_dir:
        over["output_dir"] = a.output_dir
    if a.device:
        over["device"] = a.device
    if a.no_permutation_importance:
        over["permutation_importance"] = False
    run(flavour_cfg, over, force=a.force)
