#!/usr/bin/env python3
"""
check_consistency.py - verify that the production forests, their training
records and the CMSSW deployment are mutually consistent.

Per production model (io/pixel, io/seeds, oi/pixel, oi/general):
  * artifacts present (model.json, model_compact.bin, model_xgb.onnx,
    thresholds.json, manifest.json, train.log, cmssw_cfi_snippet.py);
  * the .bin and the ONNX re-exported from model.json are byte-identical to
    the stored ones (same training run, exports reproducible); ONNX and
    model.json agree numerically; the ONNX metadata and thresholds.json carry
    the production feature ABI (= order of the CMSSW extractor);
  * tree counts and .bin md5 agree across thresholds.json / manifest.json /
    model.json / .bin;
  * train.log free of warnings, errors and tracebacks;
With --cmssw-src:
  * the deployed .bin (cfi modelPath) is byte-identical to model_compact.bin;
  * the cfi working points (decisionThreshold, ptBinEdges,
    decisionThresholds, nFeatures) equal thresholds.json exactly;
  * RecoMuon/L3TrackFinder/data holds exactly the four deployed .bin files.

Usage:
    python check_consistency.py [--cmssw-src /path/to/CMSSW_X/src]
Exit status 0 if everything is consistent.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import tempfile

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "oi"))

import forest_pipeline as fp  # noqa: E402

# Lines that must not appear in a production training log
LOG_PROBLEMS = re.compile(r"warn|error|traceback|exception|setaffinity|deprecat|falling back", re.I)
ARTIFACTS = ("model.json", "model_compact.bin", "model_xgb.onnx", "thresholds.json", "manifest.json",
             "train.log", "cmssw_cfi_snippet.py")


def flavours():
    import OI_general_xgb
    import OI_pixel_xgb
    import pixel_xgb
    import seeds_xgb

    # (full training config, cfi python variable holding the module)
    return [
        (fp.full_config(pixel_xgb.CFG), "hltPhase2MuonPixelTracksHighPurityForest"),
        (fp.full_config(seeds_xgb.CFG), "hltPhase2MuonIOTrackSelectionHighPurityForest"),
        (fp.full_config(OI_pixel_xgb.CFG), "_pixelOIForestSelector"),
        (fp.full_config(OI_general_xgb.CFG), "_seedsOIForestSelector"),
    ]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, ok, what, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}" + (f"  ({detail})" if detail else ""))
        self.failures += 0 if ok else 1
        return ok


def cfi_module_params(cfi_path, var):
    """Parameters assigned in the `var = ...(` block of a cfi (literal values
    of modelPath, decisionThreshold, ptBinEdges, decisionThresholds, nFeatures);
    a clone inherits unspecified ones from its parent block."""
    text = open(cfi_path).read()
    m = re.search(rf"^{re.escape(var)} = (.*?)(?=^\S)", text + "\nEND", re.S | re.M)
    if not m:
        raise KeyError(f"{var} not found in {cfi_path}")
    block = re.sub(r"#[^\n]*", "", m.group(1))  # comments may contain ')' (e.g. "# pT [0, 2)")
    params = {}
    parent = re.match(r"(\w+)\.clone\(", block)
    if parent:
        params.update(cfi_module_params(cfi_path, parent.group(1)))

    def arg(name, kind):
        mm = re.search(rf"\b{name} = cms\.{kind}\((.*?)\)", block, re.S)
        return None if mm is None else mm.group(1)

    for name, kind in (("modelPath", "FileInPath"), ("decisionThreshold", "double"),
                       ("ptBinEdges", "vdouble"), ("decisionThresholds", "vdouble"), ("nFeatures", "int32")):
        v = arg(name, kind)
        if v is not None:
            v = ast.literal_eval("(" + v + ")")  # parenthesised: multi-line vdouble bodies
            if kind == "vdouble":
                v = list(v) if isinstance(v, tuple) else [v]
            params[name] = v
    return params


def check_model(cfg, cfi_var, cmssw_src, rep):
    d = os.path.join(cfg["output_dir"], "")
    print(f"\n{cfg['name']}  ({d})")
    missing = [a for a in ARTIFACTS if not os.path.isfile(d + a)]
    if not rep.check(not missing, "artifacts present", f"missing {missing}" if missing else ""):
        return
    th = json.load(open(d + "thresholds.json"))
    man = json.load(open(d + "manifest.json"))
    bst = xgb.Booster()
    bst.load_model(d + "model.json")
    n_trees = bst.num_boosted_rounds()
    forest = fp.read_compact_bin(d + "model_compact.bin")
    bin_md5 = md5(d + "model_compact.bin")

    rep.check(th["feature_names"] == list(cfg["production_features"]), "feature ABI in thresholds.json",
              f"{th['n_features']} features")
    rep.check(th["chosen_trees"] == n_trees == len(forest["roots"]) == th["compact_bin"]["n_trees"],
              "tree count: thresholds.json = model.json = .bin", f"{n_trees}")
    rep.check(th["compact_bin"]["md5"] == bin_md5 == man["outputs"]["model_compact.bin"],
              ".bin md5 recorded in thresholds.json and manifest.json", bin_md5)
    rep.check(all(md5(d + f) == h for f, h in man["outputs"].items()), "manifest.json output checksums")

    with tempfile.TemporaryDirectory() as tmp:
        fp.write_compact_bin(fp.forest_arrays(bst, fp16=man["config"]["compact_bin_fp16"]), tmp + "/m.bin")
        rep.check(md5(tmp + "/m.bin") == bin_md5, ".bin re-exported from model.json is byte-identical")
        fp.export_onnx(bst, th["feature_names"], tmp + "/m.onnx", man["config"]["onnx_opset"],
                       graph_name=f"muonHP_{cfg['cache_tag']}_forest")
        rep.check(md5(tmp + "/m.onnx") == md5(d + "model_xgb.onnx"), "ONNX re-exported from model.json is byte-identical")

    import onnx

    meta = {p.key: p.value for p in onnx.load(d + "model_xgb.onnx").metadata_props}
    rep.check(meta.get("feature_names", "").split(",") == th["feature_names"], "ONNX metadata feature names")
    x = np.random.default_rng(0).normal(0, 2, (20000, th["n_features"])).astype(np.float32)
    s_xgb = bst.predict(xgb.DMatrix(x))
    sess = fp.onnx_utils.make_session(d + "model_xgb.onnx")
    probs = next(o for o in sess.run(None, {sess.get_inputs()[0].name: x}) if o.ndim == 2)
    d_onnx = float(np.abs(probs[:, 1] - s_xgb).max())
    d_bin = float(np.abs(fp.compact_forest_predict(forest, x) - s_xgb).max())
    rep.check(d_onnx < fp.VERIFY_TOL and d_bin < fp.VERIFY_TOL, "ONNX / .bin replay vs model.json on random inputs",
              f"max|diff| ONNX {d_onnx:.1e}, .bin {d_bin:.1e}")

    bad = [ln.strip() for ln in open(d + "train.log", errors="replace") if LOG_PROBLEMS.search(ln)
           and "fallback -> global F2" not in ln]
    rep.check(not bad, "train.log free of warnings/errors", f"{len(bad)} lines, e.g. {bad[:2]}" if bad else "")
    ver = man.get("verification", {})
    rep.check(ver.get("compact_bin", {}).get("decision_flips") == 0, "training-time .bin replay: 0 WP decision flips",
              f"{ver.get('compact_bin', {}).get('rows_checked')} test rows")

    if not cmssw_src:
        return
    cfi = os.path.join(cmssw_src, cfg["cmssw_cfi"])
    p = cfi_module_params(cfi, cfi_var)
    rep.check(p.get("modelPath") == cfg["cmssw_bin"], f"cfi modelPath = {cfg['cmssw_bin']}", p.get("modelPath"))
    deployed = os.path.join(cmssw_src, p.get("modelPath", ""))
    rep.check(os.path.isfile(deployed) and md5(deployed) == bin_md5, "deployed .bin is byte-identical to model_compact.bin")
    rep.check(p.get("decisionThreshold") == th["global_f2_threshold"], "cfi decisionThreshold = global F2 threshold")
    rep.check(p.get("ptBinEdges") == th["pt_bin_edges"], "cfi ptBinEdges")
    rep.check(p.get("decisionThresholds") == th["pt_bin_f2_thresholds"], "cfi decisionThresholds (exact)")
    rep.check(p.get("nFeatures") == th["n_features"], "cfi nFeatures")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmssw-src", help="CMSSW src/ directory with the deployment to check")
    a = ap.parse_args()
    rep = Report()
    fl = flavours()
    for cfg, var in fl:
        check_model(cfg, var, a.cmssw_src, rep)
    if a.cmssw_src:
        data = os.path.join(a.cmssw_src, "RecoMuon/L3TrackFinder/data")
        found = sorted(os.path.relpath(os.path.join(r, f), a.cmssw_src)
                       for r, _, fs in os.walk(data) for f in fs if f.endswith(".bin"))
        expected = sorted(cfg["cmssw_bin"] for cfg, _ in fl)
        print("\nCMSSW data directory")
        rep.check(found == expected, "only the four deployed .bin files are present",
                  f"extra {sorted(set(found) - set(expected))}, missing {sorted(set(expected) - set(found))}")
    print(f"\n{'ALL CHECKS PASSED' if not rep.failures else f'{rep.failures} CHECK(S) FAILED'}")
    sys.exit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
