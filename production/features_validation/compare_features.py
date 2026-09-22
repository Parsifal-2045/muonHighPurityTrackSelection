"""
Compare the features obtained from python build_dataset with those obtained by the C++ code in CMSSW.
To obtain a reliable feature dump from CMSSW, run on ~100 events with a single thread and set the
model's dumpFeatures flag to true.
Features should be equal at the level of floating point precision.
"""

import numpy as np
import pandas as pd

seeds_model = False

python_features_csv = "python_seeds_features.csv" if seeds_model else "python_pixel_features.csv"
cpp_features_csv = "cmssw_seeds_features.csv" if seeds_model else "cmssw_pixel_features.csv"

py = pd.read_csv(python_features_csv).drop(columns=["label"])
cpp = pd.read_csv(cpp_features_csv)  # same columns minus label

if seeds_model:
    py["event_idx"] = np.where(py["event_idx"] > 2, py["event_idx"] - 1, py["event_idx"])

m = py.merge(cpp, on=["event_idx", "track_idx"], suffixes=("_py", "_cpp"))
print(f"Matched {len(m)} / py:{len(py)} / cpp:{len(cpp)} rows")

feat_cols = [
    c[:-3]
    for c in m.columns
    if c.endswith("_py") and c not in ("event_idx_py", "track_idx_py")
]
for c in feat_cols:
    d = (m[f"{c}_py"] - m[f"{c}_cpp"]).abs()
    if d.max() > 1e-6:
        print(f"  {c:<45s} max|Δ|={d.max():.2e}  mean|Δ|={d.mean():.2e}")
