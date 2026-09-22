#!/usr/bin/env python3
"""
dump_features.py  Dump build_dataset features to CSV for cross-validating
                   the CMSSW MuonIOTracksDNNSelector C++ feature extraction.

Output CSV columns:
    event_idx, track_idx, label, <feature_0>, <feature_1>, ..., <feature_44>

The (event_idx, track_idx) pair is used to align each row with the corresponding
track on the C++ side. event_idx is 0-based within the input file; track_idx is
0-based within the event

Usage:
    python dump_features.py /path/to/ntuple.root features_py.csv [--max-events N]
"""

import argparse
import os
import sys

import awkward as ak
import numpy as np
import uproot

# Pull build_dataset and the branch lists from the training scripts (in production/dnn).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dnn"))
import os

seed_model = False
if seed_model:
    from seeds_model import (
        build_dataset,
        l1tkMuon_branches,
        main_branch,
        stub_branches,
        tk_branches,
    )
    pt_feature_name = "muon_general_tracks_pt"

else:
    from pixel_model import (
        build_dataset,
        l1tkMuon_branches,
        main_branch,
        stub_branches,
        tk_branches,
    )
    pt_feature_name = "muon_pixel_tracks_pt"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Flat n-tuple ROOT file")
    parser.add_argument("output", help="Output CSV path")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Limit to the first N events",
    )
    args = parser.parse_args()

    # Load arrays
    print(f"Reading {args.input} ...")
    with uproot.open(args.input) as rf:
        kw = {"entry_stop": args.max_events} if args.max_events else {}
        arr = rf[main_branch].arrays(
            tk_branches + l1tkMuon_branches + stub_branches, **kw
        )

    n_events = len(arr)
    print(f"  {n_events} events loaded")

    # Sanity check: warn if the n-tuple lacks stub branches (would mean the
    # Python side produces 38 features and the C++ produces 45  diff fails).
    if "L1TkMuStub_parentL1TkMu" not in arr.fields:
        print("  WARNING: stub branches absent  Python will skip features 29-35.")
        print("           Make sure CMSSW is run with useStubFeatures=false to match.")

    # Compute (event_idx, track_idx) for the surviving tracks
    # build_dataset filters on `pt > 0` then ak.flatten()s
    # in event-major order.
    # Mirror that exactly so the rows of X line up with our index arrays.
    mask = arr[pt_feature_name] > 0
    n_tracks_per_event = ak.to_numpy(ak.num(arr[pt_feature_name]))

    event_idx_jagged = ak.unflatten(
        np.repeat(np.arange(n_events, dtype=np.int64), n_tracks_per_event),
        n_tracks_per_event,
    )
    event_idx_flat = ak.to_numpy(ak.flatten(event_idx_jagged[mask])).astype(np.int64)

    track_idx_jagged = ak.local_index(arr[pt_feature_name], axis=1)
    track_idx_flat = ak.to_numpy(ak.flatten(track_idx_jagged[mask])).astype(np.int64)

    n_expected = len(event_idx_flat)
    print(f"  {n_expected} tracks survive pt > 0")

    # Run build_dataset
    file_labels = np.zeros(n_events, dtype=np.int32)
    X, y, _, feature_names = build_dataset(
        arr, file_labels, useL1TkMuFeatures=True, verbose=False
    )
    print(f"  build_dataset  X{tuple(X.shape)}, {len(feature_names)} features")

    # Verify alignment was preserved
    # build_dataset drops rows where any feature is non-finite. If that
    # happens, our (event_idx, track_idx) arrays no longer match X row-by-row.
    if X.shape[0] != n_expected:
        n_dropped = n_expected - X.shape[0]
        print(f"\nERROR: build_dataset dropped {n_dropped} non-finite rows.")
        print("Row-by-row alignment with (event_idx, track_idx) is lost.")
        print("Quick fix: comment out the `if not finite_mask.all()` block in")
        print("build_dataset so all rows are kept.")
        sys.exit(1)

    # Write CSV
    print(f"  Writing {args.output} ...")
    header = "event_idx,track_idx,label," + ",".join(feature_names)

    rows = np.column_stack(
        [
            event_idx_flat[:, None],
            track_idx_flat[:, None],
            y[:, None].astype(np.int64),
            X.astype(np.float64),  # widen so %.9e is meaningful
        ]
    )
    fmt = ["%d", "%d", "%d"] + ["%.9e"] * X.shape[1]
    np.savetxt(args.output, rows, fmt=fmt, delimiter=",", header=header, comments="")

    print(f"  Done. {X.shape[0]} rows  {X.shape[1] + 3} cols.")


if __name__ == "__main__":
    main()
