"""
io_xgb_common.py - configuration shared by the two IO forests (pixel_xgb.py,
seeds_xgb.py): the production features, the forest recipe and the working
points.

Features (pixel_features.io_production_features): 9 track quantities, the
endcap stub count of the matched L1 tracker muon and 36 raw quantities of the
candidates of the muon stations (chamber_match.py).

Recipe: RECIPE per chain (depth, learning rate, tree budget); the forest size is the smallest prefix within
0.2% of the best validation fake rejection.
Working points: per pT bin, the threshold that keeps 98% (pT < 3 GeV), 99%
(3-5 GeV) and 99.5% (above) of the validation muons of the signal event
(gun and Bs muons; the pileup muons of the samples are signal for the
training but not for the targets). With --wp-files (n-tuples of independent
events, e.g. RelVal samples) the thresholds of the pT bins those events
populate are set on their muons instead.
"""

import pixel_features as pf

PREFIX = {"pixel": pf.PIXEL_PREFIX, "seeds": pf.SEEDS_PREFIX}
# Forest recipe per chain (depth, learning rate, tree budget), chosen on the copy-aware test split and on independent
# RelVal events among depths 6-14: the forest has to learn the matching bands of every station from raw
# quantities, and depth pays up to 8 for the pixel tracks (9 and 12 are level with it) and up to 12 for the IO
# tracks of the seeds chain (14 is level; deeper forests need fewer trees). The tree budget of the seeds chain stops
# at 400: beyond it the copy-aware validation split keeps improving while the depth-12 forests keep more fakes at equal
# recall on RelVal events (about 15% more at 1000 trees), so the size selection must not see the larger sizes.
RECIPE = {"pixel": dict(max_depth=8, learning_rate=0.1, n_estimators=1500),
          "seeds": dict(max_depth=12, learning_rate=0.1, n_estimators=400)}


def io_config(flavour):
    prefix = PREFIX[flavour]
    return dict(
        branches=pf.io_branches(prefix),
        build=pf.build_dataset,
        build_kwargs=dict(prefix=prefix, useL1TkMuFeatures=True, useL1TkMuStubFeatures=True,
                          chamberMatch=True, signalMuonFlag=True),
        feature_workers=12,                                   # the chamber matching is the slow part of the build
        pt_feature=f"{prefix}_pt",
        drop_features=None,                                   # select production_features, in this order
        production_features=pf.io_production_features(prefix),
        wp_signal_feature=pf.SIGNAL_MUON_FLAG,
        wp_recall_targets=[0.98, 0.99, 0.995, 0.995, 0.995, 0.995],
        **RECIPE[flavour],
        prune_tol=0.002,   # forest size: the smallest within 0.2% of the best validation fake rejection
        min_child_weight=5.0,
        train_jitter=0.0,
        copy_group_features=[f"{prefix}_pt", f"{prefix}_absEta", f"{prefix}_dzErr"],
        ref_model_path=None,   # the previous generation used a different feature set: compared on RelVal events instead
    )
