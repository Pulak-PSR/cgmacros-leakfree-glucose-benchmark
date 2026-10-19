"""Compares freshly computed results/*.csv with expected_results/*.csv (the numbers reported in the paper).

    python scripts/verify_results.py --out results --expected expected_results

Point estimates must agree to 1e-6 (same seed, same library versions, same thread count); bootstrap intervals and
p-values to 1e-3. Small deviations (about 1e-3) can appear with different thread counts or library versions.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
ap.add_argument("--expected", default="expected_results")
args = ap.parse_args()

CHECKS = {   # file -> key columns
    "summary.csv": ["model", "horizon", "subset"],
    "tests.csv": ["horizon", "subset", "comparator"],
    "cls_summary.csv": ["model", "horizon", "endpoint", "subset"],
    "cls_tests.csv": ["horizon", "endpoint", "subset", "comparator"],
    "summary_extra.csv": ["model", "horizon", "subset"],
    "tests_extra.csv": ["horizon", "subset", "reference", "model"],
    "cal_summary.csv": ["model", "horizon", "endpoint", "subset"],
    "by_status.csv": ["model", "horizon", "subset", "group"],
}
bad = 0
for f, keys in CHECKS.items():
    a, b = os.path.join(args.out, f), os.path.join(args.expected, f)
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f"skip {f} (missing)")
        continue
    A, B = pd.read_csv(a), pd.read_csv(b)
    m = A.merge(B, on=keys, suffixes=("_new", "_exp"))
    if len(m) != len(B):
        print(f"FAIL {f}: {len(m)} of {len(B)} expected rows found")
        bad += 1
        continue
    worst, wcol = 0.0, ""
    for c in [c for c in B.columns if c not in keys and np.issubdtype(B[c].dtype, np.number)]:
        tol = 1e-3 if any(t in c for t in ("_lo", "_hi", "p_", "AUPRC", "ECE", "BSS")) else 1e-6
        diff = float(np.nanmax(np.abs(m[c + "_new"] - m[c + "_exp"]))) if len(m) else 0.0
        if diff > tol and diff > worst:
            worst, wcol = diff, c
    if worst > 0:
        print(f"FAIL {f}: column {wcol} differs by {worst:.3g}")
        bad += 1
    else:
        print(f"ok   {f}: {len(m)} rows match")
sys.exit(1 if bad else 0)
