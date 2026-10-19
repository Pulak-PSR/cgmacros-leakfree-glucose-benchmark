"""Generic interpolation audit for ANY regularly resampled glucose (or other sensor) column.

For each series (one per value of --group-col, or the whole file) it reports whether the values look like a
piecewise-linear interpolation of sparser readings: the share of exactly-zero second differences, the spacing of the
knots (non-zero second differences, i.e. the native readings), the estimated native interval, and how well the next
native reading can be recovered from two consecutive rows. Nothing here is specific to CGMacros.

    python scripts/audit_csv.py --csv path/to/file.csv --value-col "Libre GL" [--group-col pid] [--out audit.csv]
    python scripts/audit_csv.py --csv-glob "data/CGMacros-*/CGMacros-*.csv" --value-col "Libre GL"

Interpretation: a share of exactly-zero second differences far above what noisy sensor data would give (for a real
1-minute CGM series it is close to 0) together with knots at a constant spacing means the series was interpolated;
features computed at intermediate rows then contain the next reading (look-ahead leakage).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def audit_series(x, tol=1e-6):
    x = np.asarray(x, dtype=float)
    d2 = np.diff(x, 2)
    ok = ~np.isnan(d2)
    zero_frac = float(np.mean(np.abs(d2[ok]) < tol)) if ok.any() else np.nan
    knots = np.where(np.abs(d2) > tol)[0] + 1          # NaN comparisons are False
    gaps = np.diff(knots)
    if len(gaps) >= 5:
        vals, cnt = np.unique(gaps, return_counts=True)
        period = int(vals[cnt.argmax()])
        frac_period = float(cnt.max() / len(gaps))
    else:
        period, frac_period = None, np.nan
    rec_err = np.nan
    if period and period > 2:
        phase = np.bincount(knots % period, minlength=period).argmax()
        i = np.arange(2, len(x) - period)
        age = (i - phase) % period
        sel = (age >= 2) & (age <= period - 2) & ~np.isnan(x[i]) & ~np.isnan(x[i - 1])
        ii = i[sel]
        nxt = ii + (period - age[sel])
        ok2 = nxt < len(x)
        ii, nxt, a = ii[ok2], nxt[ok2], age[sel][ok2]
        if len(ii):
            rec = x[ii] + (x[ii] - x[ii - 1]) * (period - a)
            rec_err = float(np.nanmedian(np.abs(rec - x[nxt])))
    return dict(rows=len(x), missing=int(np.isnan(x).sum()), zero_second_diff_frac=zero_frac, knots=len(knots),
                estimated_native_interval_rows=period, knots_at_that_spacing=frac_period,
                next_reading_recovery_median_abs_err=rec_err,
                verdict="INTERPOLATED (next reading recoverable from the series)"
                if (zero_frac == zero_frac and zero_frac > 0.5 and rec_err == rec_err and rec_err < 1e-3) else "no evidence of interpolation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="a CSV file")
    ap.add_argument("--csv-glob", help="a glob of CSV files, one series per file (file name is the group)")
    ap.add_argument("--value-col", required=True, help="column holding the regularly resampled values")
    ap.add_argument("--group-col", help="column identifying series inside one file (optional)")
    ap.add_argument("--out", default="audit_generic.csv")
    args = ap.parse_args()
    series = {}
    if args.csv_glob:
        for f in sorted(glob.glob(args.csv_glob)):
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            series[os.path.basename(f)] = pd.to_numeric(df[args.value_col], errors="coerce").values
    elif args.csv:
        df = pd.read_csv(args.csv)
        df.columns = [c.strip() for c in df.columns]
        if args.group_col:
            for g, sub in df.groupby(args.group_col):
                series[str(g)] = pd.to_numeric(sub[args.value_col], errors="coerce").values
        else:
            series[os.path.basename(args.csv)] = pd.to_numeric(df[args.value_col], errors="coerce").values
    else:
        ap.error("give --csv or --csv-glob")
    res = pd.DataFrame([dict(series=k, **audit_series(v)) for k, v in series.items()])
    res.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
