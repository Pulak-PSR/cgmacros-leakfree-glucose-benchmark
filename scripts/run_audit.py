"""Audit of the CGMacros Libre CGM series: is the released 1-minute series measured or interpolated?

For every participant it reports the share of exactly-zero second differences, the detected native-reading phase,
irregular timestamp steps, missing readings, the number of regular forecast origins, and how accurately the NEXT native
reading can be recovered from two consecutive minutes of the released series (the leak mechanism).
Outputs: audit_per_participant.csv, audit_summary.json, figures/fig_audit.pdf.

    python scripts/run_audit.py --data-dir path/to/cgmacros
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgmacros_bench import config as C
from cgmacros_bench.data import detect_native_phase, find_cgm_dir, load_all
from cgmacros_bench.features import build_features
from cgmacros_bench.utils import open_log, parse_args, log


def main():
    args = parse_args()
    open_log(args.out, "run_log_audit.txt")
    cgm_dir = find_cgm_dir(args.data_dir)
    d0, bio_cols = load_all(cgm_dir)
    d, groups, _ = build_features(d0.copy(), bio_cols, cgm_dir, cgm_mode="native")
    status = pd.cut(d0.groupby("pid")["HbA1c"].first() if "HbA1c" in d0 else d0.groupby("pid")["hba1c"].first(),
                    [0, 5.65, 6.45, 99], labels=["healthy", "prediabetes", "T2D"])
    rows, pairs = [], []
    for pid, g0 in d0.sort_values(["pid", "Timestamp"]).groupby("pid", sort=True):
        raw = pd.to_numeric(g0["Libre GL"], errors="coerce").values.astype(float)
        ts = g0["Timestamp"]
        step = ts.diff().dt.total_seconds().div(60).dropna()
        d2 = np.diff(raw, 2)
        ok = ~np.isnan(d2)
        zero_frac = float(np.mean(np.abs(d2[ok]) < 1e-6))
        ph, knots, run_id, nan_raw = detect_native_phase(raw)
        kn_gap = np.diff(knots)
        phase_counts = np.bincount((knots % C.NATIVE), minlength=C.NATIVE)
        # leak mechanism: recover the next native reading from two consecutive minutes of the released series
        gg = d[d.pid == pid]
        gl = gg["gl"].values
        age = gg["min_since_reading"].values.astype(int)
        reg = gg["regular"].values
        i = np.arange(len(gg))
        sel = reg & (age >= 2) & (age <= 13) & (i + (C.NATIVE - age) < len(gg)) & (i >= 1)
        ii = i[sel]
        rec = gl[ii] + (gl[ii] - gl[ii - 1]) * (C.NATIVE - age[ii])
        act = gl[ii + (C.NATIVE - age[ii])]
        err = np.abs(rec - act)
        if len(ii):
            take = np.random.RandomState(0).choice(len(ii), min(150, len(ii)), replace=False)
            pairs.append(np.column_stack([act[take], rec[take]]))
        rows.append(dict(pid=pid, rows=len(g0), missing_libre_rows=int(nan_raw.sum()),
                         nonunit_time_steps=int((step != 1).sum()), nonunit_share=float((step != 1).mean()), max_time_step_min=float(step.max()), nonunit_span_min=float(step[step != 1].sum()),
                         zero_second_diff_frac=zero_frac, knots=len(knots),
                         dominant_phase=int(phase_counts.argmax()),
                         frac_knots_on_dominant_phase=float(phase_counts.max() / max(phase_counts.sum(), 1)),
                         frac_knot_gaps_equal_15=float(np.mean(kn_gap == C.NATIVE)) if len(kn_gap) else np.nan,
                         phase_changes=int((np.diff(ph) != 0).sum()),
                         regular_origin_rows=int(gg["regular"].sum()),
                         recover_next_reading_median_abs_err=float(np.median(err)) if len(err) else np.nan,
                         recover_next_reading_frac_within_1=float(np.mean(err <= 1.0)) if len(err) else np.nan,
                         hba1c_group=str(status.get(pid, "NA"))))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(args.out, "audit_per_participant.csv"), index=False)
    summ = dict(participants=int(len(A)),
                zero_second_diff_frac_min=float(A.zero_second_diff_frac.min()),
                zero_second_diff_frac_median=float(A.zero_second_diff_frac.median()),
                zero_second_diff_frac_max=float(A.zero_second_diff_frac.max()),
                participants_phase0=int((A.dominant_phase == 0).sum()),
                participants_with_phase_changes=int((A.phase_changes > 0).sum()),
                participants_with_nonunit_steps=int((A.nonunit_time_steps > 0).sum()),
                participants_with_missing=int((A.missing_libre_rows > 0).sum()),
                participants_without_regular_origins=int((A.regular_origin_rows == 0).sum()),
                recover_median_abs_err_overall_median=float(A.recover_next_reading_median_abs_err.median()),
                recover_frac_within_1_median=float(A.recover_next_reading_frac_within_1.median()))
    json.dump(summ, open(os.path.join(args.out, "audit_summary.json"), "w"), indent=2)
    log(json.dumps(summ))

    # ---------------- figure: interpolated vs native readings (one participant) and recoverability of the next reading
    os.makedirs("figures", exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    pid = "CGMacros-001"
    gg = d[d.pid == pid].reset_index(drop=True)
    start = 3000 - (3000 % C.NATIVE)
    w = gg.iloc[start:start + 150]
    native = (w["min_since_reading"].values == 0)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    t = np.arange(len(w))
    ax[0].plot(t, w["gl"].values, color="0.55", lw=1.6, label="released 1-minute series")
    ax[0].step(t, w["gl_now"].values, where="post", color="k", lw=1.0, ls="--", label="last native reading (held)")
    ax[0].plot(t[native], w["gl"].values[native], "ko", ms=4, label="native 15-minute readings")
    ax[0].set_xlabel("minutes"); ax[0].set_ylabel("glucose (mg/dL)"); ax[0].set_title(f"(a) {pid}", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6.5, loc="best")
    P = np.vstack(pairs)
    ax[1].scatter(P[:, 0], P[:, 1], s=3, color="0.3", alpha=0.35, rasterized=True)
    lim = [min(P.min(), 40), max(P.max(), 400)]
    ax[1].plot(lim, lim, color="k", lw=0.6)
    ax[1].set_xlabel("actual next native reading (mg/dL)")
    ax[1].set_ylabel("recovered from two minutes (mg/dL)")
    ax[1].set_title("(b) the next reading is recoverable", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "fig_audit.pdf"))
    log("AUDIT DONE")


if __name__ == "__main__":
    main()
