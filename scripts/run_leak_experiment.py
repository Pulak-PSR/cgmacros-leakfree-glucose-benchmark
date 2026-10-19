"""Quantifies how much the interpolated 1-minute CGM series inflates accuracy.

Same participants, same folds, same forecast origins, same target (change from the last native reading to the glucose
at t+h) and the same models; ONLY the CGM features differ:
  native         leak-free benchmark features (last native reading held; lags in multiples of 15 minutes)
  interp_same    the same feature definitions computed from the released interpolated series
  interp_minute  a typical naive pipeline on the released series (1-, 2-, 3-, 5-, 10-minute lags, 1-minute differences)
Models: ridge and CatBoost, each with CGM-only features and with all features.
Outputs: leak_per_participant.csv, leak_summary.csv, leak_delta.csv, leak_tests.csv, leak_by_staleness.csv.
Horizons: 15, 30, 60, 180 and 360 minutes. Runtime: about 75 min on 8 CPU threads.

    python scripts/run_leak_experiment.py --data-dir path/to/cgmacros [--smoke]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cgmacros_bench import config as C
from cgmacros_bench.data import find_cgm_dir, load_all
from cgmacros_bench.embeddings import get_embeddings, image_features
from cgmacros_bench.features import build_features
from cgmacros_bench.models import fit_catboost, fit_ridge_predict
from cgmacros_bench.sampling import assign_folds, participant_ids, select_origins, split_fold
from cgmacros_bench.stats import holm, part_stats, pooled, summarize_regression
from cgmacros_bench.utils import elapsed, log, open_log, parse_args

warnings.filterwarnings("ignore")
MODES = ["native", "interp_same", "interp_minute"]
BINS = {"all_age_1_4": (1, 4), "all_age_5_9": (5, 9), "all_age_10_14": (10, 14)}   # age (minutes) of the last native reading
# Origins exactly at native readings (age 0) are added as TEST-ONLY origins (never used for training or validation), so
# the age-0 subset covers every participant without changing any other result.


def main():
    args = parse_args()
    open_log(args.out, "run_log_leak.txt")
    SEED = args.seed
    cgm_dir = find_cgm_dir(args.data_dir)
    d0, bio_cols = load_all(cgm_dir)
    stats = []
    ref_check = None
    for mode in MODES:
        d, groups, uniq = build_features(d0.copy(), bio_cols, cgm_dir, cgm_mode=mode, horizons=C.LEAK_HORIZON_MIN)
        emb = get_embeddings(uniq, args.cache)
        pids = participant_ids(d, args.smoke)
        if args.smoke:
            d = d[d["pid"].isin(pids)].reset_index(drop=True)
        D0, info = select_origins(d, pids, SEED)
        use = np.zeros(len(d), bool)
        use[info["orig_idx"]] = True
        age0 = (d["min_since_reading"].values == 0) & d["regular"].values
        keep = use | age0
        sf = np.zeros(len(d), bool); sf[info["orig_idx"]] = info["stride_flag"]
        mf = np.zeros(len(d), bool); mf[info["orig_idx"]] = info["meal_flag"]
        D = d[keep].reset_index(drop=True)
        stride_flag, meal_flag = sf[keep], mf[keep]
        test_only = (~use)[keep]                       # age-0 origins that are not already regular origins
        pid_arr, img_idx = D["pid"].values, D["last_img_idx"].values
        age = D["min_since_reading"].values
        sig = (len(D), float(D["y_h3"].sum()))          # identical across modes
        ref_check = sig if ref_check is None else ref_check
        assert sig == ref_check, "forecast origins differ between modes"          # identical origins and targets
        fold_of = assign_folds(pids, SEED)
        fold_arr = np.array([fold_of[p] for p in pid_arr])
        Y = {h: D["y_" + h].values for h in C.LEAK_HORIZON_MIN}
        base_cols = sum(groups.values(), [])
        iters = 60 if args.smoke else 1500

        def record(model, h, te, yhat):
            y = Y[h][te]
            subsets = [("all", stride_flag[te]), ("meal", meal_flag[te])]
            subsets += [(nm, stride_flag[te] & (age[te] >= lo) & (age[te] <= hi)) for nm, (lo, hi) in BINS.items()]
            subsets += [("native_origin", age[te] == 0)]
            for subset, mk in subsets:
                if mk.sum():
                    stats.extend(part_stats(y[mk], yhat[mk], pid_arr[te][mk], subset, f"{model} [{mode}]", h))

        for f in range(C.N_FOLDS):
            tr, va, te = split_fold(pid_arr, fold_arr, f, SEED)
            tr, va = tr[~test_only[tr]], va[~test_only[va]]
            Ximg, _ = image_features(emb, img_idx, np.concatenate([tr, va]))
            X_cgm = np.nan_to_num(D[groups["cgm"]].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            X_full = np.nan_to_num(np.hstack([D[base_cols].values.astype(np.float32), Ximg]),
                                   nan=0.0, posinf=0.0, neginf=0.0)
            for h in C.LEAK_HORIZON_MIN:
                y = Y[h]
                record("Ridge CGM", h, te, fit_ridge_predict(X_cgm[tr], y[tr], X_cgm[te]))
                record("Ridge full", h, te, fit_ridge_predict(X_full[tr], y[tr], X_full[te]))
                for nm, X in (("CatBoost CGM", X_cgm), ("CatBoost Full (+image)", X_full)):
                    m = fit_catboost(X[tr], y[tr], X[va], y[va], iters)
                    record(nm, h, te, m.predict(X[te]))
            log(f"  mode {mode} fold {f} done  t={elapsed()}s")
        pd.DataFrame(stats).to_csv(os.path.join(args.out, "leak_per_participant.csv"), index=False)

    P = pd.DataFrame(stats)
    P.to_csv(os.path.join(args.out, "leak_per_participant.csv"), index=False)
    rng = np.random.RandomState(SEED + 2000)
    summarize_regression(P, rng).to_csv(os.path.join(args.out, "leak_summary.csv"), index=False)

    # paired differences (leaky minus native) with participant bootstrap and Wilcoxon on per-participant MAE
    deltas, tests = [], []
    base_models = ["Ridge CGM", "Ridge full", "CatBoost CGM", "CatBoost Full (+image)"]
    for (h, s), sub in P.groupby(["horizon", "subset"]):
        piv_pids = sorted(sub["pid"].unique())
        for bm in base_models:
            nat = sub[sub.model == f"{bm} [native]"].set_index("pid")
            for mode in MODES[1:]:
                lk = sub[sub.model == f"{bm} [{mode}]"].set_index("pid")
                common = sorted(set(nat.index) & set(lk.index))
                if len(common) < 10:
                    continue
                a, b = nat.loc[common].reset_index(), lk.loc[common].reset_index()
                pt_n, pt_l = pooled(a), pooled(b)
                bs = []
                for _ in range(1000):
                    ix = rng.choice(len(common), len(common), replace=True)
                    bs.append(pooled(b.iloc[ix])[:2] - pooled(a.iloc[ix])[:2])
                bs = np.array(bs)
                deltas.append(dict(model=bm, mode=mode, horizon=h, subset=s, participants=len(common),
                                   R2_native=pt_n[0], R2_leaky=pt_l[0], dR2=pt_l[0] - pt_n[0],
                                   dR2_lo=np.percentile(bs[:, 0], 2.5), dR2_hi=np.percentile(bs[:, 0], 97.5),
                                   MAE_native=pt_n[1], MAE_leaky=pt_l[1], dMAE=pt_l[1] - pt_n[1],
                                   dMAE_lo=np.percentile(bs[:, 1], 2.5), dMAE_hi=np.percentile(bs[:, 1], 97.5)))
                dl = (b["sae"] / b["n"]).values - (a["sae"] / a["n"]).values        # per-participant MAE: leaky - native
                try:
                    p = wilcoxon(dl).pvalue if (dl != 0).any() else 1.0
                except ValueError:
                    p = 1.0
                tests.append(dict(model=bm, mode=mode, horizon=h, subset=s, mean_MAE_diff=float(dl.mean()),
                                  leaky_lower_in=int((dl < 0).sum()), of=len(dl), p_raw=p))
    pd.DataFrame(deltas).to_csv(os.path.join(args.out, "leak_delta.csv"), index=False)
    T = pd.DataFrame(tests)
    for (h, s), idx in T.groupby(["horizon", "subset"]).groups.items():          # Holm within (horizon, subset)
        T.loc[idx, "p_holm"] = holm(T.loc[idx, "p_raw"].values)
    T.to_csv(os.path.join(args.out, "leak_tests.csv"), index=False)
    log("LEAK DONE in", elapsed(), "s")


if __name__ == "__main__":
    main()
