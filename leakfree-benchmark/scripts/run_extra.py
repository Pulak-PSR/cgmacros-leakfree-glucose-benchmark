"""Secondary experiments on the SAME folds, origins and validation splits as run_benchmark.py:
LSTM and Transformer sequence models, validation-tuned CatBoost, calibration of the spike classifiers, and results by
HbA1c group. Requires results/per_participant.csv from run_benchmark.py. Runtime: about 68 min on 8 CPU threads.

    python scripts/run_extra.py --data-dir path/to/cgmacros [--smoke]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catboost import CatBoostRegressor

from cgmacros_bench import config as C
from cgmacros_bench.data import find_cgm_dir, load_all
from cgmacros_bench.embeddings import get_embeddings, image_features
from cgmacros_bench.features import build_features
from cgmacros_bench.models import fit_catboost_classifier, fit_logistic
from cgmacros_bench.sampling import assign_folds, participant_ids, select_origins, split_fold
from cgmacros_bench.seqmodels import SEQ_COLS, SEQ_CONFIGS, SEQ_GAP, SEQ_STEPS, SeqLSTM, SeqTF, predict, train_seq
from cgmacros_bench.stats import (calibration_summary, holm, paired_wilcoxon_mae, part_stats, summarize_by_group,
                                  summarize_regression)
from cgmacros_bench.utils import elapsed, log, open_log, parse_args
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")


def main():
    args = parse_args()
    open_log(args.out, "run_log_extra.txt")
    SEED = args.seed
    torch.set_num_threads(C.THREADS)
    torch.manual_seed(SEED)
    cgm_dir = find_cgm_dir(args.data_dir)
    d, bio_cols = load_all(cgm_dir)
    d, groups, uniq = build_features(d, bio_cols, cgm_dir, cgm_mode="native")
    emb = get_embeddings(uniq, args.cache)
    pids = participant_ids(d, args.smoke)
    if args.smoke:
        d = d[d["pid"].isin(pids)].reset_index(drop=True)
    status = d.groupby("pid")["hba1c"].first()
    base_cols = sum(groups.values(), [])
    D, info = select_origins(d, pids, SEED)
    orig_idx, rowpos = info["orig_idx"], info["rowpos"]
    stride_flag, meal_flag = info["stride_flag"], info["meal_flag"]
    Y = {h: D["y_" + h].values for h in C.HORIZON_MIN}
    Ymat = np.stack([Y[h] for h in C.HORIZON_MIN], 1)
    pid_arr = D["pid"].values
    img_idx = D["last_img_idx"].values
    fold_of = assign_folds(pids, SEED)
    fold_arr = np.array([fold_of[p] for p in pid_arr])

    # sequence windows (raw), gathered once: 12 steps at 10-minute spacing ending at the origin
    Fm = np.nan_to_num(d[SEQ_COLS].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    start = np.arange(len(d)) - rowpos
    offs = -SEQ_GAP * np.arange(SEQ_STEPS - 1, -1, -1)
    widx = np.maximum(orig_idx[:, None] + offs[None, :], start[orig_idx][:, None])
    Xraw = Fm[widx]
    log("sequence tensor", Xraw.shape)

    iters = 60 if args.smoke else 1500
    epochs = 2 if args.smoke else 15
    stats, cls_store, tuned_rows = [], {}, []
    ENDP = {"level140": lambda h: (D["gl_now"].values + Y[h]) > C.SPIKE_LEVEL,
            "rise30": lambda h: Y[h] >= C.SPIKE_RISE}

    def record(model, h, te, yhat):
        y = Y[h][te]
        for subset, mk in (("all", stride_flag[te]), ("meal", meal_flag[te])):
            if mk.sum():
                stats.extend(part_stats(y[mk], yhat[mk], pid_arr[te][mk], subset, model, h))

    grid = [dict(depth=4, learning_rate=0.05, l2_leaf_reg=5), dict(depth=6, learning_rate=0.05, l2_leaf_reg=5),
            dict(depth=8, learning_rate=0.05, l2_leaf_reg=10), dict(depth=6, learning_rate=0.03, l2_leaf_reg=20)]

    for f in range(C.N_FOLDS):
        tr, va, te = split_fold(pid_arr, fold_arr, f, SEED)
        log(f"=== fold {f}: train {len(tr)} val {len(va)} test {len(te)}")
        # ---------------- sequence models
        k = Xraw.shape[2]
        mu = Xraw[tr].reshape(-1, k).mean(0)
        sd = Xraw[tr].reshape(-1, k).std(0) + 1e-6
        Xz = torch.tensor(np.clip((Xraw - mu) / sd, -5, 5), dtype=torch.float32)
        ym, ys = Ymat[tr].mean(0), Ymat[tr].std(0)
        Yz = torch.tensor((Ymat - ym) / ys, dtype=torch.float32)
        for name, cls in (("LSTM (CGM+time+meal+activity)", SeqLSTM), ("Transformer (CGM+time+meal+activity)", SeqTF)):
            cand = []
            for ci, cfg in enumerate(SEQ_CONFIGS):
                torch.manual_seed(SEED + f)
                m = cls(Xz.shape[2], hid=cfg["hid"], drop=cfg["drop"])
                m, ne, bv = train_seq(m, Xz[tr], Yz[tr], Xz[va], Yz[va], epochs, lr=cfg["lr"], wd=cfg["wd"],
                                      noise=cfg["noise"])
                cand.append((bv, ci, m, ne))
                log(f"  fold {f} {name} cfg{ci}: epochs {ne}, best val mse {bv:.3f}  t={elapsed()}s")
            bv, ci, m, ne = min(cand, key=lambda c: c[0])
            tuned_rows.append(dict(fold=f, horizon="all", model=name, grid_index=ci, val_rmse=bv ** 0.5,
                                   **SEQ_CONFIGS[ci]))
            pr = predict(m, Xz[te]) * ys + ym
            for i, h in enumerate(C.HORIZON_MIN):
                record(name, h, te, pr[:, i])
        # ---------------- tree-model features (identical to run_benchmark.py)
        Ximg, _ = image_features(emb, img_idx, np.concatenate([tr, va]))
        X_full = np.nan_to_num(np.hstack([D[base_cols].values.astype(np.float32), Ximg.astype(np.float32)]),
                               nan=0.0, posinf=0.0, neginf=0.0)
        X_cgm = np.nan_to_num(D[groups["cgm"]].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        for h in C.HORIZON_MIN:
            y = Y[h]
            best = None                                           # tuned CatBoost (selected on validation participants)
            for gi, g in enumerate(grid):
                m = CatBoostRegressor(iterations=iters, random_seed=SEED, verbose=0, thread_count=C.THREADS,
                                      early_stopping_rounds=60, **g)
                m.fit(X_full[tr], y[tr], eval_set=(X_full[va], y[va]), use_best_model=True)
                vs = m.get_best_score()["validation"]["RMSE"]
                if best is None or vs < best[0]:
                    best = (vs, gi, m)
            record("CatBoost tuned (all)", h, te, best[2].predict(X_full[te]))
            tuned_rows.append(dict(fold=f, horizon=h, grid_index=best[1], val_rmse=best[0], **grid[best[1]]))
            for ep, fn in ENDP.items():                           # classifiers for the calibration analysis
                lab = fn(h).astype(int)
                if lab[tr].min() == lab[tr].max() or lab[va].min() == lab[va].max():
                    continue
                lr_p = fit_logistic(X_cgm[tr], lab[tr], X_cgm[te])
                cc = fit_catboost_classifier(X_full[tr], lab[tr], X_full[va], lab[va], iters)
                for mname, sc_ in (("Logistic (CGM)", lr_p), ("CatBoost (all)", cc.predict_proba(X_full[te])[:, 1])):
                    cls_store.setdefault((mname, h, ep), []).append(pd.DataFrame(
                        {"pid": pid_arr[te], "y": lab[te], "p": sc_, "meal": meal_flag[te],
                         "stride": stride_flag[te]}))
            log(f"  fold {f} {h} done  t={elapsed()}s")
        pd.DataFrame(stats).to_csv(os.path.join(args.out, "per_participant_extra.csv"), index=False)

    # ================================================================ aggregation
    PX = pd.DataFrame(stats)
    PX.to_csv(os.path.join(args.out, "per_participant_extra.csv"), index=False)
    pd.DataFrame(tuned_rows).to_csv(os.path.join(args.out, "tuned_choices.csv"), index=False)
    PM = pd.read_csv(os.path.join(args.out, "per_participant.csv"))
    rng = np.random.RandomState(SEED + 1000)
    summarize_regression(PX, rng).to_csv(os.path.join(args.out, "summary_extra.csv"), index=False)
    PA = pd.concat([PM, PX], ignore_index=True)
    trows = []
    for ref in ("CatBoost Full (+image)", "CatBoost CGM+time+meal+activity"):
        for (h, s), sub in PA.groupby(["horizon", "subset"]):
            piv = sub.assign(mae=sub["sae"] / sub["n"]).pivot(index="pid", columns="model", values="mae")
            ps, info_ = [], []
            for m in PX["model"].unique():
                dl = (piv[m] - piv[ref]).dropna()                  # positive: the new model has the higher MAE
                p = wilcoxon(dl).pvalue if (dl != 0).any() else 1.0
                ps.append(p)
                info_.append((m, float(dl.mean()), int((dl > 0).sum()), len(dl)))
            for (m, md, nb, nn_), p, pa in zip(info_, ps, holm(ps)):
                trows.append(dict(horizon=h, subset=s, reference=ref, model=m, mean_MAE_diff_model_minus_ref=md,
                                  ref_better_in=nb, of=nn_, p_raw=p, p_holm=pa))
    pd.DataFrame(trows).to_csv(os.path.join(args.out, "tests_extra.csv"), index=False)
    # results by HbA1c group (cutoffs 5.7 and 6.5 reproduce the dataset's 15/16/14 split)
    PA["group"] = PA["pid"].map(pd.cut(status, [0, 5.65, 6.45, 99], labels=["healthy", "prediabetes", "T2D"]))
    summarize_by_group(PA, "group", rng, models=["CatBoost Full (+image)", "Ridge CGM", "Zero (persistence)",
                                                 "CatBoost tuned (all)"]).to_csv(
        os.path.join(args.out, "by_status.csv"), index=False)
    cal, rel = calibration_summary(cls_store, rng)
    cal.to_csv(os.path.join(args.out, "cal_summary.csv"), index=False)
    rel.to_csv(os.path.join(args.out, "reliability.csv"), index=False)
    log("EXTRA DONE in", elapsed(), "s")


if __name__ == "__main__":
    main()
