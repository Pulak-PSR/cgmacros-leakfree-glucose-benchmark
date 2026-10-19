"""Main benchmark: participant-grouped 5-fold CV, baselines, CatBoost feature-group ablation, XGBoost/LightGBM,
spike endpoints, grouped SHAP. Writes results/summary.csv, tests.csv, per_participant.csv, cls_summary.csv, cls_tests.csv,
shap_groups.csv, best_iters.json, fold_assignment.csv.   Runtime: about 26 min on 8 CPU threads.

    python scripts/run_benchmark.py --data-dir path/to/cgmacros [--smoke]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool

from cgmacros_bench import config as C
from cgmacros_bench.data import find_cgm_dir, load_all
from cgmacros_bench.embeddings import get_embeddings, image_features
from cgmacros_bench.features import build_features
from cgmacros_bench.models import fit_catboost, fit_catboost_classifier, fit_logistic, fit_ridge_predict
from cgmacros_bench.sampling import assign_folds, fold_table, participant_ids, select_origins, split_fold
from cgmacros_bench.stats import (calibration_summary, classification_summary, classification_tests, holm,
                                  paired_wilcoxon_mae, part_stats, summarize_regression)
from cgmacros_bench.utils import elapsed, log, open_log, parse_args

warnings.filterwarnings("ignore")


def main():
    args = parse_args()
    open_log(args.out, "run_log.txt")
    SEED = args.seed
    cgm_dir = find_cgm_dir(args.data_dir)
    d, bio_cols = load_all(cgm_dir)
    d, groups, uniq = build_features(d, bio_cols, cgm_dir, cgm_mode="native")
    log("rows", len(d), "participants", d["pid"].nunique(), "meals", int(d["is_meal"].sum()))
    emb = get_embeddings(uniq, args.cache)
    pids = participant_ids(d, args.smoke)
    if args.smoke:
        d = d[d["pid"].isin(pids)].reset_index(drop=True)
    fold_table(pids, SEED).to_csv(os.path.join(args.out, "fold_assignment.csv"), index=False)

    base_cols = sum(groups.values(), [])
    feat_sets = {
        "CGM": groups["cgm"],
        "CGM+time": groups["cgm"] + groups["time"],
        "CGM+time+meal": groups["cgm"] + groups["time"] + groups["meal"],
        "CGM+time+meal+activity": groups["cgm"] + groups["time"] + groups["meal"] + groups["activity"],
        "CGM+time+meal+activity+clinical": base_cols,
    }                                                         # "+image" (full model) is handled separately

    D, info = select_origins(d, pids, SEED)
    stride_flag, meal_flag = info["stride_flag"], info["meal_flag"]
    log("sampled rows", len(D), "meal rows", int(meal_flag.sum()))
    Y = {h: D["y_" + h].values for h in C.HORIZON_MIN}
    pid_arr = D["pid"].values
    img_idx = D["last_img_idx"].values
    fold_of = assign_folds(pids, SEED)
    fold_arr = np.array([fold_of[p] for p in pid_arr])

    iters = 60 if args.smoke else 1500
    stats, shap_rows = [], []
    gl_now_arr = D["gl_now"].values
    ENDPOINTS = {"level140": lambda h: (gl_now_arr + Y[h]) > C.SPIKE_LEVEL,     # G(t+h) above 140 mg/dL
                 "rise30": lambda h: Y[h] >= C.SPIKE_RISE}                       # rise of at least 30 mg/dL
    naive_score = {"level140": gl_now_arr,
                   "rise30": gl_now_arr - D["gl_lag_15"].fillna(D["gl_now"]).values}   # 15-minute trend
    cls_store = {}
    pmean = {h: D.groupby("pid")["y_" + h].transform("mean").values for h in C.HORIZON_MIN}   # oracle, NOT deployable
    bst_it = {}

    def record(model, h, te, yhat):
        y = Y[h][te]
        for subset, mk in (("all", stride_flag[te]), ("meal", meal_flag[te])):
            if mk.sum():
                stats.extend(part_stats(y[mk], yhat[mk], pid_arr[te][mk], subset, model, h))

    for f in range(C.N_FOLDS):
        tr, va, te = split_fold(pid_arr, fold_arr, f, SEED)
        log(f"=== fold {f}: train {len(tr)} val {len(va)} test {len(te)} (test pids {len(np.unique(pid_arr[te]))})")
        Ximg, img_cols = image_features(emb, img_idx, np.concatenate([tr, va]))

        def matrix(cols, with_img=False):
            X = D[cols].values.astype(np.float32)
            if with_img:
                X = np.hstack([X, Ximg.astype(np.float32)])
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        names_full = base_cols + img_cols
        X_full = matrix(base_cols, True)
        sets = {k: matrix(v) for k, v in feat_sets.items()}
        sets["Full (+image)"] = X_full

        for h in C.HORIZON_MIN:
            y = Y[h]
            # ---- baselines
            record("Zero (persistence)", h, te, np.zeros(len(te)))
            record("Train-mean", h, te, np.full(len(te), y[tr].mean()))
            record("Participant-mean (oracle)", h, te, pmean[h][te])
            for nm, key in (("Ridge CGM", "CGM"), ("Ridge full", "Full (+image)")):
                record(nm, h, te, fit_ridge_predict(sets[key][tr], y[tr], sets[key][te]))
            # ---- CatBoost ablation ladder
            for key, X in sets.items():
                m = fit_catboost(X[tr], y[tr], X[va], y[va], iters)
                pred = m.predict(X[te])
                record(f"CatBoost {key}", h, te, pred)
                bst_it[(f, h, key)] = int(m.get_best_iteration())
                if key == "Full (+image)":
                    idx = np.random.RandomState(SEED).choice(len(te), size=min(1500, len(te)), replace=False)
                    sv = m.get_feature_importance(Pool(X[te][idx]), type="ShapValues")[:, :-1]
                    ms = np.abs(sv).mean(0)
                    grp_of = {c: g for g, cs in groups.items() for c in cs}
                    for c, v in zip(names_full, ms):
                        shap_rows.append(dict(fold=f, horizon=h, feature=c, group=grp_of.get(c, "image"),
                                              mean_abs_shap=float(v)))
            # ---- XGBoost / LightGBM on the full feature set
            mx = xgb.XGBRegressor(n_estimators=iters, learning_rate=0.05, max_depth=6, subsample=0.8,
                                  colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, tree_method="hist",
                                  n_jobs=C.THREADS, random_state=SEED, verbosity=0, early_stopping_rounds=60)
            mx.fit(X_full[tr], y[tr], eval_set=[(X_full[va], y[va])], verbose=False)
            record("XGBoost Full (+image)", h, te, mx.predict(X_full[te]))
            ml = lgb.LGBMRegressor(n_estimators=iters, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.7, min_child_samples=30, reg_lambda=2.0,
                                   n_jobs=C.THREADS, random_state=SEED, verbose=-1)
            ml.fit(X_full[tr], y[tr], eval_set=[(X_full[va], y[va])],
                   callbacks=[lgb.early_stopping(60, verbose=False)])
            record("LightGBM Full (+image)", h, te, ml.predict(X_full[te]))
            # ---- spike classification (same folds, same train/validation split)
            for ep, fn in ENDPOINTS.items():
                lab = fn(h).astype(int)
                if lab[tr].min() == lab[tr].max() or lab[va].min() == lab[va].max():
                    continue

                def store(model, score):
                    cls_store.setdefault((model, h, ep), []).append(pd.DataFrame(
                        {"pid": pid_arr[te], "y": lab[te], "score": score,
                         "meal": meal_flag[te], "stride": stride_flag[te]}))
                store("Naive (current level / 15-min trend)", naive_score[ep][te])
                store("Logistic (CGM)", fit_logistic(sets["CGM"][tr], lab[tr], sets["CGM"][te]))
                cc = fit_catboost_classifier(X_full[tr], lab[tr], X_full[va], lab[va], iters)
                store("CatBoost (all)", cc.predict_proba(X_full[te])[:, 1])
            log(f"  fold {f} {h} done  t={elapsed()}s")
        pd.DataFrame(stats).to_csv(os.path.join(args.out, "per_participant.csv"), index=False)

    # ---------------------------------------------------------------- aggregate
    P = pd.DataFrame(stats)
    P.to_csv(os.path.join(args.out, "per_participant.csv"), index=False)
    pd.DataFrame(shap_rows).to_csv(os.path.join(args.out, "shap_groups.csv"), index=False)
    json.dump({str(k): v for k, v in bst_it.items()}, open(os.path.join(args.out, "best_iters.json"), "w"))
    rng = np.random.RandomState(SEED)
    summarize_regression(P, rng).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    paired_wilcoxon_mae(P, ref="CatBoost Full (+image)").to_csv(os.path.join(args.out, "tests.csv"), index=False)
    cs, per_part_auc = classification_summary(cls_store, rng)
    cs.to_csv(os.path.join(args.out, "cls_summary.csv"), index=False)
    classification_tests(per_part_auc).to_csv(os.path.join(args.out, "cls_tests.csv"), index=False)
    log("DONE in", elapsed(), "s")


if __name__ == "__main__":
    main()
