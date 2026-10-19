"""Metrics and inference. All uncertainty is computed over PARTICIPANTS (the unit of independence), not rows."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import N_BOOT, N_BOOT_CLS


def part_stats(y, p, pid, subset, model, h):
    """Per-participant sufficient statistics (n, sum y, sum y^2, SSE, SAE) from which pooled R2/MAE/RMSE follow."""
    rows = []
    for u in np.unique(pid):
        m = pid == u
        e = y[m] - p[m]
        rows.append(dict(model=model, horizon=h, subset=subset, pid=u, n=int(m.sum()),
                         sy=float(y[m].sum()), syy=float((y[m] ** 2).sum()),
                         sse=float((e ** 2).sum()), sae=float(np.abs(e).sum())))
    return rows


def holm(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def pooled(sub):
    """Pooled out-of-fold R2 (against the pooled mean change), MAE and RMSE from per-participant statistics."""
    n = sub["n"].sum()
    sy = sub["sy"].sum()
    syy = sub["syy"].sum()
    sst = syy - sy ** 2 / n
    return np.array([1 - sub["sse"].sum() / sst, sub["sae"].sum() / n, np.sqrt(sub["sse"].sum() / n), n])


def summarize_regression(P, rng, n_boot=N_BOOT):
    """Pooled metrics with 95% participant-bootstrap intervals, plus per-participant and within-participant R2.
    `rng` is a numpy RandomState; it is consumed in a fixed order so results are reproducible."""
    rows = []
    for (m, h, s), sub in P.groupby(["model", "horizon", "subset"]):
        pt = pooled(sub)
        bs = np.array([pooled(sub.iloc[rng.choice(len(sub), len(sub), replace=True)])[:3] for _ in range(n_boot)])
        row = dict(model=m, horizon=h, subset=s, n=int(pt[3]), participants=len(sub))
        for j, k in enumerate(["R2", "MAE", "RMSE"]):
            row[k] = pt[j]
            row[k + "_lo"] = np.percentile(bs[:, j], 2.5)
            row[k + "_hi"] = np.percentile(bs[:, j], 97.5)
        r2v = (1 - sub["sse"] / (sub["syy"] - sub["sy"] ** 2 / sub["n"])).values
        row["R2_participant_median"] = float(np.median(r2v))
        # within-participant R2: each participant against that participant's own mean change, averaged over people
        row["R2_within_mean"] = float(np.mean(r2v))
        bm = [np.mean(r2v[rng.choice(len(r2v), len(r2v), replace=True)]) for _ in range(n_boot)]
        row["R2_within_lo"], row["R2_within_hi"] = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_wilcoxon_mae(P, ref, comparators=None, sign="ref_minus_comp"):
    """Paired Wilcoxon signed-rank tests on per-participant MAE, Holm-corrected within (horizon, subset).
    Returns a long table; `ref` is the reference model."""
    rows = []
    for (h, s), sub in P.groupby(["horizon", "subset"]):
        piv = sub.assign(mae=sub["sae"] / sub["n"]).pivot(index="pid", columns="model", values="mae")
        names = [m for m in (comparators if comparators is not None else piv.columns) if m != ref and m in piv.columns]
        ps, info = [], []
        for m in names:
            dlt = (piv[ref] - piv[m]).dropna()
            try:
                p = wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
            except ValueError:
                p = 1.0
            ps.append(p)
            info.append((m, float(dlt.mean()), int((dlt < 0).sum()), len(dlt)))
        for (m, md, nb, nn), p, pa in zip(info, ps, holm(ps) if ps else []):
            rows.append(dict(horizon=h, subset=s, reference=ref, comparator=m, mean_MAE_diff_ref_minus_comp=md,
                             ref_better_in=nb, of=nn, p_raw=p, p_holm=pa))
    return pd.DataFrame(rows)


def classification_summary(cls_store, rng, n_boot=N_BOOT_CLS):
    """AUROC/AUPRC with participant-bootstrap intervals and per-participant AUROC for every stored score set.
    cls_store: {(model, horizon, endpoint): [DataFrame(pid, y, score, meal, stride) per fold]}.
    Returns (summary frame, per-participant AUROC dict)."""
    crow, per_part_auc = [], {}
    for (m, h, ep), parts in cls_store.items():
        C = pd.concat(parts, ignore_index=True)
        for subset, mk in (("all", C["stride"].values), ("meal", C["meal"].values)):
            c = C[mk]
            if c["y"].nunique() < 2:
                continue
            up = np.array(sorted(c["pid"].unique()))
            pidx = {u: np.where(c["pid"].values == u)[0] for u in up}
            y_, s_ = c["y"].values, c["score"].values
            auc = roc_auc_score(y_, s_)
            ap = average_precision_score(y_, s_)
            bs_auc, bs_ap = [], []
            for _ in range(n_boot):
                cnt = np.bincount(rng.choice(len(up), len(up), replace=True), minlength=len(up))
                w = np.zeros(len(c))
                for k, u in enumerate(up):
                    if cnt[k]:
                        w[pidx[u]] = cnt[k]
                if len(np.unique(y_[w > 0])) < 2:
                    continue
                bs_auc.append(roc_auc_score(y_, s_, sample_weight=w))
                bs_ap.append(average_precision_score(y_, s_, sample_weight=w))
            pa = {}
            for u in up:
                yy = y_[pidx[u]]
                if yy.min() != yy.max():
                    pa[u] = roc_auc_score(yy, s_[pidx[u]])
            per_part_auc[(m, h, ep, subset)] = pa
            crow.append(dict(model=m, horizon=h, endpoint=ep, subset=subset, n=len(c),
                             prevalence=float(y_.mean()), AUROC=auc,
                             AUROC_lo=np.percentile(bs_auc, 2.5), AUROC_hi=np.percentile(bs_auc, 97.5),
                             AUPRC=ap, AUPRC_lo=np.percentile(bs_ap, 2.5), AUPRC_hi=np.percentile(bs_ap, 97.5),
                             AUROC_participant_median=float(np.median(list(pa.values()))) if pa else np.nan,
                             participants_with_both_classes=len(pa)))
    return pd.DataFrame(crow), per_part_auc


def classification_tests(per_part_auc, refc="CatBoost (all)"):
    """Paired Wilcoxon on per-participant AUROC (>= 15 participants with both classes), Holm within endpoint/horizon."""
    rows = []
    for (h, ep, subset) in sorted({(k[1], k[2], k[3]) for k in per_part_auc}):
        ps, info = [], []
        for m in sorted({k[0] for k in per_part_auc}):
            if m == refc:
                continue
            a, b = per_part_auc.get((refc, h, ep, subset), {}), per_part_auc.get((m, h, ep, subset), {})
            common = sorted(set(a) & set(b))
            if len(common) < 15:
                continue
            dl = np.array([a[u] - b[u] for u in common])
            try:
                p = wilcoxon(dl).pvalue if (dl != 0).any() else 1.0
            except ValueError:
                p = 1.0
            ps.append(p)
            info.append((m, float(dl.mean()), int((dl > 0).sum()), len(common)))
        for (m, md, nb, nn), p, pa_ in zip(info, ps, holm(ps) if ps else []):
            rows.append(dict(horizon=h, endpoint=ep, subset=subset, reference=refc, comparator=m,
                             mean_AUROC_diff=md, ref_better_in=nb, of=nn, p_raw=p, p_holm=pa_))
    return pd.DataFrame(rows)


def calibration_summary(cls_store, rng, n_boot=300):
    """Brier score, Brier skill score (vs the prevalence-only forecast) and ECE (10 equal-width bins)."""
    crow, rel = [], []
    for (m, h, ep), parts in cls_store.items():
        C = pd.concat(parts, ignore_index=True)
        for subset, mk in (("all", C["stride"].values), ("meal", C["meal"].values)):
            c = C[mk]
            if c["y"].nunique() < 2:
                continue
            y_, p_ = c["y"].values.astype(float), c["p"].values
            prev = y_.mean()
            brier = np.mean((p_ - y_) ** 2)
            ref_b = prev * (1 - prev)
            bins = np.minimum((p_ * 10).astype(int), 9)
            ece = 0.0
            for b in range(10):
                mb = bins == b
                if mb.sum():
                    ece += mb.mean() * abs(p_[mb].mean() - y_[mb].mean())
                    rel.append(dict(model=m, horizon=h, endpoint=ep, subset=subset, bin=b, n=int(mb.sum()),
                                    mean_pred=float(p_[mb].mean()), obs_rate=float(y_[mb].mean())))
            up = np.array(sorted(c["pid"].unique()))
            pidx = {u: np.where(c["pid"].values == u)[0] for u in up}
            bss_b = []
            for _ in range(n_boot):
                cnt = np.bincount(rng.choice(len(up), len(up), replace=True), minlength=len(up))
                w = np.zeros(len(c))
                for k, u in enumerate(up):
                    if cnt[k]:
                        w[pidx[u]] = cnt[k]
                pv = np.sum(w * y_) / w.sum()
                bss_b.append(1 - (np.sum(w * (p_ - y_) ** 2) / w.sum()) / (pv * (1 - pv)))
            crow.append(dict(model=m, horizon=h, endpoint=ep, subset=subset, n=len(c), prevalence=prev, brier=brier,
                             brier_ref=ref_b, BSS=1 - brier / ref_b, BSS_lo=np.percentile(bss_b, 2.5),
                             BSS_hi=np.percentile(bss_b, 97.5), ECE=ece))
    return pd.DataFrame(crow), pd.DataFrame(rel)


def summarize_by_group(P, group_col, rng, models=None, n_boot=300):
    """Pooled R2/MAE with participant-bootstrap intervals within subgroups of participants."""
    rows = []
    Q = P if models is None else P[P.model.isin(models)]
    for (m, h, s, g), sub in Q.groupby(["model", "horizon", "subset", group_col]):
        pt = pooled(sub)
        bs = np.array([pooled(sub.iloc[rng.choice(len(sub), len(sub), replace=True)]) for _ in range(n_boot)])
        rows.append(dict(model=m, horizon=h, subset=s, group=g, participants=len(sub), n=int(sub["n"].sum()),
                         R2=pt[0], R2_lo=np.percentile(bs[:, 0], 2.5), R2_hi=np.percentile(bs[:, 0], 97.5),
                         MAE=pt[1], MAE_lo=np.percentile(bs[:, 1], 2.5), MAE_hi=np.percentile(bs[:, 1], 97.5)))
    return pd.DataFrame(rows)
