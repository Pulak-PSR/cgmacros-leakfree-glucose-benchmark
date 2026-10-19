"""Figures from the CSV outputs of the benchmark scripts (vector PDF, grayscale-safe).
fig_audit.pdf is written by run_audit.py (it needs the raw data).

    python scripts/make_figures.py --out results --fig-dir figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
ap.add_argument("--fig-dir", default="figures")
args = ap.parse_args()
os.makedirs(args.fig_dir, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.0})
R = lambda f: pd.read_csv(os.path.join(args.out, f))
HZ = ["h1", "h3", "h6"]
HL = {"h1": "+1 h", "h3": "+3 h", "h6": "+6 h"}
S = R("summary.csv")

# ---- Fig: pooled R2 of key models (all time points, meal onsets)
key = ["Zero (persistence)", "Ridge CGM", "Ridge full", "CatBoost CGM", "CatBoost Full (+image)",
       "XGBoost Full (+image)", "LightGBM Full (+image)"]
lab = {"Zero (persistence)": "Zero", "Ridge CGM": "Ridge (CGM)", "Ridge full": "Ridge (all)",
       "CatBoost CGM": "CatBoost (CGM)", "CatBoost Full (+image)": "CatBoost (all)",
       "XGBoost Full (+image)": "XGBoost (all)", "LightGBM Full (+image)": "LightGBM (all)"}
mk = ["x", "s", "D", "o", "^", "v", "P"]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
for ax, sub, title in zip(axes, ["all", "meal"], ["(a) All time points", "(b) Meal onsets"]):
    for i, m in enumerate(key):
        d = S[(S.model == m) & (S.subset == sub)].set_index("horizon").reindex(HZ)
        x = np.arange(3) + (i - 3) * 0.06
        ax.errorbar(x, d.R2, yerr=[d.R2 - d.R2_lo, d.R2_hi - d.R2], fmt=mk[i] + "-", ms=3, capsize=1.2,
                    elinewidth=0.5, label=lab[m], color=plt.cm.Greys(0.25 + 0.7 * (i % 4) / 3))
    ax.set_xticks(range(3)); ax.set_xticklabels([HL[h] for h in HZ]); ax.set_title(title, fontsize=8)
    ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3, lw=0.3)
axes[0].set_ylabel("Pooled $R^2$")
hd, lb = axes[0].get_legend_handles_labels()
fig.legend(hd, lb, loc="lower center", ncol=7, frameon=False, handlelength=1.4, columnspacing=0.8)
fig.tight_layout(pad=0.3, rect=[0, 0.1, 1, 1]); fig.savefig(os.path.join(args.fig_dir, "fig_r2.pdf")); plt.close(fig)

# ---- Fig: ablation ladder
ladder = ["CatBoost CGM", "CatBoost CGM+time", "CatBoost CGM+time+meal", "CatBoost CGM+time+meal+activity",
          "CatBoost CGM+time+meal+activity+clinical", "CatBoost Full (+image)"]
ll = ["CGM", "+time", "+meal", "+activity", "+clinical", "+image"]
fig, ax = plt.subplots(figsize=(4.6, 2.3))
hatch = ["", "///", "..."]
for j, h in enumerate(HZ):
    d = S[(S.horizon == h) & (S.subset == "all")].set_index("model").reindex(ladder)
    ax.bar(np.arange(6) + (j - 1) * 0.26, d.R2, 0.26, yerr=[d.R2 - d.R2_lo, d.R2_hi - d.R2], capsize=1,
           error_kw=dict(lw=0.5), color=plt.cm.Greys(0.35 + 0.25 * j), edgecolor="k", linewidth=0.4,
           hatch=hatch[j], label=HL[h])
ax.set_xticks(range(6)); ax.set_xticklabels(ll); ax.set_ylabel("Pooled $R^2$ (all time points)")
ax.set_ylim(0, 0.62); ax.legend(frameon=False, ncol=3, loc="upper center"); ax.grid(axis="y", alpha=0.3, lw=0.3)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(args.fig_dir, "fig_ablation.pdf")); plt.close(fig)

# ---- Fig: grouped SHAP shares
sh = R("shap_groups.csv")
g = sh.groupby(["horizon", "feature", "group"]).mean_abs_shap.mean().reset_index()
tot = g.groupby(["horizon", "group"]).mean_abs_shap.sum().unstack().reindex(HZ)
tot = tot.div(tot.sum(axis=1), axis=0) * 100
order = [c for c in ["cgm", "time", "meal", "activity", "clinical", "image"] if c in tot.columns]
fig, ax = plt.subplots(figsize=(4.6, 1.8))
left = np.zeros(3)
for i, c in enumerate(order):
    ax.barh([HL[h] for h in HZ], tot[c].values, left=left, color=plt.cm.Greys(0.15 + 0.14 * i), edgecolor="k",
            linewidth=0.4, hatch=["", "//", "..", "xx", "\\\\", "--"][i], label=c)
    left += tot[c].values
ax.invert_yaxis(); ax.set_xlabel("Share of mean |SHAP| (%)"); ax.set_xlim(0, 100)
ax.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.3), handlelength=1, columnspacing=0.6)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(args.fig_dir, "fig_shap.pdf")); plt.close(fig)

# ---- Fig: leak quantification (five horizons)
if os.path.exists(os.path.join(args.out, "leak_summary.csv")):
    L = R("leak_summary.csv"); LD = R("leak_delta.csv")
    LH = ["m15", "m30", "h1", "h3", "h6"]
    LHL = {"m15": "+15 min", "m30": "+30 min", "h1": "+1 h", "h3": "+3 h", "h6": "+6 h"}
    modes = [("native", "native readings (leak-free)"), ("interp_same", "interpolated series, same features"),
             ("interp_minute", "interpolated series, minute-level features")]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.7))
    w = 0.26
    for j, (mode, lbl) in enumerate(modes):
        d = L[(L.model == f"CatBoost Full (+image) [{mode}]") & (L.subset == "all")].set_index("horizon").reindex(LH)
        ax[0].bar(np.arange(5) + (j - 1) * w, d.R2, w, yerr=[d.R2 - d.R2_lo, d.R2_hi - d.R2], capsize=1,
                  error_kw=dict(lw=0.5), color=plt.cm.Greys(0.3 + 0.25 * j), edgecolor="k", linewidth=0.4,
                  hatch=["", "///", "..."][j], label=lbl)
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels([LHL[h] for h in LH]); ax[0].set_ylabel("Pooled $R^2$")
    ax[0].set_title("(a) CatBoost, all features", fontsize=8); ax[0].legend(frameon=False, loc="upper right", fontsize=6)
    ax[0].set_ylim(0, 1.15); ax[0].grid(axis="y", alpha=0.3, lw=0.3)
    bins = ["native_origin", "all_age_1_4", "all_age_5_9", "all_age_10_14"]; bl = ["0", "1-4", "5-9", "10-14"]
    mkr = {"m15": "o", "m30": "s", "h1": "^"}
    for mode, ls, col in (("interp_same", "-", "k"), ("interp_minute", "--", "0.45")):
        for h in ("m15", "m30", "h1"):
            d = LD[(LD["model"] == "CatBoost CGM") & (LD["mode"] == mode) & (LD.horizon == h)].set_index("subset").reindex(bins)
            ax[1].plot(range(4), d.dR2, ls, marker=mkr[h], ms=3.5, color=col,
                       label=f"{LHL[h]}, {'same feat.' if mode == 'interp_same' else 'minute-level'}")
    ax[1].axhline(0, color="k", lw=0.4)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(bl); ax[1].set_xlabel("age of the last native reading (minutes)")
    ax[1].set_ylabel(r"$\Delta R^2$ (interpolated minus native)"); ax[1].set_title("(b) CatBoost, CGM features only", fontsize=8)
    ax[1].legend(frameon=False, fontsize=5.6, ncol=2, loc="upper left"); ax[1].grid(alpha=0.3, lw=0.3)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(args.fig_dir, "fig_leak.pdf")); plt.close(fig)
print("figures written to", args.fig_dir)
