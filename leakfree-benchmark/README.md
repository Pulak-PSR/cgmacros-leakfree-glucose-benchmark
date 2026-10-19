# CGMacros leak-free benchmark and CGM data audit

Code for the paper *Auditing CGMacros: Interpolated Glucose Series Leak the Future, and a Leak-Free Participant-Held-Out Benchmark for Glucose Forecasting* (Pulak Saha, University of Essex).

It does four things:

1. **Audits** the released CGMacros Libre CGM series: it is a 1-minute linear interpolation of 15-minute sensor readings, so features computed at intermediate minutes contain the next reading.
2. **Quantifies the leak** in a controlled experiment (same participants, folds, origins, target and models; only the CGM features change).
3. **Provides a leak-free, participant-held-out forecasting benchmark** (native-reading features, 5-fold participant-grouped CV, baselines, paired participant-level statistics, spike endpoints, calibration, HbA1c-group analyses).
4. **Reproduces every table and figure** of the paper and verifies the numbers.

**No CGMacros data are included or redistributed.** Download them yourself from PhysioNet and follow the data-use terms on the project page: <https://physionet.org/content/cgmacros/1.0.0/>. This repository must never contain data files (`.gitignore` excludes `data/`, `cache/`, `results/`, `*.npz`, `*.zip`).

## Install

Python 3.13.9 (other recent versions probably work but results were produced with these pinned versions). CPU only; no GPU needed.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow is used only to compute MobileNetV2 meal-photo embeddings (once, cached in `cache/`).

## Get the data

Download and unzip CGMacros v1.0.0, then point the scripts to the unzipped folder (the one that contains `CGMacros_dateshifted365/CGMacros/CGMacros-001, ...` and `bio.csv`):

```bash
export CGMACROS_DIR=/path/to/cgmacros-a-scientific-dataset-...-1.0.0     # or pass --data-dir to each script
```

## One command per result

All scripts write to `results/` (`--out`) and take `--smoke` for a quick test (12 participants, few iterations). Runtimes are for an 8-thread laptop CPU (Intel Core i5-8265U, 8 GB RAM).

| Command | What it produces | Runtime |
|---|---|---|
| `python scripts/run_audit.py` | `audit_per_participant.csv`, `audit_summary.json`, `figures/fig_audit.pdf` (audit of the CGM series; paper Section 3, Table 1, Figure 1, Appendix C) | about 4 min |
| `python scripts/audit_csv.py --csv FILE --value-col COL` | `audit_generic.csv` (the same interpolation audit for any regularly resampled column or glob of files; not used for the paper's tables) | seconds |
| `python scripts/run_benchmark.py` | `summary.csv`, `tests.csv`, `per_participant.csv`, `cls_summary.csv`, `cls_tests.csv`, `shap_groups.csv`, `fold_assignment.csv` (main benchmark; paper Section 5) | about 26 min |
| `python scripts/run_leak_experiment.py` | `leak_summary.csv`, `leak_delta.csv`, `leak_tests.csv`, `leak_per_participant.csv` (leak quantification; paper Section 3.5, Table 2, Figure 2) | about 75 min |
| `python scripts/run_extra.py` | `summary_extra.csv`, `tests_extra.csv`, `cal_summary.csv`, `by_status.csv`, `tuned_choices.csv` (LSTM/Transformer, tuned CatBoost, calibration, HbA1c groups) | about 68 min |
| `python scripts/make_figures.py` | `figures/fig_r2.pdf`, `fig_ablation.pdf`, `fig_shap.pdf`, `fig_leak.pdf` | seconds |
| `python scripts/verify_results.py` | compares `results/` with `expected_results/` (the numbers in the paper) | seconds |

Run `run_benchmark.py` before `run_extra.py` (the latter reads `results/per_participant.csv`).

## Layout

```
cgmacros_bench/      library: data loading, native-reading detection, features, sampling/folds, models, statistics
scripts/             one script per result (see table)
expected_results/    summary-level CSVs behind the paper's tables (no row-level data), for verification
requirements.txt     pinned versions
```

Key design points (details in the paper and in `cgmacros_bench/features.py`):

* The released Libre series is detected as interpolated from slope changes; the native-reading phase is estimated per run of non-missing values.
* Leak-free features use only native readings at or before the forecast origin (the last reading is held until the next arrives).
* Forecast origins must have a regular 1-minute timestamp grid and no missing reading in [t-120, t+360] minutes; CGMacros-007 has no such stretch (44 of 45 participants are analyzed).
* Folds: participants shuffled with seed 42 and dealt round-robin into 5 folds (`results/fold_assignment.csv`, also listed in the paper's appendix); 15% of training participants form the validation set.
* All uncertainty is over participants (bootstrap, paired Wilcoxon with Holm correction).

## Reproducibility notes

Results use seed 42 and `BENCH_THREADS=8` (CPU threads of the boosted trees and torch). Different thread counts or library versions can change the third decimal. `verify_results.py` reports the largest deviation per file.

## Intended and non-intended uses

Intended: evaluating glucose-change forecasting under a leak-free, participant-held-out protocol; auditing CGM datasets for resampling; teaching leakage. Not intended: dietary, clinical or treatment decisions; claims of clinical-grade accuracy; fairness claims across diabetes status (groups have 14-15 people); participant re-identification or data linkage.

## Maintenance, licence and citation

MIT licence (`LICENSE`). Maintained by the author (Pulak Saha, pulak2164@gmail.com); please report problems through the GitHub issue tracker https://github.com/Pulak-PSR/CGMacros-Multimodal-AI-for-Multiple-Horizon-Glucose-Prediction/tree/main/leakfree-benchmark. Cite the paper and software using `CITATION.cff`, and cite the CGMacros dataset paper (Das et al., Scientific Data, 2025, doi:10.1038/s41597-025-05851-7).
