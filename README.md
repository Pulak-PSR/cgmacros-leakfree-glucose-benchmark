# cgmacros-leakfree-glucose-benchmark

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CI](https://github.com/Pulak-PSR/cgmacros-leakfree-glucose-benchmark/actions/workflows/ci.yml/badge.svg)
![Data](https://img.shields.io/badge/data-not%20included-lightgrey)

**Leak-free, participant-held-out benchmark for glucose forecasting on CGMacros, plus an audit of the released CGM series.**

Code for the paper *Auditing CGMacros: Interpolated Glucose Series Leak the Future, and a Leak-Free Participant-Held-Out Benchmark for Glucose Forecasting* (Pulak Saha, Independent Researcher).

## Key findings

* **The released 1-minute CGM series is an interpolation.** It is a piecewise-linear interpolation of the sensor's 15-minute readings (93.7 to 96.2% of second differences are exactly zero in every participant), so any feature at an intermediate minute contains the next reading, which can be recovered to about 1e-6 mg/dL from two consecutive rows.
* **The leak inflates short-horizon accuracy a lot.** With participants, folds, origins and models held fixed and only the CGM features changed, pooled R2 of CatBoost (all features) rises from 0.49, 0.38 and 0.35 (native readings) to 0.86, 0.63 and 0.42 (interpolated series, same features) and 0.89, 0.71 and 0.46 (minute-level features) at +15, +30 and +60 minutes. At 3 and 6 hours there is no benefit.
* **Leak-free reference results.** CatBoost with all features reaches R2 of 0.35, 0.47 and 0.41 at 1, 3 and 6 hours; a CGM-only linear model already explains 0.23 to 0.34, and activity, clinical and image features add at most 0.03.

The sensor interval is inferred from the data and has not been confirmed with the dataset creators or the manufacturer. This work does not claim that any published study is affected.

## What it does

1. **Audits** the released CGMacros Libre CGM series (`scripts/run_audit.py`; `scripts/audit_csv.py` works on any regularly resampled column).
2. **Quantifies the leak** in a controlled experiment (`scripts/run_leak_experiment.py`).
3. **Provides a leak-free, participant-held-out forecasting benchmark**: native-reading features, 5-fold participant-grouped CV, baselines, paired participant-level statistics, spike endpoints, calibration and HbA1c-group analyses.
4. **Reproduces every table and figure** of the paper; the paper's tables are the reference values for a rerun.

**No CGMacros data are included or redistributed.** Download them yourself from PhysioNet and respect its licence, Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (attribution, non-commercial use, share-alike): <https://physionet.org/content/cgmacros/1.0.0/>. This repository must never contain data files (`.gitignore` excludes `data/`, `cache/`, `results/`, `*.npz`, `*.zip`).

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

Run `run_benchmark.py` before `run_extra.py` (the latter reads `results/per_participant.csv`).

## Design

Key design points (details in the paper and in `cgmacros_bench/features.py`):

* The released Libre series is detected as interpolated from slope changes; the native-reading phase is estimated per run of non-missing values.
* Leak-free features use only native readings at or before the forecast origin (the last reading is held until the next arrives).
* Forecast origins must have a regular 1-minute timestamp grid and no missing reading in [t-120, t+360] minutes; CGMacros-007 has no such stretch (44 of 45 participants are analyzed).
* Folds: participants shuffled with seed 42 and dealt round-robin into 5 folds (`results/fold_assignment.csv`, also listed in the paper's appendix); 15% of training participants form the validation set.
* All uncertainty is over participants (bootstrap, paired Wilcoxon with Holm correction).

## Reproducibility notes

Results use seed 42 and `BENCH_THREADS=8` (CPU threads of the boosted trees and torch). Different thread counts or library versions can change the third decimal. Compare your `results/*.csv` with the tables in the paper (Appendix tables list the main numbers).

## Intended and non-intended uses

Intended: evaluating glucose-change forecasting under a leak-free, participant-held-out protocol; auditing CGM datasets for resampling; teaching leakage. Not intended: dietary, clinical or treatment decisions; claims of clinical-grade accuracy; fairness claims across diabetes status (groups have 14-15 people); participant re-identification or data linkage.

## Repository map

```
cgmacros_bench/      library: data loading, native-reading detection, features, sampling/folds, models, statistics
scripts/             one script per result (table above)
docs/                data access guide and benchmark documentation (datasheet)
legacy_multimodal/   superseded original dissertation code, kept for the record
.github/             CI (syntax and import check) and issue templates
```

## Maintenance, licence and citation

Code: MIT licence ([`LICENSE`](LICENSE)). Maintained by Pulak Saha (Independent Researcher, pulak2164@gmail.com). Please report problems or reproduction differences through the [issue tracker](https://github.com/Pulak-PSR/cgmacros-leakfree-glucose-benchmark/issues). Cite the software with [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository" button) and cite the CGMacros dataset paper (Das et al., *Scientific Data*, 2025, doi:10.1038/s41597-025-05851-7).
