# Benchmark documentation

Short datasheet for the leak-free CGMacros glucose-forecasting benchmark. The full version is Appendix G of the paper.

## Purpose and scope
Evaluate glucose-change forecasting from the first 120 minutes of history (CGM, time of day, meals, activity, clinical markers, meal-photo embeddings) under a leak-free, participant-held-out protocol, and audit CGM datasets for resampling that leaks the future.

**Intended uses:** comparing forecasting methods on identical folds; teaching and testing for look-ahead leakage; auditing other regularly resampled sensor files with `scripts/audit_csv.py`.
**Not intended for:** dietary, clinical or treatment decisions; claims of clinical-grade accuracy; fairness claims across diabetes status (groups have 14 to 15 people); re-identification or data linkage.

## Data
* Source: CGMacros v1.0.0 on PhysioNet, https://physionet.org/content/cgmacros/1.0.0/ (Das et al., *Scientific Data*, 2025). The dataset is open access under CC BY-NC-SA 4.0 (version DOI 10.13026/3z8q-x658): attribution, non-commercial use and share-alike are required. **This repository contains no CGMacros data and no row-level derivatives.**
* 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes); 44 analysed (participant 007 has no regular-grid stretch).

## Preprocessing (as implemented in `cgmacros_bench/`)
1. Detect the native-reading phase of the Libre series from non-zero second differences (`data.py`).
2. Build the leak-free series: the last native reading held until the next one arrives (`features.py`).
3. Target at horizon h: glucose at t + h minus the last native reading at t, h in {60, 180, 360} min (benchmark) and {15, 30, 60, 180, 360} min (leak experiment).
4. Forecast origins: every 5th minute with a random per-participant offset plus meal onsets, requiring a regular 1-minute grid and no missing Libre value in [t-120, t+360] min (`sampling.py`).
5. Folds: participants shuffled with seed 42 and dealt round-robin into 5 folds; 15% of training participants form the validation set. `run_benchmark.py` writes `results/fold_assignment.csv` (participant IDs per fold).

## Metrics and statistics
R2, MAE, skill versus zero-change; spike endpoints (glucose > 140 mg/dL; rise of at least 30 mg/dL) with AUROC, AUPRC, Brier skill and ECE; participant-level bootstrap (1000 resamples, 300 for classification); paired Wilcoxon with Holm correction.

## Hosting, licence, maintenance
Hosted on GitHub. Code: MIT licence. Summary-level tables in `expected_results/`: CC BY-NC-SA 4.0, matching the dataset. Maintained by the author (Pulak Saha); issues and reproduction reports via the issue tracker. Versioned releases; changes are listed in `CHANGELOG.md`.

## Known limitations
The 15-minute native interval is inferred from the data, not confirmed with the creators or manufacturer; a single fold assignment and seed; targets for origins between native readings are interpolated values; one dataset, 44 participants.
