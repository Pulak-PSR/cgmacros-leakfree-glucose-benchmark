"""Constants shared by all scripts. These are the values used for every result in the paper."""
import os

SEED = 42                                   # global seed (folds, origin offsets, models, bootstraps)
HORIZON_MIN = {"h1": 60, "h3": 180, "h6": 360}   # forecast horizons in minutes
LEAK_HORIZON_MIN = {"m15": 15, "m30": 30, "h1": 60, "h3": 180, "h6": 360}   # horizons of the leak experiment
NATIVE = 15                                 # native Libre sampling interval (minutes)
LAGS = [15, 30, 45, 60, 90, 120]            # lags (minutes) of the native-reading series
ROLLS = [30, 60, 120]                       # rolling-window lengths (minutes)
LAGS_MINUTE = [1, 2, 3, 5, 10, 15, 30, 60, 90, 120]   # minute-level lags used by the naive "interp_minute" variant
ROLLS_MINUTE = [5, 10, 30]                  # minute-level rolling windows of the naive variant
STRIDE = 5                                  # evaluation origins: every 5th minute (random per-participant offset)
N_FOLDS = 5
N_BOOT = 1000                               # participant bootstrap resamples (regression metrics)
N_BOOT_CLS = 300                            # participant bootstrap resamples (classification metrics)
N_PCA = 32                                  # PCA components of the MobileNetV2 meal-photo embedding
SPIKE_LEVEL = 140.0                         # mg/dL, "high glucose" endpoint
SPIKE_RISE = 30.0                           # mg/dL, "rise" endpoint
THREADS = int(os.environ.get("BENCH_THREADS", "8"))   # CPU threads for the boosted trees and torch
PRE_WINDOW = 120                            # minutes of history required before an origin
POST_WINDOW = 360                           # minutes of future required after an origin (largest horizon)
