"""Loading the CGMacros files from PhysioNet (we never redistribute them) and detecting native CGM readings."""
import os

import numpy as np
import pandas as pd

from .config import NATIVE


def find_cgm_dir(data_dir):
    """Return the folder that contains CGMacros-001 ... CGMacros-049 and bio.csv."""
    cands = [data_dir,
             os.path.join(data_dir, "CGMacros"),
             os.path.join(data_dir, "CGMacros_dateshifted365", "CGMacros")]
    for c in cands:
        if os.path.isfile(os.path.join(c, "bio.csv")):
            return c
    for root, dirs, files in os.walk(data_dir):
        if "bio.csv" in files and any(x.startswith("CGMacros-") for x in dirs):
            return root
    raise FileNotFoundError(
        f"bio.csv not found under {data_dir!r}. Download CGMacros 1.0.0 from PhysioNet "
        "(https://physionet.org/content/cgmacros/1.0.0/), unzip it, and pass --data-dir.")


def load_all(cgm_dir):
    """Concatenate all participant files and merge the clinical table. Returns (frame, clinical column names)."""
    bio = pd.read_csv(os.path.join(cgm_dir, "bio.csv"))
    bio.columns = bio.columns.str.strip()
    bio = bio.rename(columns={"subject": "subject_num", "A1c PDL (Lab)": "hba1c",
                              "Fasting GLU - PDL (Lab)": "fasting_glucose", "Insulin": "insulin"})
    bio["subject_num"] = bio["subject_num"].astype(int)
    bio["male"] = (bio["Gender"].astype(str).str.upper().str.startswith("M")).astype(float)
    bio_cols = [c for c in ["Age", "BMI", "hba1c", "fasting_glucose", "insulin",
                            "Triglycerides", "Cholesterol", "HDL", "male"] if c in bio.columns]
    frames = []
    pids = sorted(d for d in os.listdir(cgm_dir) if d.startswith("CGMacros-") and os.path.isdir(os.path.join(cgm_dir, d)))
    for pid in pids:
        df = pd.read_csv(os.path.join(cgm_dir, pid, pid + ".csv"))
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["pid"] = pid
        df["subject_num"] = int(pid.split("-")[-1])
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d = d.merge(bio[["subject_num"] + bio_cols], on="subject_num", how="left")
    return d, bio_cols


def detect_native_phase(raw, period=NATIVE, min_knots=5):
    """Find which rows of the released 1-minute Libre series are native sensor readings.

    The released series is a piecewise-linear interpolation of the sensor's 15-minute readings, so native readings are
    the 'knots' where the second difference is non-zero. Per run of non-missing values we take the most common
    (row index mod 15) among the knots as that run's phase; a run without enough knots inherits the previous phase.

    Returns (phase per row, knot row indices, run ids, boolean mask of missing raw values).
    """
    raw = np.asarray(raw, dtype=float)
    nan_raw = np.isnan(raw)
    ph = np.zeros(len(raw), dtype=int)
    d2 = np.abs(np.diff(raw, 2))
    knots = np.where(d2 > 1e-6)[0] + 1
    run_id = np.cumsum(np.r_[0, (np.diff(nan_raw.astype(int)) != 0).astype(int)])
    last_ph = 0
    for rid in np.unique(run_id):
        m_ = run_id == rid
        if nan_raw[m_][0]:
            ph[m_] = last_ph
            continue
        k_ = knots[m_[knots]] if len(knots) else knots
        if len(k_) >= min_knots:
            last_ph = int(np.bincount(k_ % period, minlength=period).argmax())
        ph[m_] = last_ph
    return ph, knots, run_id, nan_raw
