"""Feature and target construction.

cgm_mode controls which CGM series the CGM features are computed from:
  "native"        leak-free (the benchmark): only native sensor readings at or before the origin, held constant until
                  the next reading arrives ("last native reading", zero-order hold).
  "interp_same"   the SAME feature definitions computed from the released interpolated 1-minute series
                  (leaky: interpolated minutes between two readings contain the next reading).
  "interp_minute" a typical naive pipeline on the released series: minute-level lags (1, 2, 3, 5, 10, ... minutes),
                  1-minute differences and short rolling windows (leaky).
The target is the same in every mode: change from the last native reading at the origin to the glucose at t+h.
"""
import os

import numpy as np
import pandas as pd

from .config import HORIZON_MIN, LAGS, LAGS_MINUTE, NATIVE, POST_WINDOW, PRE_WINDOW, ROLLS, ROLLS_MINUTE
from .data import detect_native_phase

MEAL_COLS = ["m_cal", "m_carbs", "m_protein", "m_fat", "m_fiber", "m_sugar", "m_amount"]


def _cgm_features(g, ser, prefix, lags, rolls, ridx, ph, with_staleness, with_diff15, with_roc1):
    """Add CGM features computed from the series `ser`; returns the list of the new column names."""
    cols = []

    def put(name, values):
        g[prefix + name] = values
        cols.append(prefix + name)

    put("now", ser)
    for L in lags:
        put(f"lag_{L}", ser.shift(L))
    if with_roc1:
        put("roc1", ser.diff(1))
        put("roc2", ser.diff(1).diff(1))
    if with_diff15:
        put("d15", ser - ser.shift(15))
        put("a15", g[prefix + "d15"] - g[prefix + "d15"].shift(15))
    put("auc_30", ser.rolling(30, min_periods=1).sum() - ser * ser.rolling(30, min_periods=1).count())
    put("auc_60", ser.rolling(60, min_periods=1).sum() - ser * ser.rolling(60, min_periods=1).count())
    put("max_60", ser.rolling(60, min_periods=1).max() - ser)
    put("min_60", ser.rolling(60, min_periods=1).min() - ser)
    if with_staleness:                      # minutes since the last native reading (same column in every mode)
        cols.append("min_since_reading")
    for w in rolls:
        put(f"rmean_{w}", ser.rolling(w, min_periods=1).mean())
    for w in rolls:
        put(f"rstd_{w}", ser.rolling(w, min_periods=1).std().fillna(0.0))
    return cols


def build_features(d, bio_cols, cgm_dir, cgm_mode="native", horizons=None):
    """Returns (frame with features/targets/flags, feature-group dict, list of unique meal-image paths)."""
    assert cgm_mode in ("native", "interp_same", "interp_minute")
    horizons = horizons or HORIZON_MIN
    num = {"Libre GL": "gl", "HR": "heart_rate", "Calories (Activity)": "cal_activity",
           "METs": "mets", "Steps": "steps", "Intensity": "intensity",
           "Calories": "m_cal", "Carbs": "m_carbs", "Protein": "m_protein",
           "Fat": "m_fat", "Fiber": "m_fiber", "Sugar": "m_sugar",
           "Amount Consumed": "m_amount"}
    d = d.rename(columns=num)
    for c in num.values():
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["pid", "Timestamp"]).reset_index(drop=True)

    # meal-image index into the list of unique image paths
    path = d["Image path"].astype(str).str.strip()
    has_img = d["Image path"].notna() & (path != "") & (path.str.lower() != "nan")
    d["img_full"] = np.where(has_img, [os.path.join(cgm_dir, p, q) for p, q in zip(d["pid"], path)], None)
    uniq = list(dict.fromkeys([p for p in d["img_full"].values if p]))
    p2i = {p: i for i, p in enumerate(uniq)}
    d["img_idx"] = d["img_full"].map(p2i).fillna(-1).astype(int)

    d["is_meal"] = ((d["Meal Type"].notna()) | has_img | (d["m_carbs"].fillna(0) > 0)).astype(int)
    for c in MEAL_COLS:
        d[c] = d[c].fillna(0.0)

    out = []
    leaky_cols = []
    native_cols = []
    for pid, g in d.groupby("pid", sort=False):
        g = g.copy()
        raw = g["gl"].values.astype(float)
        gl = g["gl"].interpolate("linear").ffill().bfill()          # released (interpolated) series, gap-filled
        ridx = np.arange(len(g))
        ph, _, _, nan_raw = detect_native_phase(raw)
        gz = gl.where((ridx % NATIVE) == ph).ffill().bfill()         # last native reading at or before t
        g["gl"] = gl
        g["gl_now"] = gz                                            # always the native value (labels, naive scores)
        g["min_since_reading"] = ((ridx - ph) % NATIVE).astype(float)   # age of the last native reading (0-14 min)
        native_cols = _cgm_features(g, gz, "gl_", LAGS, ROLLS, ridx, ph, True, True, False)
        if cgm_mode == "interp_same":
            leaky_cols = _cgm_features(g, gl, "li_", LAGS, ROLLS, ridx, ph, True, True, False)
        elif cgm_mode == "interp_minute":
            leaky_cols = _cgm_features(g, gl, "li_", LAGS_MINUTE, ROLLS_MINUTE, ridx, ph, False, False, True)
        # origins whose past 120 and future 360 minutes lie on the regular 1-minute grid with no missing CGM value
        ts_ = g["Timestamp"]
        csn = np.r_[0, np.cumsum(nan_raw)]
        lo_ = np.clip(ridx - PRE_WINDOW, 0, len(g))
        hi_ = np.clip(ridx + POST_WINDOW + 1, 0, len(g))
        no_missing = (csn[hi_] - csn[lo_]) == 0
        g["regular"] = ((ts_.shift(-POST_WINDOW) - ts_) == pd.Timedelta(minutes=POST_WINDOW)) & \
                       ((ts_ - ts_.shift(PRE_WINDOW)) == pd.Timedelta(minutes=PRE_WINDOW)) & no_missing
        # time features
        hr = ts_.dt.hour + ts_.dt.minute / 60.0
        dow = ts_.dt.dayofweek
        g["hour_sin"], g["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
        g["dow_sin"], g["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
        # meal features, carried forward in time (causal)
        isn = g["is_meal"] == 1
        last_t = ts_.where(isn).ffill()
        g["time_since_meal"] = ((ts_ - last_t).dt.total_seconds() / 60).fillna(1440).clip(0, 1440)
        for c in MEAL_COLS:
            g["last_" + c] = g[c].where(isn).ffill().fillna(0.0)
        g["carbs_2h"] = g["m_carbs"].rolling(120, min_periods=1).sum()
        g["cal_2h"] = g["m_cal"].rolling(120, min_periods=1).sum()
        g["last_img_idx"] = g["img_idx"].where(g["img_idx"] >= 0).ffill().fillna(-1).astype(int)
        # targets: change relative to the last native reading (known at t)
        for h, m in horizons.items():
            g["y_" + h] = gl.shift(-m) - gz
        out.append(g)
    d = pd.concat(out, ignore_index=True)
    d = d.dropna(subset=["y_" + h for h in horizons]).reset_index(drop=True)

    groups = {
        "cgm": leaky_cols if cgm_mode != "native" else native_cols,
        "time": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
        "meal": ["time_since_meal", "carbs_2h", "cal_2h"] + ["last_" + c for c in MEAL_COLS],
        "activity": ["heart_rate", "cal_activity", "mets", "steps", "intensity"],
        "clinical": bio_cols,
    }
    return d, groups, uniq
