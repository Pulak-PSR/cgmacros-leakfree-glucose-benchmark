"""Forecast origins, participant-grouped folds and the train/validation/test split inside each fold."""
import numpy as np
import pandas as pd

from .config import N_FOLDS, STRIDE


def participant_ids(d, smoke=False):
    pids = np.array(sorted(d["pid"].unique()))
    if smoke:
        pids = pids[:12]
    return pids


def select_origins(d, pids, seed):
    """Evaluation/training origins: every STRIDE-th minute (random per-participant offset) plus every meal onset,
    restricted to origins with a regular 1-minute grid and no missing CGM value in [t-120, t+360].
    Returns (origin frame D, dict with row positions and flags)."""
    rs = np.random.RandomState(seed)
    off = {p: rs.randint(0, STRIDE) for p in pids}
    rowpos = d.groupby("pid").cumcount().values
    stride_mask = np.array([(rp - off[p]) % STRIDE == 0 for rp, p in zip(rowpos, d["pid"].values)])
    meal_mask = d["is_meal"].values == 1
    use = (stride_mask | meal_mask) & d["regular"].values
    D = d[use].reset_index(drop=True)
    info = dict(orig_idx=np.where(use)[0], rowpos=rowpos,
                stride_flag=stride_mask[use] & ~meal_mask[use],     # "all time points" evaluation set
                meal_flag=meal_mask[use])                           # "meal onsets" evaluation set
    return D, info


def assign_folds(pids, seed):
    """Participant-grouped folds: shuffle participants with the seed, deal them round-robin into N_FOLDS folds."""
    perm = np.random.RandomState(seed).permutation(pids)
    return {p: i % N_FOLDS for i, p in enumerate(perm)}


def fold_table(pids, seed):
    fo = assign_folds(pids, seed)
    return pd.DataFrame({"pid": list(fo.keys()), "fold": list(fo.values())}).sort_values("pid").reset_index(drop=True)


def split_fold(pid_arr, fold_arr, f, seed):
    """Indices (train, validation, test) for fold f: the test fold's participants are test; 15% of the remaining
    participants (at least 2) form the validation set used for early stopping and model selection."""
    te = np.where(fold_arr == f)[0]
    trall = np.where(fold_arr != f)[0]
    tr_p = np.unique(pid_arr[trall])
    vr = np.random.RandomState(seed + f)
    val_p = set(vr.choice(tr_p, size=max(2, int(0.15 * len(tr_p))), replace=False))
    va = trall[np.isin(pid_arr[trall], list(val_p))]
    tr = trall[~np.isin(pid_arr[trall], list(val_p))]
    return tr, va, te
