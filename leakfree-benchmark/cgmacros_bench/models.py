"""Model definitions (hyperparameters are those reported in the paper)."""
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .config import SEED, THREADS


def fit_ridge_predict(Xtr, ytr, Xte, alpha=10.0):
    """Ridge on standardized features; standardized values are clipped to +-5 SD to avoid extrapolation blow-ups
    on unseen participants."""
    sc = StandardScaler().fit(Xtr)
    zt = lambda A: np.clip(sc.transform(A), -5, 5)
    return Ridge(alpha=alpha).fit(zt(Xtr), ytr).predict(zt(Xte))


def fit_catboost(Xtr, ytr, Xva, yva, iters, depth=6, lr=0.05, l2=5):
    m = CatBoostRegressor(iterations=iters, learning_rate=lr, depth=depth, l2_leaf_reg=l2, random_seed=SEED,
                          verbose=0, thread_count=THREADS, early_stopping_rounds=60)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    return m


def fit_logistic(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    zc = lambda A: np.clip(sc.transform(A), -5, 5)
    lr = LogisticRegression(C=1.0, max_iter=300).fit(zc(Xtr), ytr)
    return lr.predict_proba(zc(Xte))[:, 1]


def fit_catboost_classifier(Xtr, ytr, Xva, yva, iters):
    cc = CatBoostClassifier(iterations=min(iters, 600), learning_rate=0.08, depth=6, l2_leaf_reg=5,
                            random_seed=SEED, verbose=0, thread_count=THREADS, early_stopping_rounds=50)
    cc.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    return cc
