"""Rank range across seasons, direction of effect, normalised SHAP and between-season agreement."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..explain import rank_rows


def rank_range(importance: pd.DataFrame) -> pd.Series:
    """importance: features x seasons. Range of the rank (1 = largest) across seasons."""
    r = importance.rank(ascending=False, method="min")
    return (r.max(axis=1) - r.min(axis=1)).astype(int)


def rank_range_bootstrap(replicates: dict[str, np.ndarray], level: float = 0.95) -> pd.DataFrame:
    """replicates: season -> (B x features) mean |SHAP|; replicate b of each season is combined."""
    stack = np.stack([rank_rows(a) for a in replicates.values()])
    rr = stack.max(0) - stack.min(0)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return pd.DataFrame({"rank_range_lo": np.percentile(rr, lo, axis=0), "rank_range_hi": np.percentile(rr, hi, axis=0),
                         "rank_range_median": np.median(rr, axis=0)})


def direction_of_effect(X: pd.DataFrame, sv: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between each feature's values and its SHAP values (n/a when SHAP is constant)."""
    rows = []
    for f in sv.columns:
        if np.ptp(sv[f].to_numpy()) < 1e-12:
            rows.append({"feature": f, "spearman": np.nan, "p": np.nan, "direction": "n/a (constant SHAP)"})
            continue
        rho, p = spearmanr(X[f], sv[f])
        rows.append({"feature": f, "spearman": rho, "p": p, "direction": "positive" if rho > 0 else "negative"})
    return pd.DataFrame(rows)


def normalised(sv: pd.DataFrame, y, pred) -> pd.DataFrame:
    """mean |SHAP|, its share of the total (%), and per SD of the target and of the model's predictions."""
    imp = sv.abs().mean()
    sd_y, sd_p = float(np.std(np.asarray(y, float), ddof=1)), float(np.std(np.asarray(pred, float), ddof=1))
    return pd.DataFrame({"mean_abs_shap": imp, "share_pct": 100 * imp / imp.sum(), "per_SD_target": imp / sd_y,
                         "per_SD_prediction": imp / sd_p})


def between_season_agreement(importance: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of the importance ranking between every pair of seasons."""
    rows = []
    for a, b in itertools.combinations(importance.columns, 2):
        rho, p = spearmanr(importance[a], importance[b])
        rows.append({"season_a": a, "season_b": b, "spearman": rho, "p": p})
    return pd.DataFrame(rows)
