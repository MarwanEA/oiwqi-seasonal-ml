"""Performance on the OIWQI scale and learning curves.

Learning curve: inside each outer LOOCV fold, m of the remaining wells are drawn (same draw for
every model: generator seeded with [seed, fold, m, repeat]), the model is fitted with fixed
hyperparameters and predicts the held-out well. Pooled LOOCV R2 per repeat, summarised over repeats.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from ..data import r2


def scale_metrics(y, pred) -> dict:
    """RMSE and MAE in index units, normalised by the target SD (sample) and range."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))
    sd, rng = float(np.std(y, ddof=1)), float(y.max() - y.min())
    return {"R2": r2(y, pred), "RMSE": rmse, "MAE": float(np.abs(y - pred).mean()), "target_SD": sd,
            "target_range": rng, "NRMSE_SD": rmse / sd, "NRMSE_range": rmse / rng}


def learning_curve(est: Any, X: pd.DataFrame, y: pd.Series, sizes: list[int], repeats: int, seed: int) -> pd.DataFrame:
    """Pooled LOOCV R2 for each training size m and repeat."""
    n = len(y)
    preds = {(m, r): np.empty(n) for m in sizes for r in range(repeats)}
    for i in range(n):
        pool = np.r_[0:i, i + 1:n]
        for m in sizes:
            for r in range(repeats):
                sub = np.random.default_rng([seed, i, m, r]).choice(pool, size=min(m, len(pool)), replace=False)
                preds[(m, r)][i] = clone(est).fit(X.iloc[sub], y.iloc[sub]).predict(X.iloc[[i]])[0]
    return pd.DataFrame([{"m_train": m, "repeat": r, "R2": r2(y, p)} for (m, r), p in preds.items()])


def summarise_curve(lc: pd.DataFrame) -> pd.DataFrame:
    g = lc.groupby("m_train").R2
    return pd.DataFrame({"R2_mean": g.mean(), "R2_p2.5": g.quantile(0.025), "R2_p97.5": g.quantile(0.975)}).reset_index()
