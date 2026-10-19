"""Bootstrap confidence intervals and paired comparisons of LOOCV predictions.

All functions take the observed values y (n wells) and a table of predictions (wells x models).
The same bootstrap resamples of wells are used for every model, so comparisons are paired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from ..data import r2


def bootstrap_indices(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def boot_metrics(y: np.ndarray, P: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R2 and MSE in every resample (B x models); R2 uses each resample's own mean."""
    yb = y[idx]
    Pb = P[:, idx].transpose(1, 0, 2)
    sse = ((yb[:, None, :] - Pb) ** 2).sum(-1)
    sst = ((yb - yb.mean(1, keepdims=True)) ** 2).sum(-1)[:, None]
    return 1 - sse / sst, sse / y.size


def _arrays(y, preds: pd.DataFrame):
    return np.asarray(y, float), preds.to_numpy(float).T, list(preds.columns)


def confidence_intervals(y, preds: pd.DataFrame, B: int, seed: int, level: float = 0.95) -> pd.DataFrame:
    """Point R2 and RMSE with percentile intervals for every model."""
    yv, P, names = _arrays(y, preds)
    R2, MSE = boot_metrics(yv, P, bootstrap_indices(len(yv), B, seed))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    rows = []
    for k, m in enumerate(names):
        rows.append({"model": m, "R2": r2(yv, P[k]), "R2_lo": np.percentile(R2[:, k], lo), "R2_hi": np.percentile(R2[:, k], hi),
                     "RMSE": float(np.sqrt(((yv - P[k]) ** 2).mean())), "RMSE_lo": np.percentile(np.sqrt(MSE[:, k]), lo),
                     "RMSE_hi": np.percentile(np.sqrt(MSE[:, k]), hi)})
    return pd.DataFrame(rows)


def p_best(y, preds: pd.DataFrame, B: int, seed: int) -> pd.Series:
    """Share of resamples in which each model has the highest R2."""
    yv, P, names = _arrays(y, preds)
    R2, _ = boot_metrics(yv, P, bootstrap_indices(len(yv), B, seed))
    return pd.Series(np.bincount(R2.argmax(1), minlength=len(names)) / B, index=names)


def paired_vs_leader(y, preds: pd.DataFrame, B: int, seed: int, level: float = 0.95) -> pd.DataFrame:
    """Each model against the model with the highest R2: paired bootstrap of the R2 and MSE differences,
    paired Wilcoxon test on absolute errors (Holm-corrected across models)."""
    yv, P, names = _arrays(y, preds)
    R2, MSE = boot_metrics(yv, P, bootstrap_indices(len(yv), B, seed))
    point = np.array([r2(yv, p) for p in P])
    lead = int(np.argmax(point))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    rows, pv = [], []
    for k, m in enumerate(names):
        if k == lead:
            continue
        d_r2, d_mse = R2[:, k] - R2[:, lead], MSE[:, k] - MSE[:, lead]
        ea, eb = np.abs(yv - P[k]), np.abs(yv - P[lead])
        p = wilcoxon(ea, eb).pvalue if np.any(ea != eb) else 1.0
        pv.append(p)
        rows.append({"leader": names[lead], "model": m, "dR2": point[k] - point[lead],
                     "dR2_lo": np.percentile(d_r2, lo), "dR2_hi": np.percentile(d_r2, hi),
                     "dMSE_lo": np.percentile(d_mse, lo), "dMSE_hi": np.percentile(d_mse, hi),
                     "share_resamples_model_beats_leader": float((d_r2 > 0).mean()), "wilcoxon_p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"] = multipletests(pv, method="holm")[1]
        out["dR2_interval_excludes_0"] = (out.dR2_hi < 0) | (out.dR2_lo > 0)
    return out
