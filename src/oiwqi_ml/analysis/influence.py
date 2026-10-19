"""Influence of a single well on LOOCV performance."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..data import drop_well, r2, zscores


def most_extreme_well(stations: pd.Series, y: pd.Series) -> str:
    """Station with the largest absolute z-score of the target."""
    return str(stations.iloc[int(np.argmax(np.abs(zscores(y).to_numpy())))])


def well_profile(stations: pd.Series, y: pd.Series, label: str) -> dict:
    """z-score (sample SD) and share of the total sum of squares of one well."""
    i = int(np.flatnonzero((stations == label).to_numpy())[0])
    ss = ((y - y.mean()) ** 2)
    return {"station": label, "value": float(y.iloc[i]), "z": float(zscores(y).iloc[i]),
            "ss_share": float(ss.iloc[i] / ss.sum())}


def r2_with_without(stations: pd.Series, y, pred, label: str) -> dict:
    """R2 on all wells, and on the other wells using the same LOOCV predictions."""
    keep = (stations != label).to_numpy()
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"R2_all": r2(y, pred), "R2_without_same_predictions": r2(y[keep], pred[keep])}


def r2_refit_without(stations: pd.Series, X: pd.DataFrame, y: pd.Series, label: str,
                     loocv_fn: Callable[[pd.DataFrame, pd.Series], np.ndarray]) -> float:
    """R2 after removing the well and re-running LOOCV (loocv_fn returns LOOCV predictions)."""
    st, Xw, yw = drop_well(stations, X, y, label)
    return r2(yw, loocv_fn(Xw, yw))


def residuals(stations: pd.Series, y, pred) -> pd.DataFrame:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return pd.DataFrame({"station": list(stations), "y_true": y, "y_pred": pred, "residual": y - pred})
