"""The Overall Irrigation Water Quality Index (OIWQI), as implemented.

Structure: six hazard classes with scores sx_k (weights W_k = sx_k / 21). Each parameter p gets a
term q_p(C) = r_p(C) * Cmax_p / (Cmin_p + C), or 0 when C > Cmax_p, where r_p is a step rating
(0-3). The class sub-index is S_k = sx_k / N_k * sum_{p in k} q_p, and OIWQI = sum_k W_k S_k.

Because q_p depends only on parameter p and is reused in every class that contains p, the index
is exactly additive over parameters: OIWQI = sum_p w_p q_p(x_p), with w_p = sum_{k ni p} W_k sx_k / N_k.
Its exact interventional Shapley values for a set of wells are therefore T_p - mean(T_p).

The thresholds, limits and class memberships below are the implementation used to compute the
targets; they are documented here as implemented.
"""

from __future__ import annotations

import pandas as pd

HAZARD_CLASSES: dict[str, dict] = {
    "salinity": {"sx": 6, "N": 2, "params": ["EC", "TDS"]},
    "sodicity": {"sx": 5, "N": 3, "params": ["SAR", "Na%", "SSP"]},
    "infiltration": {"sx": 4, "N": 4, "params": ["SSP", "Na%", "SAR", "PI"]},
    "toxicity": {"sx": 3, "N": 3, "params": ["K", "Na", "Cl"]},
    "soil_structure": {"sx": 2, "N": 3, "params": ["Na%", "SSP", "SAR"]},
    "miscellaneous": {"sx": 1, "N": 9, "params": ["pH", "Ca", "Mg", "NO3", "SO4", "HCO3", "RSBC", "MAR", "PI"]},
}
WEIGHTS: dict[str, float] = {k: v["sx"] / 21 for k, v in HAZARD_CLASSES.items()}
# (Cmax, Cmin) per parameter
LIMITS: dict[str, tuple[float, float]] = {
    "EC": (3000, 700), "TDS": (2000, 450), "Na%": (80, 20), "SSP": (80, 20), "SAR": (30, 10), "PI": (75, 25),
    "K": (35, 2), "Na": (400, 50), "Cl": (300, 30), "pH": (8.5, 7.0), "Ca": (4000, 50), "Mg": (600, 10),
    "NO3": (300, 2), "SO4": (2000, 10), "HCO3": (6000, 50), "RSBC": (15, 1), "MAR": (50, 10),
}
PARAMETERS = list(LIMITS)


def rating(p: str, v: float) -> int:
    """Step rating r_p(v) in {0, 1, 2, 3}, as implemented."""
    if p == "EC":
        return 3 if v < 700 else 2 if v <= 1500 else 1 if v <= 3000 else 0
    if p == "TDS":
        return 3 if v < 450 else 2 if v <= 900 else 1 if v <= 2000 else 0
    if p in ("Na%", "SSP"):
        return 3 if v < 20 else 2 if v <= 40 else 1 if v <= 80 else 0
    if p == "SAR":
        return 3 if v < 10 else 2 if v <= 18 else 1 if v <= 30 else 0
    if p == "PI":
        return 3 if v > 90 else 2 if v >= 75 else 1
    if p == "K":
        return 3 if v < 2 else 2 if v <= 5 else 1 if v <= 35 else 0
    if p == "Na":
        return 3 if v < 50 else 2 if v <= 150 else 1 if v <= 400 else 0
    if p == "Cl":
        return 3 if v < 30 else 2 if v <= 150 else 1 if v <= 300 else 0
    if p == "pH":
        if 7 <= v <= 7.5:
            return 3
        if 7.5 < v <= 8:
            return 2
        return 1 if (6.5 <= v < 7 or 8 < v <= 8.5) else 0
    if p == "Ca":
        if 50 <= v <= 75:
            return 3
        if 75 < v <= 150:
            return 2
        return 1 if (150 < v <= 4000 or v < 50) else 0
    if p == "Mg":
        if 10 <= v <= 20:
            return 3
        if 20 < v <= 30:
            return 2
        return 1 if (30 < v <= 600 or v < 10) else 0
    if p == "NO3":
        return 3 if v < 2 else 2 if v <= 5 else 1 if v <= 30 else 0
    if p == "SO4":
        return 3 if v < 10 else 2 if v <= 50 else 1 if v <= 200 else 0
    if p == "HCO3":
        return 3 if v < 50 else 2 if v <= 150 else 1 if v <= 600 else 0
    if p == "RSBC":
        return 3 if v < 1.25 else 2 if v <= 2.5 else 1
    if p == "MAR":
        return 3 if v < 10 else 2 if v <= 30 else 1 if v <= 50 else 0
    raise KeyError(p)


def q(p: str, v: float) -> float:
    """Parameter term before weighting: 0 above Cmax, else r_p(v) * Cmax / (Cmin + v)."""
    c_max, c_min = LIMITS[p]
    return 0.0 if v > c_max else c_max / (c_min + v) * rating(p, v)


def effective_weights() -> pd.Series:
    """w_p = sum over the classes containing p of W_k * sx_k / N_k."""
    w = dict.fromkeys(PARAMETERS, 0.0)
    for k, c in HAZARD_CLASSES.items():
        for p in c["params"]:
            w[p] += WEIGHTS[k] * c["sx"] / c["N"]
    return pd.Series(w)


def class_share() -> pd.DataFrame:
    """share[p, k] = (W_k sx_k / N_k) / w_p: the rule for splitting a parameter's attribution across classes."""
    w = effective_weights()
    s = pd.DataFrame(0.0, index=PARAMETERS, columns=list(HAZARD_CLASSES))
    for k, c in HAZARD_CLASSES.items():
        for p in c["params"]:
            s.loc[p, k] += WEIGHTS[k] * c["sx"] / c["N"] / w[p]
    return s


def parameter_terms(X: pd.DataFrame) -> pd.DataFrame:
    """T_p = w_p * q_p(x_p) per well; their row sum is the OIWQI."""
    w = effective_weights()
    return pd.DataFrame({p: [w[p] * q(p, v) for v in X[p]] for p in PARAMETERS}, index=X.index)


def class_terms(X: pd.DataFrame) -> pd.DataFrame:
    """W_k * S_k per well, computed class by class as in the original implementation."""
    rows = []
    for _, r in X.iterrows():
        rows.append({k: WEIGHTS[k] * c["sx"] / c["N"] * sum(q(p, r[p]) for p in c["params"])
                     for k, c in HAZARD_CLASSES.items()})
    return pd.DataFrame(rows, index=X.index)


def oiwqi(X: pd.DataFrame) -> pd.Series:
    """The index per well."""
    return class_terms(X).sum(axis=1).rename("O_IWQI")


def index_shap(X: pd.DataFrame) -> pd.DataFrame:
    """Exact interventional Shapley values of the additive index, with these wells as background."""
    T = parameter_terms(X)
    return T - T.mean()


def category(v: float) -> str:
    """Suitability class of an OIWQI value, as implemented."""
    for lim, name in ((3.5, "Rejection"), (6, "Poor"), (12, "Fair"), (23, "Moderate"), (35, "Good")):
        if v < lim:
            return name
    return "Excellent"
