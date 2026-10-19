"""Analytical decomposition of the index and its agreement with model SHAP.

For additive terms T_j with sum OIWQI: covariance share cov(T_j, OIWQI) / var(OIWQI) (sums to 1),
variance share var(T_j) / var(OIWQI), and the exact index Shapley value mean |T_j - mean T_j|.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .. import index


def shares(T: pd.DataFrame) -> pd.DataFrame:
    o = T.sum(axis=1)
    v = o.var(ddof=1)
    return pd.DataFrame({"cov_share": [T[c].cov(o) / v for c in T], "var_share": [T[c].var(ddof=1) / v for c in T],
                         "index_shap": (T - T.mean()).abs().mean().to_numpy()}, index=T.columns)


def parameter_and_class_shares(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return shares(index.parameter_terms(X)), shares(index.class_terms(X))


def class_level_shap(sv: pd.DataFrame) -> pd.DataFrame:
    """Split each parameter's SHAP value across its hazard classes in proportion to its class weights."""
    s = index.class_share().loc[sv.columns]
    return pd.DataFrame(sv.to_numpy() @ s.to_numpy(), columns=s.columns, index=sv.index)


def agreement(a: pd.Series, b: pd.Series, k: int = 5) -> dict:
    """Rank agreement between two importance vectors (Spearman, Kendall, top-k overlap, exact zeros)."""
    b = b.loc[a.index]
    rs, ps = spearmanr(a, b)
    kt, pk = kendalltau(a, b)
    return {"spearman": rs, "spearman_p": ps, "kendall": kt, "kendall_p": pk,
            f"top{k}_overlap": len(set(a.nlargest(k).index) & set(b.nlargest(k).index)),
            "n_exact_zeros_a": int((a.abs() < 1e-12).sum()), "n_exact_zeros_b": int((b.abs() < 1e-12).sum())}
