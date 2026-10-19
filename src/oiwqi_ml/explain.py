"""SHAP explanations.

kinds:
  "corr"   LinearExplainer, correlation-dependent. Its transform is estimated by Monte Carlo sampling,
           so the seed and nsamples are always explicit.
  "interv" LinearExplainer, interventional (exact for linear models).
  "tree"   TreeSHAP, interventional with the given background; the additivity error is computed and
           checked (scikit-learn and SHAP can disagree when background values lie exactly on split
           thresholds, so the error is reported instead of being hidden).
  "kernel" KernelSHAP (model-agnostic), l1_reg=False, explicit seed and nsamples.

Grouped SHAP sums the SHAP values of a group per well before taking mean |.| over wells.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from . import models as M
from .validation import run_parallel, with_params

ADDITIVITY_TOL = 1e-3
# Interventional TreeSHAP with a background set fails for XGBoost 3.3 under shap 0.52 ("categorical split"
# error without categorical features), so XGBoost models fall back to KernelSHAP, as LightGBM does.
TREE_EXPLAINED = [m for m in M.TREES if m not in ("LightGBM", "XGBoost", M.OPTUNA)]


def explainer_kind(model_name: str) -> str:
    """Default explainer for a model: linear -> corr, supported trees -> tree, otherwise kernel."""
    if model_name in M.LINEAR:
        return "corr"
    if model_name in TREE_EXPLAINED:
        return "tree"
    return "kernel"


def additivity_error(model: Any, base_value: float, X: pd.DataFrame, sv: np.ndarray) -> float:
    return float(np.max(np.abs(base_value + sv.sum(axis=1) - model.predict(X))))


def explain(model: Any, X_bg: pd.DataFrame, X: pd.DataFrame, kind: str, seed: int = 42, nsamples: int | None = None,
            check: bool = True) -> tuple[np.ndarray, float]:
    """SHAP values (wells x features) and the additivity error |base + sum(phi) - prediction| (max over wells)."""
    if kind == "tree":
        e = shap.TreeExplainer(model, X_bg)
        sv = np.asarray(e.shap_values(X, check_additivity=False))
        base = float(np.ravel(e.expected_value)[0])
    elif kind in ("corr", "interv"):
        np.random.seed(seed)  # the correlation-dependent transform samples from numpy's global generator
        fp = "correlation_dependent" if kind == "corr" else "interventional"
        kw = {"nsamples": nsamples} if (kind == "corr" and nsamples) else {}
        e = shap.LinearExplainer(model, X_bg, feature_perturbation=fp, **kw)
        sv = np.asarray(e.shap_values(X))
        base = float(np.ravel(e.expected_value)[0])
    elif kind == "kernel":
        np.random.seed(seed)
        e = shap.KernelExplainer(model.predict, X_bg)
        sv = np.asarray(e.shap_values(X, nsamples=nsamples or "auto", l1_reg=False, silent=True))
        base = float(np.ravel(e.expected_value)[0])
    else:
        raise ValueError(kind)
    err = additivity_error(model, base, X, sv)
    if check and kind in ("tree", "interv") and err > ADDITIVITY_TOL:
        raise ValueError(f"SHAP additivity error {err:.3g} exceeds {ADDITIVITY_TOL}")
    return sv, err


def full_data_shap(name: str, params: dict, X: pd.DataFrame, y: pd.Series, cfg: dict, kind: str | None = None,
                   seed: int | None = None, nsamples: int | None = None) -> tuple[pd.DataFrame, float]:
    """Refit the model with fixed hyperparameters on all wells and explain those wells (background = all wells)."""
    kind = kind or explainer_kind(name)
    model = with_params(name, params, cfg).fit(X, y)
    ns = nsamples or (cfg["shap"]["corr_nsamples"] if kind == "corr" else cfg["shap"]["kernel_nsamples"])
    sv, err = explain(model, X, X, kind, seed if seed is not None else cfg["seed"], ns)
    return pd.DataFrame(sv, columns=X.columns, index=X.index), err


def mean_abs(sv: pd.DataFrame) -> pd.Series:
    return sv.abs().mean()


def ranks(values: pd.Series) -> pd.Series:
    """Rank 1 = largest."""
    return values.rank(ascending=False, method="min").astype(int)


def grouped(sv: pd.DataFrame, groups: dict[str, list[str]]) -> pd.Series:
    """Grouped mean |SHAP|: sum within each group per well, then mean of the absolute value."""
    return pd.Series({g: sv[f].sum(axis=1).abs().mean() for g, f in groups.items()})


def _replicates(name, params, X, y, cfg, kind, idx, seeds, nsamples, groups):
    feat, grp, err = [], [], []
    for ix, sd in zip(idx, seeds):
        model = with_params(name, params, cfg).fit(X.iloc[ix], y.iloc[ix])
        sv, e = explain(model, X, X, kind, int(sd), nsamples, check=False)
        df = pd.DataFrame(sv, columns=X.columns)
        feat.append(mean_abs(df).to_numpy())
        grp.append(grouped(df, groups).to_numpy())
        err.append(e)
    return np.array(feat), np.array(grp), np.array(err)


def bootstrap_shap(name: str, params: dict, X: pd.DataFrame, y: pd.Series, cfg: dict, B: int, seed: int,
                   kind: str | None = None, per_replicate_seed: bool = True, nsamples: int | None = None,
                   workers: int = 1, chunk: int = 25) -> dict[str, Any]:
    """Bootstrap over wells: refit on each resample, explain the original wells.

    With per_replicate_seed the Monte Carlo seed of the explainer changes with the replicate
    (seed + 1000 + b), so the intervals include the explainer's sampling noise.
    """
    kind = kind or explainer_kind(name)
    groups = cfg["groups"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(B, len(y)))
    seeds = seed + 1000 + np.arange(B) if per_replicate_seed else np.full(B, seed)
    ns = nsamples or (cfg["shap"]["corr_nsamples"] if kind == "corr" else cfg["shap"]["kernel_nsamples"])
    parts = run_parallel(_replicates, [(name, params, X, y, cfg, kind, idx[a:a + chunk], seeds[a:a + chunk], ns, groups)
                                       for a in range(0, B, chunk)], workers)
    return {"features": list(X.columns), "groups": list(groups), "resamples": idx,
            "feature_mean_abs": np.vstack([p[0] for p in parts]), "group_mean_abs": np.vstack([p[1] for p in parts]),
            "additivity_error": np.concatenate([p[2] for p in parts]), "kind": kind}


def rank_rows(a: np.ndarray) -> np.ndarray:
    """Rank (1 = largest) within each row."""
    return (-a).argsort(axis=1).argsort(axis=1) + 1


def pairwise_probability(boot: dict, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """P(feature A has larger mean |SHAP| than feature B) over bootstrap replicates."""
    f, a = boot["features"], boot["feature_mean_abs"]
    rows = []
    for x, z in pairs:
        p = float((a[:, f.index(x)] > a[:, f.index(z)]).mean())
        rows.append({"A": x, "B": z, "P_A_above_B": p, "exceeds_sampling_variability": p >= 0.95 or p <= 0.05})
    return pd.DataFrame(rows)


def intervals(boot: dict, level: float = 0.95) -> pd.DataFrame:
    """Percentile intervals for mean |SHAP| and rank, per feature and per group."""
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    rows = []
    for key, names in (("feature_mean_abs", boot["features"]), ("group_mean_abs", boot["groups"])):
        a = boot[key]
        r = rank_rows(a)
        for j, nm in enumerate(names):
            rows.append({"level": key.split("_")[0], "name": nm, "ci_lo": np.percentile(a[:, j], lo),
                         "ci_hi": np.percentile(a[:, j], hi), "rank_lo": np.percentile(r[:, j], lo),
                         "rank_hi": np.percentile(r[:, j], hi), "P_rank1": float((r[:, j] == 1).mean())})
    return pd.DataFrame(rows)


def permutation(model: Any, X: pd.DataFrame, y: pd.Series, repeats: int, seed: int) -> pd.Series:
    """In-sample permutation importance (mean drop in R2)."""
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(model, X, y, n_repeats=repeats, random_state=seed, scoring="r2", n_jobs=1)
    return pd.Series(pi.importances_mean, index=X.columns)
