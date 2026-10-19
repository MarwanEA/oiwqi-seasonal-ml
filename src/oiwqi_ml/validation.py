"""Validation protocols.

original_protocol: one GridSearchCV (unshuffled K-fold) on a fixed train/test split, then LOOCV
    with the selected hyperparameters fixed (the protocol as first run).
nested_loocv: outer LOOCV; inside every outer fold, the search is repeated on the remaining wells
    with a shuffled inner K-fold (seed from the config).
scale_in_fold: when True, scale-sensitive models are wrapped as Pipeline(StandardScaler, model), so the
    scaler is fitted inside every fold. The default (False) uses raw features.

Runtime helpers: one thread per worker (BLAS/OpenMP variables), parallel outer folds with joblib,
a memory guard on the number of workers, one checkpoint file per fold, and a battery check that is
only active on Windows laptops.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import models as M

THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
SCORING = "neg_root_mean_squared_error"


# ── runtime helpers ─────────────────────────────────────────────────────────
def set_worker_env() -> None:
    """One thread per worker; make the package importable in joblib worker processes."""
    os.environ.update(THREAD_ENV)
    src = str(Path(__file__).resolve().parents[1])
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if src not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([src, *paths])


def battery_ok() -> bool:
    """False only on a Windows machine running on battery; True elsewhere."""
    if platform.system() != "Windows":
        return True
    import psutil
    b = psutil.sensors_battery()
    return b is None or b.power_plugged


def safe_workers(requested: int, mb_per_worker: float = 300.0) -> int:
    """Limit the number of workers so that they fit in the available memory (20% margin)."""
    import psutil
    avail = psutil.virtual_memory().available / 2**20
    return max(1, min(requested, int(avail / (mb_per_worker * 1.2))))


def write_json(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    tmp.replace(path)


# ── building blocks ─────────────────────────────────────────────────────────
def maybe_scaled(name: str, est: Any, scale: bool) -> Any:
    """Wrap a scale-sensitive estimator in Pipeline(StandardScaler, model) when scale is True."""
    if scale and name in M.SCALE_SENSITIVE:
        return Pipeline([("scaler", StandardScaler()), ("model", est)])
    return est


def prefixed(grid: dict, est: Any) -> dict:
    return {f"model__{k}": v for k, v in grid.items()} if isinstance(est, Pipeline) else grid


def unprefixed(params: dict) -> dict:
    return {k.removeprefix("model__"): v for k, v in params.items()}


def loocv_predict(est: Any, X: pd.DataFrame, y: pd.Series, scale: bool = False) -> np.ndarray:
    """Leave-one-out predictions; a fresh clone is fitted on the other n - 1 wells for every well.

    scale=True wraps the clone in Pipeline(StandardScaler, estimator), so scaling is fitted in the fold.
    """
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        tr = np.r_[0:i, i + 1:n]
        m = Pipeline([("scaler", StandardScaler()), ("model", clone(est))]) if scale else clone(est)
        pred[i] = m.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[[i]])[0]
    return pred


def tune(name: str, X: pd.DataFrame, y: pd.Series, cv: Any, cfg: dict, scale: bool = False,
         grid: dict | None = None) -> tuple[Any, dict]:
    """GridSearchCV for one tuned model; returns the refitted best estimator and its (unprefixed) parameters."""
    est = maybe_scaled(name, M.estimator(name, cfg["seed"]), scale)
    g = prefixed(grid or M.grid(name, cfg), est)
    gs = GridSearchCV(est, g, cv=cv, scoring=SCORING, n_jobs=1, refit=True).fit(X, y)
    return gs.best_estimator_, unprefixed(gs.best_params_)


def with_params(name: str, params: dict, cfg: dict, scale: bool = False) -> Any:
    """Unfitted estimator with fixed hyperparameters."""
    if name == M.OPTUNA:
        return M.xgb(params, cfg["seed"])
    est = M.estimator(name, cfg["seed"])
    if name in M.GRIDS:
        est.set_params(**params)
    return maybe_scaled(name, est, scale)


def optuna_search(X: pd.DataFrame, y: pd.Series, cfg: dict, n_trials: int | None = None) -> dict:
    """TPE search (seeded) minimising the LOOCV RMSE of XGBoost on the given wells."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials = n_trials or cfg["models"]["optuna_trials"]

    def objective(trial):
        p = loocv_predict(M.xgb(M.optuna_space(trial), cfg["seed"]), X, y)
        return float(np.sqrt(np.mean((y.to_numpy() - p) ** 2)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params


def ensemble_bases(base_params: dict[str, dict], cfg: dict, scale: bool) -> list[tuple[str, Any]]:
    return [(b, with_params(b, base_params[b], cfg, scale)) for b in cfg["models"]["base_learners"]]


# ── original protocol ───────────────────────────────────────────────────────
def original_protocol(X: pd.DataFrame, y: pd.Series, cfg: dict, model_names: list[str] | None = None,
                      scale_in_fold: bool = False, n_trials: int | None = None) -> dict[str, dict]:
    """One search on a fixed split (unshuffled K-fold), then LOOCV with fixed hyperparameters.

    Returns {model: {"pred": LOOCV predictions, "params": selected hyperparameters}}.
    """
    v = cfg["validation"]
    names = model_names or M.ALL_MODELS
    tr, _ = train_test_split(np.arange(len(y)), test_size=v["test_size"], random_state=cfg["seed"])
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    out: dict[str, dict] = {}
    need = set(names) | (set(cfg["models"]["base_learners"]) if set(names) & set(M.ENSEMBLES) else set())
    for name in [m for m in M.TUNED if m in need]:
        best, params = tune(name, Xtr, ytr, v["original_cv_folds"], cfg, scale_in_fold)
        out[name] = {"params": params, "pred": loocv_predict(best, X, y)}
    for name in [m for m in M.ENSEMBLES if m in names]:
        ens = M.ensemble(name, ensemble_bases({b: out[b]["params"] for b in cfg["models"]["base_learners"]},
                                              cfg, scale_in_fold))
        out[name] = {"params": {b: out[b]["params"] for b in cfg["models"]["base_learners"]},
                     "pred": loocv_predict(ens, X, y)}
    if M.OPTUNA in names:
        params = optuna_search(X, y, cfg, n_trials)  # selection on the same LOOCV that is reported
        out[M.OPTUNA] = {"params": params, "pred": loocv_predict(M.xgb(params, cfg["seed"]), X, y)}
    if M.GPR in names:
        out[M.GPR] = {"params": {"kernel": "RBF(1) + White(1), re-optimised per fit"},
                      "pred": loocv_predict(maybe_scaled(M.GPR, M.estimator(M.GPR, cfg["seed"]), scale_in_fold), X, y)}
    return {k: out[k] for k in names}


# ── nested LOOCV ────────────────────────────────────────────────────────────
def inner_cv(cfg: dict) -> KFold:
    v = cfg["validation"]
    return KFold(v["inner_cv_folds"], shuffle=v["inner_shuffle"], random_state=cfg["seed"] if v["inner_shuffle"] else None)


def nested_fold(name: str, X: pd.DataFrame, y: pd.Series, i: int, cfg: dict, scale: bool = False,
                base_params: dict[str, dict] | None = None, n_trials: int | None = None) -> dict:
    """One outer fold: tune on all wells but i, refit, predict well i."""
    tr = np.r_[0:i, i + 1:len(y)]
    Xt, yt = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    t0 = time.perf_counter()
    if name in M.TUNED:
        fitted, params = tune(name, Xt, yt, inner_cv(cfg), cfg, scale)
    elif name in M.ENSEMBLES:
        if base_params is None:
            base_params = {b: tune(b, Xt, yt, inner_cv(cfg), cfg, scale)[1] for b in cfg["models"]["base_learners"]}
        fitted = M.ensemble(name, ensemble_bases(base_params, cfg, scale)).fit(Xt, yt)
        params = base_params
    elif name == M.OPTUNA:
        params = optuna_search(Xt, yt, cfg, n_trials)
        fitted = M.xgb(params, cfg["seed"]).fit(Xt, yt)
    elif name == M.GPR:
        fitted = maybe_scaled(name, M.estimator(name, cfg["seed"]), scale).fit(Xt, yt)
        k = fitted[-1].kernel_ if scale else fitted.kernel_
        params = {"kernel_fitted": repr(k)}
    else:
        raise ValueError(name)
    return {"model": name, "fold": i, "y_true": float(y.iloc[i]), "y_pred": float(fitted.predict(X.iloc[[i]])[0]),
            "params": params, "seconds": time.perf_counter() - t0}


def run_parallel(fn: Callable, tasks: list[tuple], workers: int) -> list:
    """Run fn(*task) over tasks, in parallel worker processes when workers > 1."""
    if workers <= 1:
        return [fn(*t) for t in tasks]
    from joblib import Parallel, delayed
    set_worker_env()
    return Parallel(n_jobs=workers)(delayed(fn)(*t) for t in tasks)


def _checkpointed(path: Path | None, name, X, y, i, cfg, scale, base_params, n_trials) -> dict:
    if path is not None and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rec = nested_fold(name, X, y, i, cfg, scale, base_params, n_trials)
    if path is not None:
        write_json(path, rec)
    return rec


def nested_loocv(X: pd.DataFrame, y: pd.Series, cfg: dict, model_names: list[str] | None = None,
                 scale_in_fold: bool = False, workers: int = 1, checkpoint_dir: str | Path | None = None,
                 stations: pd.Series | None = None, n_trials: int | None = None) -> pd.DataFrame:
    """Nested LOOCV for the requested models; one checkpoint file per (model, fold) when a directory is given.

    Stacking and Voting reuse, in each outer fold, the base learners tuned by their own nested search.
    """
    names = model_names or M.ALL_MODELS
    ens = [m for m in names if m in M.ENSEMBLES]
    bases = cfg["models"]["base_learners"] if ens else []
    stage1 = [m for m in dict.fromkeys([*names, *bases]) if m not in M.ENSEMBLES]
    ck = Path(checkpoint_dir) if checkpoint_dir else None

    def path(m, i):
        return ck / ("scaled" if scale_in_fold else "raw") / m / f"fold{i:02d}.json" if ck else None
    n = len(y)
    recs = run_parallel(_checkpointed, [(path(m, i), m, X, y, i, cfg, scale_in_fold, None, n_trials)
                                        for m in stage1 for i in range(n)], workers)
    by = {(r["model"], r["fold"]): r for r in recs}
    tasks = [(path(m, i), m, X, y, i, cfg, scale_in_fold, {b: by[(b, i)]["params"] for b in bases}, n_trials)
             for m in ens for i in range(n)]
    recs += run_parallel(_checkpointed, tasks, workers)
    df = pd.DataFrame([r for r in recs if r["model"] in names])
    if stations is not None:
        df.insert(0, "station", df.fold.map(dict(enumerate(stations))))
    df["inner_cv"] = repr(inner_cv(cfg))
    df["scale_in_fold"] = scale_in_fold
    return df.sort_values(["model", "fold"]).reset_index(drop=True)
