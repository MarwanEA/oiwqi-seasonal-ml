"""Model registry: 15 regression models, their hyperparameter grids and builders.

Tier 1-3 models are tuned with GridSearchCV over GRIDS. Stacking and Voting combine base
learners fixed a priori (each tuned beforehand); Stacking uses a Ridge meta-learner with its
default inner 5-fold. Optuna-XGB searches optuna_space() with a TPE sampler. GPR uses an
RBF + White kernel whose hyperparameters are re-optimised at every fit. Every estimator is
built single-threaded; parallelism is applied across outer folds instead.
"""

from __future__ import annotations

from typing import Any

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

GRIDS: dict[str, dict[str, list]] = {
    "XGBoost": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1],
                "reg_alpha": [0, 0.1, 1.0], "reg_lambda": [0.1, 1.0, 10.0], "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]},
    "RandomForest": {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 15],
                     "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
    "ExtraTrees": {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 15], "min_samples_split": [2, 5, 10]},
    "GradientBoosting": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1],
                         "subsample": [0.8, 1.0]},
    "LightGBM": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7, -1], "learning_rate": [0.01, 0.05, 0.1],
                 "num_leaves": [15, 31, 63], "reg_alpha": [0, 0.1, 1.0], "reg_lambda": [0.1, 1.0, 10.0]},
    "CatBoost": {"iterations": [100, 200, 300], "depth": [4, 6, 8], "learning_rate": [0.01, 0.05, 0.1],
                 "l2_leaf_reg": [1, 3, 9]},
    "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "ElasticNet": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "SVR": {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.5], "kernel": ["rbf", "poly"]},
    "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"], "p": [1, 2]},
}
# Optional LightGBM grid for very small samples (the library defaults need 20 rows per leaf)
LIGHTGBM_SMALL_SAMPLE_GRID: dict[str, list] = {
    "min_child_samples": [2, 3, 5], "min_data_in_bin": [1, 3], "min_split_gain": [0], "num_leaves": [4, 8, 15],
    "learning_rate": [0.05, 0.1], "n_estimators": [100, 300],
}

TUNED = list(GRIDS)
ENSEMBLES = ["StackingRegressor", "VotingRegressor"]
OPTUNA = "Optuna_XGBoost"
GPR = "GaussianProcessRegressor"
ALL_MODELS = TUNED + ENSEMBLES + [OPTUNA, GPR]
BASE_LEARNERS = ["Lasso", "ExtraTrees", "SVR"]
LINEAR = ["Lasso", "Ridge", "ElasticNet"]
TREES = ["ExtraTrees", "RandomForest", "GradientBoosting", "XGBoost", "CatBoost", "LightGBM", OPTUNA]
SCALE_SENSITIVE = ["Ridge", "Lasso", "ElasticNet", "SVR", "KNN", GPR]


def estimator(name: str, seed: int = 42) -> Any:
    """Unfitted, single-threaded estimator with the registry defaults."""
    builders = {
        "XGBoost": lambda: XGBRegressor(random_state=seed, verbosity=0, n_jobs=1),
        "RandomForest": lambda: RandomForestRegressor(random_state=seed, n_jobs=1),
        "ExtraTrees": lambda: ExtraTreesRegressor(random_state=seed, n_jobs=1),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=seed),
        "LightGBM": lambda: LGBMRegressor(random_state=seed, verbosity=-1, n_jobs=1),
        "CatBoost": lambda: CatBoostRegressor(random_seed=seed, verbose=0, thread_count=1, allow_writing_files=False),
        "Ridge": lambda: Ridge(random_state=seed),
        "Lasso": lambda: Lasso(random_state=seed, max_iter=10000),
        "ElasticNet": lambda: ElasticNet(random_state=seed, max_iter=10000),
        "SVR": lambda: SVR(),
        "KNN": lambda: KNeighborsRegressor(n_jobs=1),
        GPR: lambda: GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
                                              n_restarts_optimizer=5, normalize_y=True, random_state=seed),
    }
    return builders[name]()


def grid(name: str, cfg: dict | None = None) -> dict[str, list]:
    """Grid for a tuned model; the config (if given) takes precedence."""
    if cfg is not None and name in cfg.get("grids", {}):
        return cfg["grids"][name]
    return GRIDS[name]


def ensemble(kind: str, bases: list[tuple[str, Any]]) -> Any:
    """Stacking (Ridge meta-learner, default inner 5-fold) or equal-weight Voting over the given bases."""
    if kind == "StackingRegressor":
        return StackingRegressor(estimators=bases, final_estimator=Ridge(), n_jobs=1)
    if kind == "VotingRegressor":
        return VotingRegressor(estimators=bases, n_jobs=1)
    raise ValueError(kind)


def optuna_space(trial) -> dict:
    """Optuna-XGB search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
    }


def xgb(params: dict, seed: int = 42) -> XGBRegressor:
    """XGBoost with Optuna-selected parameters."""
    return XGBRegressor(**params, random_state=seed, verbosity=0, n_jobs=1)


def n_combinations(name: str, cfg: dict | None = None) -> int:
    n = 1
    for v in grid(name, cfg).values():
        n *= len(v)
    return n
