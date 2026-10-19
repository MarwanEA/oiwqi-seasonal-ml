"""Shared figure style and plotting helpers. Every function returns the figure; save() writes it at 300 dpi."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .models import ALL_MODELS  # noqa: E402

_CMAP = matplotlib.colormaps["tab20"]
ORDER = ["Lasso", "Ridge", "ElasticNet", "ExtraTrees", "RandomForest", "GradientBoosting", "XGBoost", "CatBoost",
         "SVR", "KNN", "LightGBM", "StackingRegressor", "VotingRegressor", "Optuna_XGBoost", "GaussianProcessRegressor"]
assert sorted(ORDER) == sorted(ALL_MODELS)
COLOR = {m: _CMAP(i) for i, m in enumerate(ORDER)}
SHORT = {"StackingRegressor": "Stacking", "VotingRegressor": "Voting", "Optuna_XGBoost": "Optuna-XGB",
         "GaussianProcessRegressor": "GPR", "GradientBoosting": "GradBoost"}


def short(m: str) -> str:
    return SHORT.get(m, m)


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def forest(ci: pd.DataFrame, title: str, xmin: float = -1.0):
    """ci: output of analysis.uncertainty.confidence_intervals. Point R2 with interval, sorted by R2."""
    c = ci.sort_values("R2").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.5, 0.3 * len(c) + 1.2))
    for i, r in c.iterrows():
        col = COLOR.get(r.model, "grey")
        ax.plot([max(r.R2_lo, xmin), r.R2_hi], [i, i], color=col, lw=2.2)
        ax.plot(max(r.R2, xmin), i, "o", color=col, mec="k", ms=5)
        if r.R2_lo < xmin:
            ax.text(xmin + 0.02, i + 0.25, f"lower {r.R2_lo:.2f}", fontsize=6)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(c)), [short(m) for m in c.model], fontsize=8)
    ax.set(xlim=(xmin, 1), xlabel="LOOCV R² (95% bootstrap CI)", title=title)
    ax.title.set_fontsize(10)
    return fig


def interval_bars(mean: pd.Series, lo: pd.Series, hi: pd.Series, title: str, xlabel: str, color="tab:blue"):
    """Horizontal bars with intervals, largest at the top."""
    order = mean.sort_values().index
    m, lo, hi = mean[order], lo[order], hi[order]
    fig, ax = plt.subplots(figsize=(5.5, 0.28 * len(m) + 1.2))
    ax.barh(range(len(m)), m, xerr=[(m - lo).clip(lower=0), (hi - m).clip(lower=0)], color=color,
            edgecolor="k", linewidth=0.3, error_kw={"lw": 0.8})
    ax.set_yticks(range(len(m)), order, fontsize=8)
    ax.set(xlabel=xlabel, title=title)
    ax.title.set_fontsize(10)
    return fig


def heatmap(table: pd.DataFrame, title: str, fmt: str = "{:.2f}", cmap: str = "viridis"):
    """table: rows x columns of numbers, values printed in each cell."""
    fig, ax = plt.subplots(figsize=(1.2 * table.shape[1] + 2, 0.32 * table.shape[0] + 1.2))
    im = ax.imshow(table.to_numpy(float), aspect="auto", cmap=cmap)
    ax.set_xticks(range(table.shape[1]), table.columns, fontsize=8)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
    lim = np.nanmean(table.to_numpy(float))
    for (i, j), v in np.ndenumerate(table.to_numpy(float)):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6, color="w" if v < lim else "k")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    return fig


def learning_curves(summary: dict[str, pd.DataFrame], title: str):
    """summary: model -> output of analysis.learning_curves.summarise_curve."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m, s in summary.items():
        col = COLOR.get(m, "grey")
        ax.plot(s.m_train, s.R2_mean, "o-", color=col, label=short(m), ms=4)
        ax.fill_between(s.m_train, s["R2_p2.5"], s["R2_p97.5"], color=col, alpha=0.15)
    ax.set(xlabel="training wells per fold", ylabel="pooled LOOCV R²", title=title)
    ax.title.set_fontsize(10)
    ax.legend(fontsize=7)
    return fig
