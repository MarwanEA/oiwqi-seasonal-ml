"""Loading and validating the seasonal tables.

Each season is one table with a station column, the 17 predictors and the OIWQI target.
A raw electrical-conductivity column named CE is renamed to EC. Station labels are used
exactly as given; they are never renamed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Generic plausibility bounds for groundwater (mg/L unless noted); used for warnings only.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "NO3": (0, 200), "pH": (5.5, 9.0), "EC": (0, 5000), "TDS": (0, 3500), "HCO3": (0, 800), "Cl": (0, 400),
    "Mg": (0, 200), "Ca": (0, 300), "Na": (0, 300), "K": (0, 50), "SO4": (0, 300),
}


def read_table(path: str | Path, ec_raw: str = "CE") -> pd.DataFrame:
    """Read one seasonal table: strip header whitespace, rename CE to EC."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    if ec_raw in df.columns and "EC" not in df.columns:
        df = df.rename(columns={ec_raw: "EC"})
    return df


def validate(df: pd.DataFrame, cfg: dict, require_target: bool = True) -> list[str]:
    """Return a list of problems (empty when the table is usable). Range issues are reported as warnings."""
    st, target, feats = cfg["data"]["station_column"], cfg["data"]["target"], cfg["features"]["all"]
    need = [st, *feats] + ([target] if require_target else [])
    problems = [f"missing column: {c}" for c in need if c not in df.columns]
    if problems:
        return problems
    for c in feats + ([target] if require_target else []):
        if not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f"non-numeric column: {c}")
        elif df[c].isna().any():
            problems.append(f"missing values in: {c}")
    if df[st].duplicated().any():
        problems.append("duplicated station labels")
    for c, (lo, hi) in PLAUSIBLE_RANGES.items():
        if c in df and pd.api.types.is_numeric_dtype(df[c]) and ((df[c] < lo) | (df[c] > hi)).any():
            problems.append(f"warning: {c} outside the plausible range {lo}-{hi}")
    return problems


def load_season(data_dir: str | Path, season: str, cfg: dict) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Stations (as given), predictors (configured order, float) and target for one season."""
    d = cfg["data"]
    df = read_table(Path(data_dir) / d["files"][season], d.get("ec_raw_column", "CE"))
    errors = [p for p in validate(df, cfg) if not p.startswith("warning")]
    if errors:
        raise ValueError(f"{season}: " + "; ".join(errors))
    feats = cfg["features"]["all"]
    return df[d["station_column"]].astype(str), df[feats].astype(float), df[d["target"]].astype(float)


def load_all(data_dir: str | Path, cfg: dict) -> dict[str, tuple[pd.Series, pd.DataFrame, pd.Series]]:
    return {s: load_season(data_dir, s, cfg) for s in cfg["data"]["seasons"]}


def drop_well(stations: pd.Series, X: pd.DataFrame, y: pd.Series, label: str):
    """The same season without one well (rows re-indexed)."""
    keep = (stations != label).to_numpy()
    return (stations[keep].reset_index(drop=True), X[keep].reset_index(drop=True), y[keep].reset_index(drop=True))


def zscores(y: pd.Series) -> pd.Series:
    """z-scores with the sample SD (n - 1)."""
    return (y - y.mean()) / y.std(ddof=1)


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
