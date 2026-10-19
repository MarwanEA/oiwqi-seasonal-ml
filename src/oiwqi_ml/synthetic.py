"""Synthetic seasonal tables with the same schema as the real data.

Ion concentrations are drawn from generic groundwater ranges (not from any study), the six
derived indices are computed with the formulas as implemented (all ions in meq/L), EC and TDS
follow the ionic load, and the OIWQI is computed with oiwqi_ml.index. Values are rounded to two
decimals before the index is computed, so the stored target is exact for the stored predictors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import index

EQ_WEIGHT = {"Ca": 20.04, "Mg": 12.15, "Na": 22.99, "K": 39.10, "HCO3": 61.02, "Cl": 35.45, "SO4": 48.03}
ION_RANGES = {"Ca": (40, 180), "Mg": (15, 120), "Na": (20, 200), "K": (2, 25), "HCO3": (150, 550),
              "Cl": (40, 300), "SO4": (30, 200), "NO3": (1, 60)}
SEASON_FACTOR = {0: 1.0, 1: 1.15, 2: 0.75}  # by season position: concentrated, flushed, diluted


def derive_indices(ions: pd.DataFrame) -> pd.DataFrame:
    """SAR, Na%, MAR, PI, RSBC and SSP from ions in mg/L, with the formulas as implemented."""
    m = {k: ions[k] / w for k, w in EQ_WEIGHT.items()}
    cat = m["Ca"] + m["Mg"] + m["Na"] + m["K"]
    return pd.DataFrame({
        "SAR": m["Na"] / np.sqrt((m["Ca"] + m["Mg"]) / 2),
        "Na%": (m["Na"] + m["K"]) / cat * 100,
        "MAR": m["Mg"] / (m["Ca"] + m["Mg"]) * 100,
        "PI": (m["Na"] + m["HCO3"]) / cat * 100,
        "RSBC": m["HCO3"] - (m["Ca"] + m["Mg"]),
        "SSP": m["Na"] / cat * 100,
    }, index=ions.index)


def generate(cfg: dict, n_wells: int = 20, seed: int | None = None) -> dict[str, pd.DataFrame]:
    """One table per configured season: station column, 17 predictors and the OIWQI target."""
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    base = pd.DataFrame({k: rng.uniform(lo, hi, n_wells) for k, (lo, hi) in ION_RANGES.items()})
    base_ph = rng.uniform(7.0, 8.3, n_wells)
    out = {}
    for i, season in enumerate(cfg["data"]["seasons"]):
        f = SEASON_FACTOR.get(i, 1.0)
        ions = base * f * rng.lognormal(0.0, 0.08, base.shape)
        df = ions.copy()
        df["pH"] = np.clip(base_ph + rng.normal(0, 0.1, n_wells), 6.6, 8.6)
        cat_meq = sum(ions[k] / EQ_WEIGHT[k] for k in ("Ca", "Mg", "Na", "K"))
        df["EC"] = cat_meq * 100 * rng.lognormal(0.0, 0.03, n_wells)
        df["TDS"] = df["EC"] * 0.64 * rng.lognormal(0.0, 0.03, n_wells)
        df = pd.concat([df, derive_indices(ions)], axis=1)[cfg["features"]["all"]].round(2)
        df[cfg["data"]["target"]] = index.oiwqi(df).round(6)
        df.insert(0, cfg["data"]["station_column"], [f"S{j:02d}" for j in range(1, n_wells + 1)])
        out[season] = df
    return out


def write(data_dir: str | Path, cfg: dict, n_wells: int = 20, seed: int | None = None) -> Path:
    """Write the synthetic tables under the configured file names; returns the directory."""
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for season, df in generate(cfg, n_wells, seed).items():
        df.to_csv(d / cfg["data"]["files"][season], index=False)
    return d
