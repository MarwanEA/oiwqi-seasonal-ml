"""Command-line options and shared helpers for the pipeline scripts."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", message="Objective did not converge")  # Lasso/ElasticNet at small alpha, as run
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")

import pandas as pd  # noqa: E402

from oiwqi_ml import synthetic  # noqa: E402
from oiwqi_ml.config import load_config  # noqa: E402
from oiwqi_ml.data import load_all  # noqa: E402
from oiwqi_ml.explain import explainer_kind  # noqa: E402
from oiwqi_ml.validation import safe_workers, set_worker_env, write_json  # noqa: E402


def setup(description: str) -> SimpleNamespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", help="folder with one table per season (see data/README.md)")
    ap.add_argument("--synthetic", action="store_true", help="generate and use a synthetic dataset")
    ap.add_argument("--full", action="store_true", help="with --synthetic: use the full settings, not the quick profile")
    ap.add_argument("--config", help="YAML file whose keys override configs/default.yaml")
    ap.add_argument("--out", default=".", help="output folder for results/, figures/ and checkpoints/ (default: .)")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes (one thread each)")
    a = ap.parse_args()
    if not a.synthetic and not a.data_dir:
        ap.error("give --data-dir or --synthetic")
    cfg = load_config(a.config, profile="quick" if a.synthetic and not a.full else None)
    out = Path(a.out)
    data_dir = synthetic.write(out / "data" / "synthetic", cfg) if a.synthetic else Path(a.data_dir)
    set_worker_env()
    run = SimpleNamespace(cfg=cfg, data_dir=data_dir, out=out, results=out / "results", figures=out / "figures",
                          checkpoints=out / "checkpoints", workers=safe_workers(a.workers))
    run.results.mkdir(parents=True, exist_ok=True)
    run.figures.mkdir(parents=True, exist_ok=True)
    return run


def data(run) -> dict:
    return load_all(run.data_dir, run.cfg)


def label(run, season: str) -> str:
    return run.cfg["data"]["season_labels"].get(season, season)


def save_csv(df: pd.DataFrame, run, name: str, index: bool = False) -> None:
    df.to_csv(run.results / name, index=index)
    print(f"wrote results/{name}")


def save_params(params: dict, run, name: str) -> None:
    write_json(run.results / name, params)
    print(f"wrote results/{name}")


def load_params(run, name: str = "original_params.json") -> dict:
    return json.loads((run.results / name).read_text(encoding="utf-8"))


def require(run, name: str) -> Path:
    p = run.results / name
    if not p.exists():
        sys.exit(f"missing results/{name}: run the earlier pipeline steps first")
    return p


def explained(run) -> dict[str, str]:
    """Model explained per season: the configured mapping, or the best original-protocol R2 among
    linear and tree models (models with an exact explainer)."""
    e = run.cfg["models"]["explained"]
    if isinstance(e, dict):
        return e
    m = pd.read_csv(require(run, "original_metrics.csv"))
    m = m[m.model.map(explainer_kind) != "kernel"]
    return {s: g.sort_values("R2", ascending=False).model.iloc[0] for s, g in m.groupby("season", sort=False)}
