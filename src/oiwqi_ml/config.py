"""Configuration loading (YAML): default file, optional profile, user file and overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None,
                profile: str | None = None) -> dict:
    """Merge, in order: the default configuration, a named profile from it, a user file, then overrides."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if profile:
        cfg = _merge(cfg, cfg["profiles"][profile])
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return _merge(cfg, overrides or {})
