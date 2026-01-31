from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def suppress_demo_warnings() -> None:
    """
    Suppress warning noise for non-optimising demo networks.

    - Keeps exceptions visible (not suppressed)
    - Suppresses typical library warnings that are irrelevant for the JOSS demo
    """
    # Common scientific stack deprecations / chatter
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # PyPSA tends to emit many UserWarnings when building partial networks
    warnings.filterwarnings("ignore", category=UserWarning, module="pypsa")


def configure_pypsa_logging(level: int = logging.ERROR) -> None:
    """
    Reduce PyPSA logger verbosity for demos. Errors still surface.
    """
    logging.getLogger("pypsa").setLevel(level)


def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dicts: values from `other` override `base`.
    """
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_and_merge_yaml(config_paths: List[str]) -> Dict[str, Any]:
    """
    Load multiple YAML configs and merge them (later overrides earlier).
    """
    merged: Dict[str, Any] = {}
    for p in config_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a YAML mapping/dict: {path}")
        merged = deep_update(merged, data)
    return merged


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_months_arg(months: str) -> List[int]:
    """
    Parse --months argument.

    Accepts:
      - "all"
      - "jan" / "feb" / ...
      - "jan,feb,mar"
      - "1..12" (optional shorthand)
      - "1,2,12"

    Returns sorted unique month integers in [1,12].
    """
    m = months.strip().lower()
    if m == "all":
        return list(range(1, 13))

    if ".." in m:
        left, right = m.split("..", 1)
        left_i = int(left.strip())
        right_i = int(right.strip())
        if not (1 <= left_i <= 12 and 1 <= right_i <= 12 and left_i <= right_i):
            raise ValueError(f"Invalid month range: {months}")
        return list(range(left_i, right_i + 1))

    parts = [x.strip().lower() for x in m.split(",") if x.strip()]
    out: List[int] = []
    for p in parts:
        if p in _MONTHS:
            out.append(_MONTHS[p])
        else:
            out.append(int(p))
    out = sorted(set(out))
    if not out or any((mm < 1 or mm > 12) for mm in out):
        raise ValueError(f"Invalid --months value: {months}")
    return out


def require_cfg(cfg: Dict[str, Any], dotted_key: str) -> Any:
    """
    Require a dotted config key like 'paths.dirpath_d_MW'.
    """
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {dotted_key}")
        cur = cur[part]
    return cur


def resolve_under_root(root: Path, rel: str) -> Path:
    """
    Resolve a config-relative path under the workspace root.
    """
    # Ensure rel is treated as relative to root even if it includes leading "./"
    return (root / rel).resolve()
