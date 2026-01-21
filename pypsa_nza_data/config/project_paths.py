from __future__ import annotations
from importlib.resources import files
from pathlib import Path
import platform
import yaml

# Project root is two levels above this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _select_os_path(value):
    """Allow YAML values to be either a string path or a dict with OS keys."""
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        system = platform.system().lower()
        # Normalise keys
        keys = {k.lower(): v for k, v in value.items()}

        if system.startswith("win") and "windows" in keys:
            return keys["windows"]
        if (system.startswith("linux") or system.startswith("darwin")) and "linux" in keys:
            return keys["linux"]

        # Fallbacks
        if "default" in keys:
            return keys["default"]

        raise ValueError(f"OS-specific path dict missing suitable key: {value}")

    raise TypeError(f"Unsupported path config type: {type(value)}")

def load_paths_config(config_path: str | Path | None = None) -> dict[str, Path]:
    # 1) locate default config shipped in the package
    if config_path is None:
        config_path = files("pypsa_nza_data").joinpath("config/paths.yaml")
    else:
        config_path = Path(config_path)

    # 2) load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 3) interpret config (you may already have this part)
    # For now assume it is a flat mapping of {key: relative_path_string}
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root in editable installs

    resolved = {}
    for key, rel_path in cfg.items():
        if isinstance(rel_path, dict):
            raise TypeError(f"Expected string path for '{key}', got dict. Update YAML or resolver.")
        resolved[key] = (PROJECT_ROOT / rel_path).resolve()

    return resolved
