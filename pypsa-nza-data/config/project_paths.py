from pathlib import Path
import yaml

# Project root is two levels above this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "pypsa_nza_data" / "config"

def load_paths_config(filename="paths.yaml"):
    path = CONFIG_DIR / filename
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Convert all paths to Path objects relative to PROJECT_ROOT
    resolved = {}
    for key, rel_path in cfg.items():
        resolved[key] = (PROJECT_ROOT / rel_path).resolve()

    return resolved
