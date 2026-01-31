from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._demo_utils import load_and_merge_yaml, parse_months_arg, require_cfg, resolve_under_root


def _glob_monthly(dirpath: Path, year: int, suffix: str) -> Dict[int, Path]:
    """
    Return {month:int -> file:Path} for files like YYYYMM<suffix>
    """
    out: Dict[int, Path] = {}
    for m in range(1, 13):
        label = f"{year}{m:02d}"
        p = dirpath / f"{label}{suffix}"
        if p.exists():
            out[m] = p
    return out


def _read_index_info(csv_path: Path) -> Dict[str, str]:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    idx = df.index
    info = {
        "path": str(csv_path),
        "start": str(idx.min()) if len(idx) else "",
        "end": str(idx.max()) if len(idx) else "",
        "n": str(len(idx)),
    }
    if len(idx) >= 2:
        deltas = (idx[1:] - idx[:-1]).astype("timedelta64[s]").astype(int)
        # most common timestep in seconds
        vals, counts = np.unique(deltas, return_counts=True)
        mode = vals[np.argmax(counts)]
        info["mode_timestep_seconds"] = str(int(mode))
    else:
        info["mode_timestep_seconds"] = ""
    info["nans"] = str(int(df.isna().sum().sum()))
    return info


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pypsa_nza_data.demo.nza_demo_data_inventory",
        description="Demo: inventory raw/processed outputs and basic completeness checks.",
    )
    ap.add_argument("--root", required=True, help="Workspace root directory (external).")
    ap.add_argument("--config", action="append", default=[], help="YAML config path (can be repeated).")
    ap.add_argument("--year", type=int, default=2024, help="Year to check (default: 2024).")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not args.config:
        raise SystemExit("At least one --config YAML must be provided.")

    cfg = load_and_merge_yaml(args.config)

    # Directories: raw + processed
    raw_base = None
    if "directories" in cfg and isinstance(cfg["directories"], dict) and "base" in cfg["directories"]:
        raw_base = resolve_under_root(root, cfg["directories"]["base"])
    processed_static = resolve_under_root(root, require_cfg(cfg, "paths.dirpath_static"))
    demand_dir = resolve_under_root(root, require_cfg(cfg, "paths.dirpath_d_MW"))

    gen_dir = None
    if isinstance(cfg.get("paths", {}), dict) and "dirpath_g_MW" in cfg["paths"]:
        gen_dir = resolve_under_root(root, cfg["paths"]["dirpath_g_MW"])

    out_dir = root / "demo" / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inventory basic tree sizes
    def folder_stats(p: Path) -> Dict[str, float]:
        if p is None or not p.exists():
            return {"exists": 0, "files": 0, "bytes": 0}
        files = [x for x in p.rglob("*") if x.is_file()]
        return {"exists": 1, "files": len(files), "bytes": float(sum(x.stat().st_size for x in files))}

    inventory = {
        "root": str(root),
        "year": args.year,
        "raw_base": str(raw_base) if raw_base else None,
        "processed_static": str(processed_static),
        "demand_dir": str(demand_dir),
        "gen_dir": str(gen_dir) if gen_dir else None,
        "stats": {
            "raw_base": folder_stats(raw_base) if raw_base else {"exists": 0, "files": 0, "bytes": 0},
            "processed_static": folder_stats(processed_static),
            "demand_dir": folder_stats(demand_dir),
            "gen_dir": folder_stats(gen_dir) if gen_dir else {"exists": 0, "files": 0, "bytes": 0},
        },
        "completeness": {},
        "samples": {},
    }

    # Monthly completeness checks
    demand_files = _glob_monthly(demand_dir, args.year, "_d_MW.csv")
    inventory["completeness"]["demand_months_present"] = sorted(list(demand_files.keys()))
    inventory["completeness"]["demand_months_missing"] = [m for m in range(1, 13) if m not in demand_files]

    if gen_dir:
        gen_files = _glob_monthly(gen_dir, args.year, "_g_MW.csv")
        inventory["completeness"]["gen_months_present"] = sorted(list(gen_files.keys()))
        inventory["completeness"]["gen_months_missing"] = [m for m in range(1, 13) if m not in gen_files]

    # Sample sanity on one month (Jan if available else first available)
    sample_month = demand_files.get(1) or (next(iter(demand_files.values())) if demand_files else None)
    if sample_month:
        inventory["samples"]["demand_sample"] = _read_index_info(sample_month)

    if gen_dir and "gen_months_present" in inventory["completeness"]:
        # pick gen sample if available
        gen_files = _glob_monthly(gen_dir, args.year, "_g_MW.csv")
        sample_g = gen_files.get(1) or (next(iter(gen_files.values())) if gen_files else None)
        if sample_g:
            inventory["samples"]["gen_sample"] = _read_index_info(sample_g)

    # Write outputs
    json_path = out_dir / "data_inventory.json"
    txt_path = out_dir / "data_inventory.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    # readable text summary
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("NZ data inventory (demo)\n")
        f.write("========================\n\n")
        f.write(f"Workspace root: {root}\n")
        f.write(f"Year: {args.year}\n\n")

        for k, st in inventory["stats"].items():
            f.write(f"{k}:\n")
            f.write(f"  exists: {int(st['exists'])}\n")
            f.write(f"  files: {int(st['files'])}\n")
            f.write(f"  size_MB: {st['bytes'] / (1024**2):.2f}\n")

        f.write("\nCompleteness:\n")
        for k, v in inventory["completeness"].items():
            f.write(f"  {k}: {v}\n")

        f.write("\nSamples:\n")
        for k, v in inventory["samples"].items():
            f.write(f"  {k}:\n")
            for kk, vv in v.items():
                f.write(f"    {kk}: {vv}\n")

    print(f"? Wrote {json_path}")
    print(f"? Wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
