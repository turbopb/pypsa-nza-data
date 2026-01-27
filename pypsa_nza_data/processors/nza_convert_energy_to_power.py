#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nza_convert_energy_to_power.py

Convert energy time series (MWh) to average power time series (MW/MVAR) for PyPSA
network modeling.

This script operates on CSV files created earlier in the pipeline (e.g. by
nza_create_load_profile.py). For 30-minute trading periods, average power is:

    Power (MW) = Energy (MWh) / 0.5 = Energy (MWh) * 2.0

Execution model (reviewer-safe):
- Config file path is independent of workspace.
- All inputs/outputs/logs are resolved relative to --root (workspace directory).
- No files are written inside the installed package.

Typical usage:
    python -m pypsa_nza_data.processors.nza_convert_energy_to_power \
        --config pypsa_nza_data/config/nza_convert_energy_to_power.yaml \
        --root /home/reviewer/nz_workspace \
        --profile gen_demand
"""

import argparse
import glob
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
import importlib.resources as resources

# ============================================================================
# CONSTANTS
# ============================================================================

ENERGY_TO_POWER_FACTOR_DEFAULT = 2.0  # for 30-minute periods
TRADING_PERIOD_HOURS = 0.5

logger = logging.getLogger(__name__)

# ============================================================================
# LOGGING & PATH HELPERS
# ============================================================================

def resolve_default_config_path() -> Path:
    """Return the packaged default YAML config path."""
    with resources.as_file(resources.files("pypsa_nza_data").joinpath("config", "nza_convert_energy_to_power")) as p:
        return Path(p)

def resolve_workspace_root(arg_root: Optional[str]) -> Path:
    if not arg_root:
        raise ValueError("No workspace root provided. Pass --root <WORKSPACE>.")
    return Path(arg_root).expanduser().resolve()

def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nza_convert_energy_to_power_{ts}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    fmt_file = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fmt_console = logging.Formatter("%(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt_console)
    root_logger.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt_file)
    root_logger.addHandler(fh)

    return log_file

def resolve_relpath(root: Path, rel: str) -> Path:
    p = Path(rel)
    # force resolution under workspace even if YAML provides absolute paths
    if p.is_absolute():
        return (root / p.name).resolve()
    return (root / p).resolve()

# ============================================================================
# CONFIG
# ============================================================================

def load_config_file(config_path: Path, profile: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    defaults = config.get("default", {}) or {}
    energy_types_config = config.get("energy_types", {}) or {}

    if profile:
        profiles = config.get("profiles", {}) or {}
        if profile not in profiles:
            available = list(profiles.keys())
            raise KeyError(
                f"Profile '{profile}' not found in config file. Available profiles: {available}"
            )
        energy_types_to_process = profiles[profile].get("energy_types", list(energy_types_config.keys()))
    else:
        energy_types_to_process = list(energy_types_config.keys())

    result = {
        "defaults": defaults,
        "energy_types": {k: energy_types_config[k] for k in energy_types_to_process if k in energy_types_config},
        "types_to_process": [k for k in energy_types_to_process if k in energy_types_config],
        # optional workspace-relative logs path (not required in YAML)
        "paths": (config.get("paths") or {}),
    }
    return result

# ============================================================================
# UTILITIES
# ============================================================================

def ensure_directory_exists(directory_path: Path) -> bool:
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"✗ Could not create directory: {directory_path} ({e})")
        return False


def get_csv_files_by_pattern(directory_path: Path, pattern: str = "*.csv") -> List[Path]:
    if not directory_path.is_dir():
        return []
    full_pattern = str(directory_path / pattern)
    return sorted(Path(p) for p in glob.glob(full_pattern) if Path(p).is_file())


def generate_output_filename(input_filename: str, output_suffix: str) -> str:
    prefix = input_filename[:6]  # YYYYMM
    return prefix + output_suffix


def validate_energy_dataframe(df: pd.DataFrame, allow_negative: bool = False) -> Tuple[bool, Optional[str]]:
    if df.empty:
        return False, "DataFrame is empty"
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "Index is not DatetimeIndex (expected DATE column parsed as index)"
    if df.isna().any().any():
        num_missing = int(df.isna().sum().sum())
        return False, f"Contains {num_missing} missing values"
    if not allow_negative and (df < 0).any().any():
        num_negative = int((df < 0).sum().sum())
        return False, f"Contains {num_negative} negative values"
    if not all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
        return False, "Non-numeric columns found"
    return True, None


def convert_energy_to_power(df_energy: pd.DataFrame, conversion_factor: float) -> pd.DataFrame:
    return df_energy * conversion_factor


def process_energy_file(input_filepath: Path, output_filepath: Path, conversion_factor: float, allow_negative: bool) -> bool:
    try:
        df_energy = pd.read_csv(input_filepath, index_col="DATE", parse_dates=True)
        ok, msg = validate_energy_dataframe(df_energy, allow_negative=allow_negative)
        if not ok:
            logger.error(f"     ✗ Validation failed: {msg}")
            return False

        df_power = convert_energy_to_power(df_energy, conversion_factor)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        # Keep DATE as index in output (PyPSA time series style)
        df_power.to_csv(output_filepath, index=True)

        input_kb = input_filepath.stat().st_size / 1024
        output_kb = output_filepath.stat().st_size / 1024
        logger.info(f"     ✓ Converted: {output_filepath.name}")
        logger.info(f"       Rows: {len(df_power)}, Columns: {len(df_power.columns)}")
        logger.info(f"       Size: {input_kb:.1f} KB -> {output_kb:.1f} KB")
        return True

    except FileNotFoundError:
        logger.error(f"     ✗ File not found: {input_filepath}")
        return False
    except pd.errors.EmptyDataError:
        logger.error(f"     ✗ Empty file: {input_filepath}")
        return False
    except Exception as e:
        logger.error(f"     ✗ Error processing file: {e}")
        return False

def process_energy_type(root: Path, energy_type: str, cfg: dict, conversion_factor: float) -> Dict[str, int]:
    stats = {"total": 0, "successful": 0, "failed": 0}

    input_dir_rel = cfg.get("input_dir")
    output_dir_rel = cfg.get("output_dir")
    input_pattern = cfg.get("input_pattern", "*.csv")
    output_suffix = cfg.get("output_suffix", ".csv")
    allow_negative = bool(cfg.get("allow_negative", False))
    description = cfg.get("description", energy_type)

    if not input_dir_rel or not output_dir_rel:
        logger.error(f"✗ انرژی type '{energy_type}' missing input_dir/output_dir in YAML; skipping.")
        return stats

    input_dir = resolve_relpath(root, str(input_dir_rel))
    output_dir = resolve_relpath(root, str(output_dir_rel))

    unit = "MVAR" if "MVAR" in str(output_suffix) else "MW"

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PROCESSING: {description.upper()}")
    logger.info("=" * 80)
    logger.info(f"Input:   {input_dir}")
    logger.info(f"Output:  {output_dir}")
    logger.info(f"Pattern: {input_pattern}")
    logger.info(f"Units:   MWh -> {unit}")
    logger.info(f"Factor:  {conversion_factor}")
    logger.info(f"Allow negative: {allow_negative}")

    if not input_dir.is_dir():
        logger.warning("  ⚠ Input directory does not exist, skipping.")
        return stats

    files = get_csv_files_by_pattern(input_dir, input_pattern)
    stats["total"] = len(files)
    if stats["total"] == 0:
        logger.warning(f"  ⚠ No files matching pattern '{input_pattern}' found.")
        return stats

    logger.info(f"  Found {stats['total']} files to process")

    if not ensure_directory_exists(output_dir):
        logger.error("  ✗ Could not create output directory, skipping.")
        return stats

    logger.info("")
    logger.info("  Processing files:")
    for idx, input_filepath in enumerate(files, 1):
        logger.info("")
        logger.info(f"  [{idx}/{stats['total']}] {input_filepath.name}")

        output_filename = generate_output_filename(input_filepath.name, str(output_suffix))
        output_filepath = output_dir / output_filename

        success = process_energy_file(input_filepath, output_filepath, conversion_factor, allow_negative)
        if success:
            stats["successful"] += 1
        else:
            stats["failed"] += 1

    logger.info("")
    logger.info("  Summary:")
    logger.info(f"    Total files:      {stats['total']}")
    logger.info(f"    Successful:       {stats['successful']}")
    logger.info(f"    Failed:           {stats['failed']}")
    if stats["total"] > 0:
        logger.info(f"    Success rate:     {stats['successful']/stats['total']*100:.1f}%")
    return stats

# ============================================================================
# ENTRY-POINT
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert energy time series (MWh) to average power (MW/MVAR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--root", default=None, help="Workspace root (required).")
    parser.add_argument("-c", "--config", default=None, help="Path to YAML configuration file.")
    parser.add_argument("-p", "--profile", default=None, help='Profile name from config (e.g., "gen_only").')
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    parser.add_argument("--version", action="version", version="%(prog)s 2.1.0")

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    start_time = time.time()

    try:
        workspace_root = resolve_workspace_root(args.root)
    except ValueError as e:
        print(f"Fatal error: {e}")
        return 2

    config_path = Path(args.config).expanduser().resolve() if args.config else resolve_default_config_path()
    profile = args.profile

    # Load config first (to find logs path), then set up logging
    try:
        cfg = load_config_file(config_path, profile)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 2

    logs_rel = (cfg.get("paths") or {}).get("logs", "logs")
    log_dir = resolve_relpath(workspace_root, str(logs_rel))
    log_file = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("NZA ENERGY TO POWER CONVERTER")
    logger.info("=" * 80)
    logger.info(f"Start time:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config:         {config_path}")
    logger.info(f"Workspace root: {workspace_root}")
    logger.info(f"Log file:       {log_file}")
    logger.info(f"Profile:        {profile if profile else '(none)'}")
    logger.info("")

    defaults = cfg.get("defaults", {}) or {}
    conversion_factor = float(defaults.get("conversion_factor", ENERGY_TO_POWER_FACTOR_DEFAULT))
    types_to_process = cfg.get("types_to_process", [])
    energy_types_config = cfg.get("energy_types", {})

    logger.info("Configuration:")
    logger.info(f"  Conversion factor:  {conversion_factor}")
    logger.info(f"  Energy types:       {', '.join(types_to_process) if types_to_process else '(none)'}")

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN: Resolved directories per energy type:")
        for et in types_to_process:
            c = energy_types_config.get(et, {})
            inp = resolve_relpath(workspace_root, str(c.get("input_dir", ""))) if c.get("input_dir") else None
            outp = resolve_relpath(workspace_root, str(c.get("output_dir", ""))) if c.get("output_dir") else None
            logger.info(f"  - {et}:")
            logger.info(f"      input_dir:  {inp}")
            logger.info(f"      output_dir: {outp}")
        return 0

    all_stats: Dict[str, Dict[str, int]] = {}
    for et in types_to_process:
        stats = process_energy_type(workspace_root, et, energy_types_config[et], conversion_factor)
        all_stats[et] = stats

    logger.info("")
    logger.info("=" * 80)
    logger.info("OVERALL SUMMARY")
    logger.info("=" * 80)

    total_files = sum(s["total"] for s in all_stats.values())
    total_successful = sum(s["successful"] for s in all_stats.values())
    total_failed = sum(s["failed"] for s in all_stats.values())

    logger.info("Processing Statistics:")
    logger.info(f"  Energy types processed:  {len(all_stats)}")
    logger.info(f"  Total files found:       {total_files}")
    logger.info(f"  Successfully converted:  {total_successful}")
    logger.info(f"  Failed conversions:      {total_failed}")
    if total_files > 0:
        logger.info(f"  Overall success rate:    {total_successful/total_files*100:.1f}%")

    logger.info("")
    logger.info("By Energy Type:")
    for et, s in all_stats.items():
        if s["total"] > 0:
            rate = s["successful"] / s["total"] * 100
            logger.info(f"  {et:10s}: {s['successful']:3d}/{s['total']:3d} ({rate:5.1f}%)")
        else:
            logger.info(f"  {et:10s}: No files found")

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End time:            {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed time:  {elapsed:.2f} seconds")
    if total_files > 0:
        logger.info(f"Average time/file:   {elapsed/total_files:.2f} seconds")

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        raise SystemExit(130)
