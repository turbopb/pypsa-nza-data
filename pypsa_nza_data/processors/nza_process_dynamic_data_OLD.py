#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nza_process_dynamic_data.py

Process and standardize raw electricity generation and load data from New 
Zealand's electricity market for PyPSA network modeling.

DESCRIPTION
-----------
This module transforms raw monthly electricity market data from Transpower and 
the Electricity Authority into clean, standardized time series suitable for 
PyPSA (Python for Power System Analysis). The processing pipeline handles data 
aggregation, missing value imputation, unit conversion, and temporal formatting.

INPUT DATA FORMAT
-----------------
CSV files with the following structure:
    - POC: Point of Connection identifier (string)
    - Trading_Date: Date stamp in format YYYY-MM-DD
    - TP1...TP50: Trading Periods (30-min intervals, values in kWh)
        * TP1 = 00:00-00:30, TP48 = 23:30-24:00
        * TP49, TP50 = Daylight saving transition periods (discarded)

Each file contains one calendar month of data. Multiple rows may exist for the 
same POC-date combination (which are aggregated). Some dates may be missing 
(which are zero-filled).

PROCESSING STEPS
----------------
1. Standardize POC column naming
2. Aggregate duplicate entries (sum) for same POC + Trading_Date
3. Fill missing dates with zeros to complete the calendar month
4. Remove daylight saving adjustment columns (TP49, TP50)
5. Flatten wide format into continuous time series vectors
6. Convert units from kWh to MWh (scale factor: 0.001)
7. Apply 30-minute datetime index to output

OUTPUT FORMAT
-------------
CSV files with:
    - Rows: 30-minute datetime index (DATE column)
    - Columns: One column per POC containing energy values in MWh
    - Filename pattern: YYYYMM_<energy_type>_md.csv

Total rows per month: num_days × 48 half-hourly periods
    Example: 31 days = 1,488 rows (31 × 48)

CONFIGURATION
-------------
Reads YAML configuration file specifying:
    - Root directory path
    - Input/output directories per energy type
    - Grid energy type labels
    - File naming conventions

Path: <ROOT_DIR>/config/nza_raw_dyn_data.yaml

LOGGING
-------
- Creates timestamped log files in the logs/ directory
- Logs to both console (simple format) and file (detailed format)
- Log files include full processing details for debugging

PATH HANDLING
-------------
- Uses pathlib.Path throughout for cross-platform compatibility
- Works identically on Windows and Linux
- YAML config should use forward slashes (/) - works on both platforms
- All paths converted to Path objects at initialization

USAGE
-----
    python nza_process_dynamic_data.py

The script processes all energy types specified in the configuration file
(typically 'generation' and 'load') and outputs cleaned monthly CSV files.

DEPENDENCIES
------------
    - pandas: DataFrame operations and time series handling
    - pyyaml: Configuration file parsing
    - numpy: Numerical operations (via pandas)
    - datetime, calendar: Date/time utilities

NOTES
-----
    - All numeric values are scaled from kWh to MWh
    - Missing dates are backfilled with zeros (not interpolated)
    - Duplicate POC-date entries are summed (not averaged)
    - TP49 and TP50 are always removed regardless of content

AUTHOR
------
    Philippe Bruneau

CREATED
-------
    Created on Thu Dec  4 15:38:23 2025

MODIFIED
--------
    2025-12-17 - Added professional logging and formatted output
               - Unified path handling with pathlib.Path
               - Improved console output with headers and sections
               - Cross-platform compatibility improvements

VERSION
-------
    1.2.0

nza_process_dynamic_data.py

Process NZ electricity authority "dynamic" time series (half-hourly trading-period data)
into clean, standardized time series suitable for downstream modelling (e.g. PyPSA).

This module is intended to be runnable as an installed package module:

    python -m pypsa_nza_data.processors.nza_process_dynamic_data --config <path>

If --config is omitted, a packaged default YAML is used from:
    pypsa_nza_data/config/nza_process_dynamic_data.yaml
"""

from __future__ import annotations

import sys
import time
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import pandas as pd
import numpy as np

try:
    from importlib.resources import files as pkg_files
except Exception:  # pragma: no cover
    pkg_files = None


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    """Set up console + file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nza_process_dynamic_data_{timestamp}.log"

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter("%(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear handlers if re-run in the same interpreter session
    while root_logger.handlers:
        root_logger.handlers.pop()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)


def print_header(title: str, char: str = "-") -> None:
    line = char * 80
    logger.info(line)
    logger.info(title.center(80))
    logger.info(line)


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def _default_config_path() -> Path:
    """
    Return the packaged default config path:
        pypsa_nza_data/config/nza_process_dynamic_data.yaml
    """
    if pkg_files is None:
        raise RuntimeError("importlib.resources.files not available; please use Python >= 3.9")

    candidate = pkg_files("pypsa_nza_data").joinpath("config/nza_process_dynamic_data.yaml")
    return Path(str(candidate))


def load_config(config_file: Path) -> Dict:
    """Read YAML configuration file."""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config if config else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


# ============================================================================
# DIRECTORY AND FILE UTILITIES
# (Your existing functions follow unchanged)
# ============================================================================

def list_filenames_in_directory(directory_path: Path, include_full_path: bool = True) -> List[str]:
    if not directory_path.exists():
        return []
    files = sorted([p for p in directory_path.iterdir() if p.is_file()])
    if include_full_path:
        return [str(p) for p in files]
    return [p.name for p in files]


# ----------------------------------------------------------------------------
# Your existing processing logic below (kept intact)
# ----------------------------------------------------------------------------

DEFAULT_SCALE_FACTOR = 1e-3  # kWh → MWh


def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()
    if df_.empty:
        return df_
    df_.columns = df_.columns.astype(str)
    return df_.groupby(level=0, axis=1).sum()


def generate_half_hourly_index(year: int, month: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=year, month=month, day=1, hour=0, minute=0)
    # End is inclusive for the last trading period of the month (30-minute steps)
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(minutes=30)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(minutes=30)
    return pd.date_range(start=start, end=end, freq="30min")


def standardize_time_index(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Ensure DF has a complete 30-min index for the month; fill missing with zeros."""
    expected_index = generate_half_hourly_index(year, month)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.reindex(expected_index)
    df = df.fillna(0.0)
    return df


def safe_read_csv(file_path: Path) -> pd.DataFrame:
    """Read CSV robustly."""
    return pd.read_csv(file_path)


def process_single_file(file_path: Path, year: int, month: int, scale_factor: float) -> pd.DataFrame:
    """
    Process one monthly file:
    - load CSV
    - parse datetime column (or infer)
    - aggregate duplicate columns
    - reindex to complete trading periods
    - apply scale factor
    """
    df_raw = safe_read_csv(file_path)

    # Heuristic: first column is timestamp
    if df_raw.shape[1] < 2:
        raise ValueError(f"Unexpected structure in {file_path}: needs at least 2 columns")

    ts_col = df_raw.columns[0]
    df_raw[ts_col] = pd.to_datetime(df_raw[ts_col])
    df_raw = df_raw.set_index(ts_col)

    # Aggregate duplicate columns (e.g. duplicate POC names)
    df = aggregate_duplicates(df_raw)

    # Fill missing trading periods
    df = standardize_time_index(df, year=year, month=month)

    # Drop known non-grid points (your earlier logic)
    for bad in ("TP49", "TP50"):
        if bad in df.columns:
            df = df.drop(columns=[bad])

    # Convert kWh → MWh (or other scaling as configured)
    df = df * scale_factor

    return df


def process_files_in_directory(input_dir: Path, output_dir: Path, file_tag: str, scale_factor: float) -> int:
    """
    Process all monthly CSVs in a directory. Expects filenames containing YYYYMM.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [Path(p) for p in list_filenames_in_directory(input_dir, include_full_path=True)]

    if not files:
        logger.warning(f"No files found in: {input_dir}")
        return 0

    processed_count = 0

    for file_path in files:
        name = file_path.name
        # Find YYYYMM in filename
        digits = "".join(ch for ch in name if ch.isdigit())
        if len(digits) < 6:
            logger.warning(f"Skipping file without YYYYMM in name: {name}")
            continue

        yyyymm = digits[:6]
        year = int(yyyymm[:4])
        month = int(yyyymm[4:6])

        logger.info(f"Processing: {name}  (year={year}, month={month})")

        try:
            df = process_single_file(file_path, year=year, month=month, scale_factor=scale_factor)
        except Exception as e:
            logger.error(f"✗ Failed processing {name}: {e}")
            continue

        out_file = output_dir / f"{yyyymm}_{file_tag}.csv"
        df.to_csv(out_file, index=True)
        processed_count += 1
        logger.info(f"  ✓ Wrote: {out_file}")

    logger.info("")
    logger.info(f"Successfully processed {processed_count}/{len(files)} files")
    return processed_count


# ============================================================================
# CLI + MAIN
# ============================================================================

def parse_args(argv: Optional[List[str]] = None):
    import argparse

    p = argparse.ArgumentParser(
        prog="nza_process_dynamic_data",
        description="Process NZ dynamic (half-hourly) datasets into standardised time series.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If omitted, uses packaged default.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    warnings.filterwarnings("ignore")

    args = parse_args(argv)
    config_file = Path(args.config) if args.config else _default_config_path()

    start_time = time.time()

    try:
        config = load_config(config_file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"✗ Fatal error loading configuration: {e}")
        return 1

    # Root resolution:
    # - Prefer YAML paths.root if present
    # - Otherwise default to the current working directory (reviewer-friendly)
    root = Path(config.get("paths", {}).get("root", Path.cwd()))
    log_dir = root / "logs"

    setup_logging(log_dir)

    print_header("PYPSA-NZA DYNAMIC DATA PROCESSOR", "=")
    logger.info(f"Config: {config_file}")
    logger.info(f"Root:   {root}")
    logger.info("")

    grid_energy_types = config.get("grid_energy_types", [])
    if not grid_energy_types:
        logger.error("✗ No grid_energy_types specified in configuration")
        return 1

    scale_factor = float(config.get("scale_factor", DEFAULT_SCALE_FACTOR))
    logger.info(f"Energy types to process: {', '.join(grid_energy_types)}")
    logger.info(f"Scale factor:            {scale_factor}")
    logger.info("")

    processed_total = 0

    for grid_energy in grid_energy_types:
        print_header(f"PROCESSING: {grid_energy.upper()}", "=")

        input_rel = config.get("paths", {}).get(f"{grid_energy}_in", "")
        output_rel = config.get("paths", {}).get(f"{grid_energy}_out", "")

        input_dir = root / input_rel
        output_dir = root / output_rel
        file_tag = f"{grid_energy}_md"

        if not input_dir.exists():
            logger.error(f"✗ Input directory does not exist: {input_dir}")
            continue

        logger.info(f"Input directory:  {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"File tag:         {file_tag}")
        logger.info("")

        processed_total += process_files_in_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            file_tag=file_tag,
            scale_factor=scale_factor,
        )

    elapsed = time.time() - start_time
    print_header("SUMMARY", "=")
    logger.info(f"Total files processed: {processed_total}")
    logger.info(f"Elapsed:              {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
