 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nza_create_load_profile.py

DESCRIPTION
-----------
Creates and prepares bus, generation and demand load profiles for capital
expansion planning analysis using PyPSA.

WORKFLOW OVERVIEW
-----------------
This script processes energy grid data to create load profiles for buses, generation,
and demand analysis. The main workflow consists of:

1. Configuration Loading: Reads YAML configuration file containing paths and parameters
2. Data Import: Loads monthly grid import/export data and aggregates by site prefix
3. Bus Processing:
   - Extracts participating bus information from master dataset
   - Validates bus connectivity using transmission line topology
   - Generates monthly bus data files
4. Generation Processing:
   - Aggregates generation data by site prefix
   - Handles site anomalies (HWB/WPG merger, CML/CYD consolidation)
   - Creates monthly generation dispatch files
5. Demand Calculation:
   - Computes net energy flow (export - import) for each site
   - Combines generation and net flow data to calculate demand
   - Applies data cleaning (removes negatives, handles missing sites)
   - Outputs monthly demand profile files

KEY FEATURES
------------
- Processes 12 months of energy data automatically
- Handles site name consolidations and data anomalies
- Validates grid connectivity for all participating buses
- Implements robust error handling and data validation
- Generates standardized CSV outputs for PyPSA integration

DATA SOURCES
------------
- Grid import/export files (monthly, by site)
- Generation dispatch data (monthly, by site)
- Master bus information (coordinates, metadata)
- Transmission line topology data

OUTPUT FILES
------------
- Monthly bus data files with participating sites
- Monthly aggregated generation dispatch files
- Monthly net energy flow (delta) files
- Monthly demand profile files for capital expansion analysis

NOTES
-----
- HWB and WPG sites are consolidated (WPG data added to HWB)
- CML site data is merged with CYD when present
- All negative demand values are clipped to zero
- Disconnected buses are identified and reported

DEPENDENCIES
------------
    - pandas: DataFrame operations
    - numpy: Numerical operations
    - pyyaml: Configuration file parsing
    - pathlib: Path operations

AUTHOR
------
    Philippe Bruneau

CREATED
-------
    Wed May 28 16:13:34 2025

MODIFIED
--------
    2025-12-07 - Standardized with nza_root.py integration
               - Fixed ROOT_DIR conflicts
               - Added proper error handling
               - Improved cross-platform compatibility
               - Standardized documentation

VERSION
-------
    1.2.0
"""

import sys
import os
import calendar
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import importlib
from importlib import resources as importlib_resources



logger = logging.getLogger(__name__)

def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nza_create_load_profile_{ts}.log"

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


def resolve_path(p: str, base: Path) -> str:
    """Return a filesystem path string: absolute if p absolute, else resolved under base."""
    pth = Path(p)
    return str(pth) if pth.is_absolute() else str((base / pth).resolve())

def default_config_path() -> Path:
    """Find the default YAML config shipped in pypsa_nza_data/config."""
    pkg = importlib.import_module("pypsa_nza_data.config")
    candidates = [
        "nza_create_load_profile.yaml",
        "nza_create_load_profiles.yaml",
        "nza_cx_config.yaml",
    ]
    for name in candidates:
        t = importlib_resources.files(pkg) / name
        try:
            with importlib_resources.as_file(t) as p:
                if Path(p).exists():
                    return Path(p)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "No default load-profile config found in pypsa_nza_data/config. "
        "Pass --config explicitly."
    )

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="nza_create_load_profile",
        description="Create bus-level half-hourly load profiles from downloaded NZ data."
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config. If omitted, uses a packaged default from pypsa_nza_data/config/.")
    p.add_argument("--root", type=str, default=None,
                   help="Workspace root used to resolve relative paths in the YAML (data outside repo).")
    return p.parse_args(argv)


import yaml
import pandas as pd
import numpy as np

# =============================================================================
# PROJECT ROOT DETECTION
# =============================================================================

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

def load_config(config_file: str) -> Dict:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration data as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    yaml.YAMLError
        If there's an error parsing the YAML file.
        
    Examples
    --------
    >>> config = load_config('config/settings.yaml')
    >>> print(config['paths']['root'])
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Configuration file '{config_file}' not found."
        )
    
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config if config else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


# =============================================================================
# DIRECTORY AND FILE UTILITIES
# =============================================================================

def ensure_directory_exists(directory_path: str, verbose: bool = True) -> bool:
    """
    Check if a directory exists and create it if it doesn't.

    Parameters
    ----------
    directory_path : str
        Path to the directory to check/create.
    verbose : bool, optional
        If True, print status messages. Default is True.

    Returns
    -------
    bool
        True if directory exists or was successfully created, False otherwise.

    Examples
    --------
    >>> ensure_directory_exists('/path/to/output')
    Directory created: /path/to/output
    True
    """
    if not directory_path:
        print("Error: Directory path cannot be empty or None.")
        return False

    directory_path = os.path.normpath(directory_path)

    try:
        if os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                if verbose:
                    print(f"Directory already exists: {directory_path}")
                return True
            else:
                print(f"Error: Path exists but is not a directory: {directory_path}")
                return False

        os.makedirs(directory_path, exist_ok=True)

        if verbose:
            print(f"Directory created: {directory_path}")
        return True

    except PermissionError:
        print(f"Error: Permission denied. Cannot create directory: {directory_path}")
        print("Please check your permissions or try running with appropriate privileges.")
        return False
    except OSError as e:
        print(f"Error: Could not create directory '{directory_path}'")
        print(f"Reason: {e}")
        return False


def list_filenames_in_directory(
    directory_path: str, 
    include_full_path: bool = True
) -> List[str]:
    """
    Return a list of filenames in the specified directory.

    Parameters
    ----------
    directory_path : str
        Path to the directory.
    include_full_path : bool, optional
        If True, returns full file paths instead of just filenames.

    Returns
    -------
    List[str]
        A list of filenames or file paths.

    Raises
    ------
    ValueError
        If the specified path does not exist or is not a directory.
        
    Examples
    --------
    >>> files = list_filenames_in_directory('/data/generation')
    >>> print(len(files))
    12
    """
    if not os.path.isdir(directory_path):
        raise ValueError(
            f"The specified path does not exist or is not a directory: "
            f"{directory_path}"
        )

    files = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files.append(full_path if include_full_path else entry)

    return files


def generate_monthly_filenames(
    year: int, 
    months: str, 
    basename: str, 
    extension: str
) -> List[str]:
    """
    Generate monthly filenames based on year, months, basename, and extension.

    Parameters
    ----------
    year : int
        Year for the filenames.
    months : str
        Month specification ('all', 'jan-mar', 'jan,feb,mar', etc.).
    basename : str
        Base name for the files.
    extension : str
        File extension (without dot).

    Returns
    -------
    List[str]
        List of generated filenames in format: YYYYMM_basename.extension
        
    Examples
    --------
    >>> files = generate_monthly_filenames(2024, 'all', 'bus_data', 'csv')
    >>> print(files[0])
    '202401_bus_data.csv'
    
    >>> files = generate_monthly_filenames(2024, 'jan-mar', 'demand_md', 'csv')
    >>> print(len(files))
    3
    """
    year = str(year)
    extension = extension.strip().lstrip(".")
    basename = basename.strip()

    def month_to_number(m: str) -> int:
        """Convert month name/abbreviation to number."""
        m = m.strip().lower()
        for i in range(1, 13):
            if m == calendar.month_abbr[i].lower() or m == calendar.month_name[i].lower():
                return i
        if m.isdigit() and 1 <= int(m) <= 12:
            return int(m)
        raise ValueError(f"Invalid month: {m}")

    # Handle 'all' case
    if isinstance(months, str) and months.strip().lower() == 'all':
        month_nums = list(range(1, 13))
    # Handle comma-separated string
    elif isinstance(months, str) and "," in months:
        parts = [m.strip() for m in months.split(",")]
        month_nums = [month_to_number(m) for m in parts]
    # Handle ranges like "mar-may" or "3-6"
    elif isinstance(months, str) and "-" in months:
        start, end = months.split("-")
        start_num = month_to_number(start)
        end_num = month_to_number(end)
        if end_num < start_num:
            raise ValueError("End month must not be earlier than start month.")
        month_nums = list(range(start_num, end_num + 1))
    # Handle a list of strings or ints
    elif isinstance(months, list):
        month_nums = [month_to_number(m) if isinstance(m, str) else int(m) for m in months]
    # Handle a single string
    else:
        month_nums = [month_to_number(months)]

    # Build the filenames
    filenames = [f"{year}{month:02d}_{basename}.{extension}" for month in month_nums]
    return filenames


# =============================================================================
# DATA AGGREGATION AND PROCESSING
# =============================================================================

def aggregate_columns_by_prefix(csv_file_path: str) -> pd.DataFrame:
    """
    Aggregate columns in a CSV file by their 3-character prefix.

    This function groups all columns with the same 3-character prefix (e.g., 
    'HWB001', 'HWB002' -> 'HWB') and sums their values.

    Parameters
    ----------
    csv_file_path : str
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns aggregated by 3-character prefix, plus DATE column.
        
    Examples
    --------
    >>> df = aggregate_columns_by_prefix('202401_gen_md.csv')
    >>> print(df.columns[:5])
    ['DATE', 'ARE', 'AVL', 'BEN', 'BRY']
    """
    # Read the input CSV
    df = pd.read_csv(csv_file_path)

    # Separate the DATE column
    date_col = df[['DATE']].copy()

    # Select only the data columns
    data_cols = df.columns[1:]

    # Create a dictionary to group columns by 3-character prefix
    prefix_groups = {}
    for col in data_cols:
        prefix = col[:3]
        prefix_groups.setdefault(prefix, []).append(col)

    # Aggregate columns by summing those with the same prefix
    aggregated_data = {
        prefix: df[cols].sum(axis=1)
        for prefix, cols in prefix_groups.items()
    }

    # Create a new DataFrame from the aggregated data
    aggregated_df = pd.DataFrame(aggregated_data)

    # Concatenate the DATE column with the aggregated data
    result_df = pd.concat([date_col, aggregated_df], axis=1)

    return result_df


def subtract_site_data(
    df1: pd.DataFrame, 
    df2: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Subtract corresponding columns of two DataFrames (df1 - df2).

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame (minuend).
    df2 : pd.DataFrame
        Second DataFrame (subtrahend).

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str]]
        - Result DataFrame with subtracted values
        - List of sites missing from df1
        - List of sites where values were clipped to prevent negative results

    Raises
    ------
    ValueError
        If either DataFrame is missing the 'DATE' column.
        
    Examples
    --------
    >>> df_delta, missing, clipped = subtract_site_data(df_export, df_import)
    >>> print(f"Net flow calculated for {df_delta.shape[1]-1} sites")
    """
    # Validate 'DATE' column
    if 'DATE' not in df1.columns or 'DATE' not in df2.columns:
        raise ValueError("Both DataFrames must contain a 'DATE' column.")

    # Store the DATE column separately
    date_col = df1[['DATE']].copy()

    # Prepare data without DATE
    data1 = df1.drop(columns='DATE')
    data2 = df2.drop(columns='DATE')

    # Lists to store site info
    missing_sites = []
    clipped_sites = []

    # Copy data1 for modification
    result_data = data1.copy()

    # Iterate over columns in data2
    for col in data2.columns:
        if col in data1.columns:
            subtracted = data1[col] - data2[col]
            clipped = subtracted.clip(lower=-99999999)
            if not subtracted.equals(clipped):
                clipped_sites.append(col)
            result_data[col] = clipped
        else:
            missing_sites.append(col)

    # Combine back the DATE column
    result_df = pd.concat([date_col, result_data], axis=1).round(3)

    # Report issues
    if missing_sites:
        print(f"  Warning: Columns in df2 not found in df1 (skipped): {missing_sites}")
    if clipped_sites:
        print(f"  Warning: Subtracted values clipped for sites: {clipped_sites}")

    return result_df, missing_sites, clipped_sites


def add_matching_columns_with_timestamp(
    df1: pd.DataFrame, 
    df2: pd.DataFrame
) -> pd.DataFrame:
    """
    Add the values of corresponding named columns in two DataFrames.

    The first column is assumed to be timestamps and is preserved.
    Matching columns are summed, unmatched columns are retained.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame with timestamp column.
    df2 : pd.DataFrame
        Second DataFrame with timestamp column.

    Returns
    -------
    pd.DataFrame
        New DataFrame with summed matched columns and all unmatched columns.
        
    Examples
    --------
    >>> df_demand = add_matching_columns_with_timestamp(df_gen, df_delta)
    >>> print(f"Combined {df_demand.shape[1]-1} sites")
    """
    # Extract timestamp columns (assumed to be first column)
    timestamp1 = df1.iloc[:, 0]
    timestamp2 = df2.iloc[:, 0]

    if not timestamp1.equals(timestamp2):
        print("  Warning: Timestamp columns do not match between DataFrames.")

    # Drop the timestamp columns before processing
    df1_data = df1.iloc[:, 1:]
    df2_data = df2.iloc[:, 1:]

    # Find matched and unmatched columns
    matched_cols = df1_data.columns.intersection(df2_data.columns)
    unmatched_df1 = df1_data.columns.difference(df2_data.columns)
    unmatched_df2 = df2_data.columns.difference(df1_data.columns)

    print(f"  Matched columns: {len(matched_cols)}")
    if len(unmatched_df1) > 0:
        print(f"  Unmatched columns only in df1: {len(unmatched_df1)}")
    if len(unmatched_df2) > 0:
        print(f"  Unmatched columns only in df2: {len(unmatched_df2)}")

    # Add matched columns
    df_sum = df1_data[matched_cols].add(df2_data[matched_cols], fill_value=0)

    # Combine with unmatched columns
    result = pd.concat([
        df_sum,
        df1_data[unmatched_df1],
        df2_data[unmatched_df2]
    ], axis=1)

    # Re-insert the timestamp column at the beginning
    result.insert(0, df1.columns[0], timestamp1)

    # Sort non-timestamp columns alphabetically
    non_ts_cols = sorted(result.columns[1:])
    result = result[[result.columns[0]] + non_ts_cols]

    return result


def replace_negatives_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all negative values in a DataFrame with zero.

    The first column (assumed to be timestamps) is preserved unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with negative values set to 0 (except first column).
        
    Examples
    --------
    >>> df_clean = replace_negatives_with_zero(df_demand)
    >>> print((df_clean < 0).sum().sum())  # Should be 0
    0
    """
    df_clean = df.copy()

    # Operate only on numeric columns after the first
    numeric_cols = df_clean.columns[1:]

    # Replace negatives with 0
    df_clean[numeric_cols] = df_clean[numeric_cols].map(
        lambda x: max(x, 0) if pd.notnull(x) else x
    )

    return df_clean


# =============================================================================
# BUS AND CONNECTIVITY PROCESSING
# =============================================================================

def extract_matching_buses(df: pd.DataFrame, bus_list: List[str]) -> pd.DataFrame:
    """
    Extract rows from DataFrame where 'site' column matches values in bus_list.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame with a 'site' column.
    bus_list : List[str]
        List of bus names to extract.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only matching rows.

    Raises
    ------
    KeyError
        If the DataFrame does not contain a 'site' column.
        
    Examples
    --------
    >>> df_filtered = extract_matching_buses(df_bus_master, ['HWB', 'BEN', 'CYD'])
    >>> print(df_filtered.shape[0])
    3
    """
    if 'site' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'site' column.")

    # Filter the DataFrame using vectorized boolean masking
    filtered_df = df[df['site'].isin(bus_list)].copy()

    # Reset index
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df


def check_bus_connectivity(
    df: pd.DataFrame, 
    bus_list: List[str],
    verbose: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Check if buses appear in transmission line topology (bus0 or bus1 columns).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'bus0' and 'bus1' columns representing line connections.
    bus_list : List[str]
        List of bus names to check connectivity.
    verbose : bool, optional
        If True, print detailed connectivity information.

    Returns
    -------
    Tuple[List[str], List[str]]
        - List of connected buses
        - List of disconnected buses
        
    Examples
    --------
    >>> connected, disconnected = check_bus_connectivity(df_lines, bus_list)
    >>> print(f"{len(connected)} connected, {len(disconnected)} disconnected")
    """
    # Collect all unique connected buses
    connected_set = set(df['bus0']).union(set(df['bus1']))

    # Check each bus for presence
    connected_buses = [bus for bus in bus_list if bus in connected_set]
    disconnected_buses = [bus for bus in bus_list if bus not in connected_set]

    if verbose:
        print(f"  Connected buses: {len(connected_buses)}")

    if disconnected_buses:
        print(f"  Warning: Disconnected buses: {disconnected_buses}")

    return connected_buses, disconnected_buses


# =============================================================================
# MONTHLY DATA PROCESSING
# =============================================================================

def aggregate_monthly_flow_data(dirpath: str) -> List[pd.DataFrame]:
    """
    Aggregate monthly energy flow data from all files in a directory.

    Parameters
    ----------
    dirpath : str
        Directory path containing monthly flow data files.

    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames with aggregated data for each month (12 months).
        
    Examples
    --------
    >>> df_list = aggregate_monthly_flow_data('/data/import/')
    >>> print(f"Loaded {len(df_list)} months of data")
    Loaded 12 months of data
    """
    file_list = list_filenames_in_directory(dirpath, False)

    df_list = []
    for file in file_list:
        file_path = os.path.join(dirpath, file)
        df_f = aggregate_columns_by_prefix(file_path)
        df_list.append(df_f)

    return df_list


def process_generation_anomalies(df_g: pd.DataFrame) -> pd.DataFrame:
    """
    Handle known generation data anomalies.

    Specifically:
    - Merge HWB and WPG sites (WPG data added to HWB, then WPG dropped)
    - Merge CML and CYD sites when CML exists (CML data added to CYD, then CML dropped)

    Parameters
    ----------
    df_g : pd.DataFrame
        Generation DataFrame with potential anomalies.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with anomalies resolved.
        
    Notes
    -----
    These anomalies are based on known site consolidations in the NZ electricity
    market data.
        
    Examples
    --------
    >>> df_clean = process_generation_anomalies(df_gen)
    Merged WPG data into HWB and dropped WPG column.
    """
    df_clean = df_g.copy()

    # Handle HWB/WPG merger
    if 'HWB' in df_clean.columns and 'WPG' in df_clean.columns:
        df_clean['HWB'] = df_clean['HWB'] + df_clean['WPG']
        df_clean = df_clean.drop(columns=['WPG'])
        print("  Merged WPG data into HWB and dropped WPG column.")

    # Handle CML/CYD merger
    if 'CML' in df_clean.columns:
        if 'CYD' in df_clean.columns:
            df_clean['CYD'] = df_clean['CYD'] + df_clean['CML']
            df_clean = df_clean.drop(columns=['CML'])
            print("  Merged CML data into CYD and dropped CML column.")
        else:
            print("  Warning: CML found but CYD column missing. CML data retained.")

    return df_clean



def create_master_bus_file(bus_dir: str, output_path: str) -> int:
    """
    Create a master bus data file containing all unique buses from monthly files.
    
    Reads all monthly bus data CSV files, extracts unique buses (by 'name' or 'site'),
    and saves them to a single master file with the same structure as monthly files.
    
    Parameters
    ----------
    bus_dir : str
        Directory containing monthly bus data CSV files
    output_path : str
        Full path where the master bus file should be saved
        
    Returns
    -------
    int
        Number of unique buses in the master file
        
    Raises
    ------
    FileNotFoundError
        If no bus data files are found in the directory
    ValueError
        If bus data files have inconsistent column structures
        
    Examples
    --------
    >>> num_buses = create_master_bus_file('/data/bus/', '/data/static/busses.csv')
    Created master bus file with 167 unique buses
    167
    
    Notes
    -----
    - Buses are identified as unique by their 'name' column (or 'site' if 'name' not present)
    - All bus attributes (coordinates, voltage, etc.) are preserved from first occurrence
    - Output file has same column structure as monthly bus files
    - Buses are sorted alphabetically by name in the output
    """
    bus_files = [f for f in os.listdir(bus_dir) if f.endswith('_bus_data.csv')]
    
    if not bus_files:
        raise FileNotFoundError(f"No bus data files found in {bus_dir}")
    
    print(f"\nCreating master bus file from {len(bus_files)} monthly files...")
    
    # List to collect all bus dataframes
    all_buses = []
    
    # Read each monthly file
    for bus_file in sorted(bus_files):
        file_path = os.path.join(bus_dir, bus_file)
        df = pd.read_csv(file_path)
        all_buses.append(df)
    
    # Concatenate all dataframes
    df_all = pd.concat(all_buses, ignore_index=True)
    
    # Determine identifier column (prefer 'name', fallback to 'site')
    id_col = 'name' if 'name' in df_all.columns else 'site'
    
    # Remove duplicates (keep first occurrence)
    df_unique = df_all.drop_duplicates(subset=[id_col], keep='first')
    
    # Sort by identifier
    df_unique = df_unique.sort_values(by=id_col).reset_index(drop=True)
    
    # Save master file
    df_unique.to_csv(output_path, index=False)
    
    print(f"✓ Created master bus file: {os.path.basename(output_path)}")
    print(f"  Total unique buses: {len(df_unique)}")
    print(f"  Saved to: {output_path}")
    
    return len(df_unique)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(argv=None):
    """
    Main execution function for load profile creation.
    
    Processes 12 months of grid data to create bus, generation, and demand profiles
    for PyPSA capital expansion analysis.
    """
    # Load configuration
    args = parse_args(argv)
    
    config_path = Path(args.config).resolve() if args.config else default_config_path()
    try:
        config = load_config(str(config_path))
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Fatal error loading configuration: {e}")
        return
    # Workspace root (data/output location): require --root for reviewer-safe runs
    workspace_root = Path(args.root).resolve() if args.root else None
    if workspace_root is None:
        print("Fatal error: no workspace root provided. Pass --root <WORKSPACE>.")
        return

    # Logging (under workspace_root / paths.logs, default "logs")
    logs_rel = config.get("paths", {}).get("logs", "logs")
    log_dir = Path(resolve_path(str(logs_rel), workspace_root))
    log_file = setup_logging(log_dir)
    
    logger.info("=" * 80)
    logger.info("NZA LOAD PROFILE CREATOR")
    logger.info("=" * 80)
    logger.info(f"Start time:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config:         {config_path}")
    logger.info(f"Workspace root: {workspace_root}")
    logger.info(f"Log file:       {log_file}")
    logger.info("")
    if workspace_root is None:
        print("Fatal error: no workspace root provided. Pass --root <WORKSPACE>, or set paths.root in the YAML.")
        return
    
    # Extract paths from configuration (relative paths resolve under workspace_root)
    paths = {
        'export': resolve_path(config['paths']['dirpath_export'], workspace_root),
        'import': resolve_path(config['paths']['dirpath_import'], workspace_root),
        'gen':    resolve_path(config['paths']['dirpath_gen'], workspace_root),
        'g':      resolve_path(config['paths']['dirpath_g_MWh'], workspace_root),
        'bus':    resolve_path(config['paths']['dirpath_bus'], workspace_root),
        'demand': resolve_path(config['paths']['dirpath_demand'], workspace_root),
        'd'     : resolve_path(config['paths']['dirpath_d_MWh'], workspace_root),
        'f1f2':   resolve_path(config['paths']['dirpath_f1f2'], workspace_root),
        'static': resolve_path(config['paths']['dirpath_static'], workspace_root),
    }

    # Create output directories if they don't exist
    print("\nEnsuring output directories exist...")
    for key, path in paths.items():
        if key != 'static':  # Don't create static directory
            ensure_directory_exists(path, verbose=False)

    year = config['base_year']
    print(f"\nPROCESSING YEAR: {year}")
    print("\nLOADING AND PROCESSING ENERGY FLOW DATA...")

    # Load grid export/import data (f1 & f2)
    try:
        df_f2 = aggregate_monthly_flow_data(paths['import'])  # Import data
        df_f1 = aggregate_monthly_flow_data(paths['export'])  # Export data
        print(f"Loaded {len(df_f1)} months of import/export data")
    except Exception as e:
        print(f"Error loading flow data: {e}")
        return

    # Read master bus data
    bus_file = os.path.join(paths['static'], "nodes.csv")
    if not os.path.exists(bus_file):
        print(f"Error: Bus data file not found: {bus_file}")
        return
    
    df_bus_master = pd.read_csv(bus_file)
    print(f"\nLoaded {len(df_bus_master)} buses from NODES master file")
    print(f"NODES file : {bus_file}")

    # Generate file lists
    bus_file_list = generate_monthly_filenames(year, "all", "bus_data", "csv")
    gen_file_list = list_filenames_in_directory(paths['gen'], False)
    delta_file_list = generate_monthly_filenames(year, "all", "delta_f1f2_md", "csv")
    demand_file_list = generate_monthly_filenames(year, "all", "demand_md", "csv")

    # Load transmission line data for connectivity checking
    lines_file = os.path.join(paths['static'], "lines_data.csv")
    if not os.path.exists(lines_file):
        print(f"Error: Lines data file not found: {lines_file}")
        return
    
    df_lines = pd.read_csv(lines_file)
    print(f"Loaded {len(df_lines)} transmission lines")

    print(f"\nProcessing {len(df_f1)} months of data...")
    print("=" * 80)

    # Process each month
    for i in range(12):
        month_name = calendar.month_name[i + 1]
        print(f"\n{month_name} (Month {i + 1}/12)")
        print("-" * 40)

        # Get participating sites
        f2_sites = set(df_f2[i].columns[1:])
        f1_sites = set(df_f1[i].columns[1:])
        f1f2_sites = sorted(list(f2_sites.union(f1_sites)))

        print(f"  Found {len(f1f2_sites)} participating sites")

        # PROCESS BUSES
        df_filtered = extract_matching_buses(df_bus_master, f1f2_sites)
        connected_buses, disconnected_buses = check_bus_connectivity(
            df_lines, f1f2_sites, verbose=False
        )

        # Save bus data
        bus_output_path = os.path.join(paths['bus'], bus_file_list[i])
        df_filtered.to_csv(bus_output_path, index=False)
        print(f"  Saved bus data: {bus_file_list[i]}")

        # PROCESS GENERATION
        gen_file_path = os.path.join(paths['gen'], gen_file_list[i])
        df_g = aggregate_columns_by_prefix(gen_file_path)
        df_g = process_generation_anomalies(df_g)

        # Save generation data
        gen_output_path = os.path.join(paths['g'], gen_file_list[i])
        df_g.round(4).to_csv(gen_output_path, index=False)
        print(f"  Saved generation: {gen_file_list[i]}")

        # PROCESS DEMAND
        # Compute net flow (export - import)
        df_delta, missing_sites, clipped_sites = subtract_site_data(
            df_f1[i], df_f2[i]
        )

        # Save delta data
        delta_output_path = os.path.join(paths['f1f2'], delta_file_list[i])
        df_delta.round(4).to_csv(delta_output_path, index=False)
        print(f"  Saved delta flow: {delta_file_list[i]}")

        # Combine generation and net flow to get demand
        df_demand = add_matching_columns_with_timestamp(df_g, df_delta)
        df_demand = replace_negatives_with_zero(df_demand)

        # Save demand data
        demand_output_path = os.path.join(paths['d'], demand_file_list[i])
        df_demand.to_csv(demand_output_path, index=False)
        print(f"  Saved demand: {demand_file_list[i]}")


    # Create master bus file from all monthly bus files
    try:
        master_bus_path = os.path.join(paths['static'], "busses.csv")
        num_buses = create_master_bus_file(paths['bus'], master_bus_path)
    except Exception as e:
        print(f"\n⚠ Warning: Could not create master bus file: {e}")

    print("\n" + "=" * 80)
    print("DATA PROCESSING COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {len(bus_file_list)} monthly bus data files")
    print(f"  - 1 master bus data file (busses.csv)")
    print(f"  - {len(gen_file_list)} generation files")
    print(f"  - {len(delta_file_list)} delta flow files")
    print(f"  - {len(demand_file_list)} demand files")
    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)  # Explicit success
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Explicit failure
    