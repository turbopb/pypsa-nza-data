#!/usr/bin/env python
# coding: utf-8

"""
nza_create_load_profile.py

DESCRIPTION:
============
Creates and prepares bus, generation and demand load profiles for capital
expansion planning analysis using PyPSA.

WORKFLOW OVERVIEW:
==================
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

KEY FEATURES:
=============
- Processes 12 months of energy data automatically
- Handles site name consolidations and data anomalies
- Validates grid connectivity for all participating buses
- Implements robust error handling and data validation
- Generates standardized CSV outputs for PyPSA integration

DATA SOURCES:
=============
- Grid import/export files (monthly, by site)
- Generation dispatch data (monthly, by site)
- Master bus information (coordinates, metadata)
- Transmission line topology data

OUTPUT FILES:
=============
- Monthly bus data files with participating sites
- Monthly aggregated generation dispatch files
- Monthly net energy flow (delta) files
- Monthly demand profile files for capital expansion analysis

NOTES:
======
- HWB and WPG sites are consolidated (WPG data added to HWB)
- CML site data is merged with CYD when present
- All negative demand values are clipped to zero
- Disconnected buses are identified and reported

Author: Phillippe Bruneau
Created: Wed May 28 16:13:34 2025
Updated: Fri Sep 26 20:29:00 2025
"""

import numpy as np
import pandas as pd
import os
import yaml
import calendar
from typing import Dict, List, Tuple, Optional


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
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}


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

    >>> ensure_directory_exists('/existing/path')
    Directory already exists: /existing/path
    True

    >>> ensure_directory_exists('/invalid/\x00/path')
    Error: Could not create directory '/invalid/\x00/path'
    Reason: [Errno 22] Invalid argument: '/invalid/\x00/path'
    False
    """
    # Handle empty or None input
    if not directory_path:
        print("Error: Directory path cannot be empty or None.")
        return False

    # Normalize the path
    directory_path = os.path.normpath(directory_path)

    try:
        # Check if directory already exists
        if os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                if verbose:
                    print(f"Directory already exists: {directory_path}")
                return True
            else:
                print(f"Error: Path exists but is not a directory: {directory_path}")
                return False

        # Try to create the directory (including parent directories)
        os.makedirs(directory_path, exist_ok=True)

        if verbose:
            print(f"Directory created: {directory_path}")
        return True

    except PermissionError:
        print(f"Error: Permission denied. Cannot create directory: {directory_path}")
        print("Please check your permissions or try running with appropriate privileges.")
        return False


def generate_monthly_filenames(year: int, months: str, basename: str, extension: str) -> List[str]:
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
        List of generated filenames.
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


def list_filenames_in_directory(directory_path: str, include_full_path: bool = True) -> List[str]:
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
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"The specified path does not exist or is not a directory: {directory_path}")

    files = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files.append(full_path if include_full_path else entry)

    return files


def aggregate_columns_by_prefix(csv_file_path: str) -> pd.DataFrame:
    """
    Aggregate columns in a CSV file by their 3-character prefix.

    Parameters
    ----------
    csv_file_path : str
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns aggregated by prefix.
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


def subtract_site_data(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
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
        print(f"*** Columns in df2 not found in df1 (skipped): {missing_sites}")
    if clipped_sites:
        print(f"*** Subtracted values clipped for sites: {clipped_sites}")

    return result_df, missing_sites, clipped_sites


def add_matching_columns_with_timestamp(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
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
    """
    # Extract timestamp columns (assumed to be first column)
    timestamp1 = df1.iloc[:, 0]
    timestamp2 = df2.iloc[:, 0]

    if not timestamp1.equals(timestamp2):
        print("Warning: Timestamp columns do not match between DataFrames.")

    # Drop the timestamp columns before processing
    df1_data = df1.iloc[:, 1:]
    df2_data = df2.iloc[:, 1:]

    # Find matched and unmatched columns
    matched_cols = df1_data.columns.intersection(df2_data.columns)
    unmatched_df1 = df1_data.columns.difference(df2_data.columns)
    unmatched_df2 = df2_data.columns.difference(df1_data.columns)

    print(f"Matched columns: {list(matched_cols)}")
    if not unmatched_df1.empty:
        print(f"Unmatched columns only in df1: {list(unmatched_df1)}")
    if not unmatched_df2.empty:
        print(f"Unmatched columns only in df2: {list(unmatched_df2)}")

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
    """
    df_clean = df.copy()

    # Operate only on numeric columns after the first
    numeric_cols = df_clean.columns[1:]

    # Replace negatives with 0
    df_clean[numeric_cols] = df_clean[numeric_cols].map(
        lambda x: max(x, 0) if pd.notnull(x) else x
    )

    return df_clean


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
    """
    if 'site' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'site' column.")

    # Filter the DataFrame using vectorized boolean masking
    filtered_df = df[df['site'].isin(bus_list)].copy()

    # Reset index
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df


def check_bus_connectivity(df: pd.DataFrame, bus_list: List[str],
                          verbose: bool = False) -> Tuple[List[str], List[str]]:
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
    """
    # Collect all unique connected buses
    connected_set = set(df['bus0']).union(set(df['bus1']))

    # Check each bus for presence
    connected_buses = [bus for bus in bus_list if bus in connected_set]
    disconnected_buses = [bus for bus in bus_list if bus not in connected_set]

    if verbose:
        print(f"Connected buses: {connected_buses}")

    if disconnected_buses:
        print(f"Disconnected buses: {disconnected_buses}")

    return connected_buses, disconnected_buses


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
        List of DataFrames with aggregated data for each month.
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
    """
    df_clean = df_g.copy()

    # Handle HWB/WPG merger
    if 'HWB' in df_clean.columns and 'WPG' in df_clean.columns:
        df_clean['HWB'] = df_clean['HWB'] + df_clean['WPG']
        df_clean = df_clean.drop(columns=['WPG'])
        print("Merged WPG data into HWB and dropped WPG column.")

    # Handle CML/CYD merger
    if 'CML' in df_clean.columns:
        if 'CYD' in df_clean.columns:
            df_clean['CYD'] = df_clean['CYD'] + df_clean['CML']
            df_clean = df_clean.drop(columns=['CML'])
            print("Merged CML data into CYD and dropped CML column.")
        else:
            print("Warning: CML found but CYD column missing. CML data retained.")

    return df_clean


def main():
    """Main execution function."""
    # Load configuration
    config_file = '/home/pbrun/pypsa-nza/config/nza_cx_config.yaml'
    config = load_config(config_file)

    if not config:
        print("Failed to load configuration. Exiting.")
        return

    # Extract paths from configuration
    root = config['paths']['root']
    paths = {
        'export': root + config['paths']['dirpath_export'],
        'import': root + config['paths']['dirpath_import'],
        'gen': root + config['paths']['dirpath_gen'],
        'g': root + config['paths']['dirpath_g'],
        'bus': root + config['paths']['dirpath_bus'],
        'demand': root + config['paths']['dirpath_demand'],
        'f1f2': root + config['paths']['dirpath_f1f2'],
        'static': root + config['paths']['dirpath_static'],
        #'cx': root + config['paths']['dirpath_cx']
    }

    # Create directories if they dont already exist
    for p in paths.values() : ensure_directory_exists(p)

    year = config['start_year']

    print("Loading and processing energy flow data...")

    # Load grid export/import data (f1 & f2)
    df_f2 = aggregate_monthly_flow_data(paths['import'])  # Import data
    df_f1 = aggregate_monthly_flow_data(paths['export'])  # Export data

    # Read master bus data
    df_bus_master = pd.read_csv(os.path.join(paths['static'], "bus_data.csv"))

    # Generate file lists
    bus_file_list = generate_monthly_filenames(year, "all", "bus_data", "csv")
    gen_file_list = list_filenames_in_directory(paths['gen'], False)
    delta_file_list = generate_monthly_filenames(year, "all", "delta_f1f2_md", "csv")
    demand_file_list = generate_monthly_filenames(year, "all", "demand_md", "csv")

    # Load transmission line data for connectivity checking
    df_lines = pd.read_csv(os.path.join(paths['static'], "lines_data.csv"))

    print(f"Processing {len(df_f1)} months of data...")

    # Process each month
    for i in range(12):
        print(f"\nProcessing month {i + 1}...")

        # Get participating sites
        f2_sites = set(df_f2[i].columns[1:])
        f1_sites = set(df_f1[i].columns[1:])
        f1f2_sites = sorted(list(f2_sites.union(f1_sites)))

        print(f"  Found {len(f1f2_sites)} participating sites")

        # PROCESS BUSES
        df_filtered = extract_matching_buses(df_bus_master, f1f2_sites)
        connected_buses, disconnected_buses = check_bus_connectivity(df_lines, f1f2_sites)

        # Save bus data
        bus_output_path = os.path.join(paths['bus'], bus_file_list[i])
        df_filtered.to_csv(bus_output_path, index=False)

        # PROCESS GENERATION
        gen_file_path = os.path.join(paths['gen'], gen_file_list[i])
        df_g = aggregate_columns_by_prefix(gen_file_path)
        df_g = process_generation_anomalies(df_g)

        # Save generation data
        gen_output_path = os.path.join(paths['g'], gen_file_list[i])
        df_g.round(4).to_csv(gen_output_path, index=False)

        # PROCESS DEMAND
        # Compute net flow (export - import)
        df_delta, missing_sites, clipped_sites = subtract_site_data(df_f1[i], df_f2[i])

        # Save delta data
        delta_output_path = os.path.join(paths['f1f2'], delta_file_list[i])
        print(f"\ndelta output path {delta_output_path}")
        df_delta.round(4).to_csv(delta_output_path, index=False)

        # Combine generation and net flow to get demand
        df_demand = add_matching_columns_with_timestamp(df_g, df_delta)
        df_demand = replace_negatives_with_zero(df_demand)

        # Save demand data
        demand_output_path = os.path.join(paths['demand'], demand_file_list[i])
        df_demand.to_csv(demand_output_path, index=False)

        print(f"  Completed processing for month {i + 1}")
        print(f"\ngen output path: {gen_output_path}")
        print(f"\ndelta output path: {delta_output_path}")
        print(f"\ndemand output path: {demand_output_path}")

    print("\nData processing complete!")
    print("Generated files:")
    print(f"  - {len(bus_file_list)} bus data files")
    print(f"  - {len(gen_file_list)} generation files")
    print(f"  - {len(delta_file_list)} delta flow files")
    print(f"  - {len(demand_file_list)} demand files")


if __name__ == '__main__':
    main()