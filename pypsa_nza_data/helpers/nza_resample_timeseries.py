#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nza_resample_timeseries.py

DESCRIPTION
-----------
Resamples electricity time series data from base 30-minute resolution to 
user-specified temporal resolution (e.g., hourly, daily, weekly).

This script processes CSV files containing electricity market data with datetime
indices and multiple data columns, providing flexible resampling with various
aggregation methods.

FEATURES
--------
- Resample to any pandas-compatible frequency (1H, 2H, 1D, 1W, etc.)
- Multiple aggregation methods: mean, sum, min, max, median, first, last
- Batch processing of multiple files in a directory
- Preserves column names and data structure
- Automatic output directory creation
- Comprehensive error handling and validation

INPUT FORMAT
------------
CSV files with:
- First column: 'DATE' containing datetime values
- Remaining columns: Numeric data (power, energy, etc.)
- Index: Sequential (not datetime-indexed in file)

Example input (30-minute data):
    DATE,ARE,BEN,CYD,HWB,MAT
    2024-01-01 00:00:00,245.3,521.8,1243.2,2156.7,412.5
    2024-01-01 00:30:00,243.1,519.3,1241.8,2153.4,410.2
    2024-01-01 01:00:00,241.5,517.2,1240.1,2151.8,408.9
    ...

OUTPUT FORMAT
-------------
CSV files with resampled data:
- Same column structure as input
- Reduced number of rows based on new frequency
- Aggregated values based on chosen method

Example output (1-hour data with mean):
    DATE,ARE,BEN,CYD,HWB,MAT
    2024-01-01 00:00:00,244.2,520.55,1242.5,2155.05,411.35
    2024-01-01 01:00:00,239.8,515.1,1238.3,2149.2,407.1
    ...

USAGE EXAMPLES
--------------
# Run from Spyder (just press F5 - uses defaults)
python nza_resample_timeseries.py

# Run from command line with config file
python nza_resample_timeseries.py --config config/nza_resample_config.yaml

# Use specific profile
python nza_resample_timeseries.py --profile gen_power_hourly

# Resample to hourly using mean (default)
python nza_resample_timeseries.py -i data/cx/2024/demand/ -f 1H

# Resample to hourly using sum
python nza_resample_timeseries.py -i data/cx/2024/demand/ -f 1H -m sum

# Resample to daily using mean, specify output directory
python nza_resample_timeseries.py -i data/cx/2024/demand/ -f 1D -o data/cx/2024/demand_daily/

# Resample to 2-hour intervals
python nza_resample_timeseries.py -i data/cx/2024/gen/ -f 2H -m mean

# Resample to weekly maximum values
python nza_resample_timeseries.py -i data/cx/2024/demand/ -f 1W -m max

RESAMPLING FREQUENCIES
----------------------
Common pandas frequency strings:
- '1H'  : 1 hour
- '2H'  : 2 hours
- '6H'  : 6 hours
- '1D'  : 1 day
- '1W'  : 1 week
- '1M'  : 1 month (calendar month end)
- '15T' : 15 minutes
- '5T'  : 5 minutes

AGGREGATION METHODS
-------------------
- 'mean'   : Average values (default for power data)
- 'sum'    : Sum values (use for energy data)
- 'min'    : Minimum value
- 'max'    : Maximum value
- 'median' : Median value
- 'first'  : First value in period
- 'last'   : Last value in period

DEPENDENCIES
------------
    - pandas >= 1.3.0

AUTHOR
------
    Philippe Bruneau

CREATED
-------
    2025-12-07

VERSION
-------
    2.0.0 - Added config file support and Spyder compatibility
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Optional
import re

import pandas as pd
import yaml

# Setup path to find nza_root
ROOT_PROJECT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_PROJECT))

print(f"Looking for nza_root in: {ROOT_PROJECT}")

from nza_root import ROOT_DIR

# =============================================================================
# CONSTANTS
# =============================================================================

# Valid aggregation methods
VALID_METHODS = ['mean', 'sum', 'min', 'max', 'median', 'first', 'last']

# Default resampling parameters
DEFAULT_FREQUENCY = '1H'
DEFAULT_METHOD = 'mean'

# Expected datetime column name
DATE_COLUMN = 'DATE'

# Default configuration file path (for Spyder/IDE use)
DEFAULT_CONFIG_FILE = os.path.join(ROOT_DIR, 'config', 'nza_resample_config.yaml')

# Default profile (set to None to use 'default' from config, or specify profile name(s))
# Options: 
#   None                                    - Use 'default' from config
#   "gen_power_hourly"                      - Single profile
#   ["gen_power_hourly", "demand_power_hourly"]  - Multiple profiles
DEFAULT_PROFILE = ["gen_power_hourly", "demand_power_hourly"]  #DEFAULT_PROFILE = None  # Change this to your most common profile(s)


# =============================================================================
# CONFIGURATION FILE HANDLING
# =============================================================================

def load_config_file(config_path: str, profile: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    profile : str, optional
        Named profile to use from config. If None, uses 'default'.
        
    Returns
    -------
    dict
        Configuration dictionary.
        
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    KeyError
        If requested profile doesn't exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if profile:
        # Use named profile
        if 'profiles' not in config or profile not in config['profiles']:
            available = list(config.get('profiles', {}).keys())
            raise KeyError(
                f"Profile '{profile}' not found in config file. "
                f"Available profiles: {available}"
            )
        return config['profiles'][profile]
    else:
        # Use default config
        if 'default' not in config:
            raise KeyError("No 'default' configuration found in config file")
        return config['default']


# =============================================================================
# UTILITY FUNCTIONS
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
    """
    if not directory_path:
        print("Error: Directory path cannot be empty or None.")
        return False

    directory_path = os.path.normpath(directory_path)

    try:
        if os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                if verbose:
                    print(f"  Output directory exists: {directory_path}")
                return True
            else:
                print(f"Error: Path exists but is not a directory: {directory_path}")
                return False

        os.makedirs(directory_path, exist_ok=True)

        if verbose:
            print(f"  Created output directory: {directory_path}")
        return True

    except PermissionError:
        print(f"Error: Permission denied. Cannot create directory: {directory_path}")
        return False
    except OSError as e:
        print(f"Error: Could not create directory '{directory_path}': {e}")
        return False


def list_csv_files(directory_path: str) -> List[str]:
    """
    List all CSV files in a directory.

    Parameters
    ----------
    directory_path : str
        Path to the directory.

    Returns
    -------
    List[str]
        List of CSV filenames (not full paths).

    Raises
    ------
    ValueError
        If directory doesn't exist or is not a directory.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(
            f"The specified path does not exist or is not a directory: "
            f"{directory_path}"
        )

    files = []
    for entry in os.listdir(directory_path):
        if entry.lower().endswith('.csv'):
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path):
                files.append(entry)

    return sorted(files)


def validate_frequency_string(freq_str: str) -> bool:
    """
    Validate that frequency string is pandas-compatible.

    Parameters
    ----------
    freq_str : str
        Frequency string to validate (e.g., '1H', '2D').

    Returns
    -------
    bool
        True if valid, False otherwise.

    Examples
    --------
    >>> validate_frequency_string('1H')
    True
    >>> validate_frequency_string('30T')
    True
    >>> validate_frequency_string('invalid')
    False
    """
    # Common valid frequency patterns
    pattern = r'^\d+[TSMHDWMY]$|^[TSMHDWMY]$'
    
    if not re.match(pattern, freq_str, re.IGNORECASE):
        return False
    
    # Try to create a date range with this frequency
    try:
        pd.date_range('2024-01-01', periods=2, freq=freq_str)
        return True
    except ValueError:
        return False


def generate_output_filename(input_filename: str, frequency: str) -> str:
    """
    Generate output filename by adding frequency suffix.

    Parameters
    ----------
    input_filename : str
        Original input filename.
    frequency : str
        Resampling frequency string.

    Returns
    -------
    str
        Output filename with frequency suffix.

    Examples
    --------
    >>> generate_output_filename('202401_gen_md.csv', '1H')
    '202401_gen_md_1H.csv'
    
    >>> generate_output_filename('demand.csv', '1D')
    'demand_1D.csv'
    """
    # Split filename and extension
    name, ext = os.path.splitext(input_filename)
    
    # Add frequency suffix
    output_filename = f"{name}_{frequency}{ext}"
    
    return output_filename


# =============================================================================
# RESAMPLING FUNCTIONS
# =============================================================================

def resample_timeseries(
    df: pd.DataFrame,
    frequency: str,
    method: str = 'mean',
    date_column: str = DATE_COLUMN
) -> pd.DataFrame:
    """
    Resample time series data to specified frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime column and data columns.
    frequency : str
        Pandas frequency string (e.g., '1H', '1D', '1W').
    method : str, optional
        Aggregation method. Default is 'mean'.
        Options: 'mean', 'sum', 'min', 'max', 'median', 'first', 'last'
    date_column : str, optional
        Name of the datetime column. Default is 'DATE'.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with same column structure.

    Raises
    ------
    ValueError
        If date_column not found or method is invalid.
    KeyError
        If required columns are missing.

    Examples
    --------
    >>> df_hourly = resample_timeseries(df_30min, '1H', method='mean')
    >>> print(df_hourly.shape)
    (744, 25)  # Half the rows for hourly vs 30-min
    """
    # Validate inputs
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    
    if method not in VALID_METHODS:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: {VALID_METHODS}"
        )
    
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Convert DATE column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Set datetime as index for resampling
    df_copy.set_index(date_column, inplace=True)
    
    # Perform resampling with specified method
    if method == 'mean':
        df_resampled = df_copy.resample(frequency).mean()
    elif method == 'sum':
        df_resampled = df_copy.resample(frequency).sum()
    elif method == 'min':
        df_resampled = df_copy.resample(frequency).min()
    elif method == 'max':
        df_resampled = df_copy.resample(frequency).max()
    elif method == 'median':
        df_resampled = df_copy.resample(frequency).median()
    elif method == 'first':
        df_resampled = df_copy.resample(frequency).first()
    elif method == 'last':
        df_resampled = df_copy.resample(frequency).last()
    
    # Reset index to make DATE a column again
    df_resampled.reset_index(inplace=True)
    
    # Round to reasonable precision
    numeric_cols = df_resampled.select_dtypes(include=['float64', 'float32']).columns
    df_resampled[numeric_cols] = df_resampled[numeric_cols].round(4)
    
    return df_resampled


def process_file(
    input_path: str,
    output_path: str,
    frequency: str,
    method: str,
    verbose: bool = True
) -> bool:
    """
    Process a single CSV file: read, resample, and save.

    Parameters
    ----------
    input_path : str
        Full path to input CSV file.
    output_path : str
        Full path to output CSV file.
    frequency : str
        Resampling frequency.
    method : str
        Aggregation method.
    verbose : bool, optional
        Print progress messages. Default is True.

    Returns
    -------
    bool
        True if successful, False if error occurred.
    """
    try:
        # Read input file
        if verbose:
            print(f"  Reading: {os.path.basename(input_path)}")
        
        df = pd.read_csv(input_path)
        
        # Validate structure
        if DATE_COLUMN not in df.columns:
            print(f"  Error: '{DATE_COLUMN}' column not found in {input_path}")
            return False
        
        if df.empty:
            print(f"  Warning: Empty file {input_path}, skipping")
            return False
        
        original_rows = len(df)
        original_cols = len(df.columns)
        
        # Resample
        if verbose:
            print(f"    Rows: {original_rows}, Columns: {original_cols}")
            print(f"    Resampling to {frequency} using {method}...")
        
        df_resampled = resample_timeseries(df, frequency, method)
        
        new_rows = len(df_resampled)
        
        # Save output
        df_resampled.to_csv(output_path, index=False)
        
        if verbose:
            reduction = (1 - new_rows/original_rows) * 100
            print(f"    Output rows: {new_rows} ({reduction:.1f}% reduction)")
            print(f"  Saved: {os.path.basename(output_path)}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {input_path}: {e}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    
    Processes command-line arguments and resamples all CSV files in the 
    specified input directory.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Resample electricity time series data to different temporal resolutions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from Spyder (uses defaults from script)
  %(prog)s

  # Use config file with default profile
  %(prog)s --config config/nza_resample_config.yaml

  # Use single profile from config
  %(prog)s --profile gen_power_hourly

  # Use MULTIPLE profiles (runs in sequence)
  %(prog)s --profile gen_power_hourly demand_power_hourly

  # Resample generation AND demand with one command
  %(prog)s --profile gen_power_hourly demand_power_hourly export_hourly

  # Traditional mode (no config, single operation)
  %(prog)s -i data/cx/2024/demand/ -f 1H

  # Resample to hourly using sum
  %(prog)s -i data/cx/2024/demand/ -f 1H -m sum

  # Resample to daily, specify output directory
  %(prog)s -i data/cx/2024/demand/ -f 1D -o data/cx/2024/demand_daily/

Note: Multiple profiles process sequentially with summary at end.
      If no config or input directory specified, uses DEFAULT_CONFIG_FILE
      and DEFAULT_PROFILE from script constants.

Frequency strings:
  1H  = 1 hour          2H  = 2 hours         6H  = 6 hours
  1D  = 1 day           1W  = 1 week          1M  = 1 month
  15T = 15 minutes      30T = 30 minutes

Aggregation methods:
  mean   - Average values (default, good for power data)
  sum    - Sum values (good for energy data)
  min    - Minimum value in period
  max    - Maximum value in period
  median - Median value in period
  first  - First value in period
  last   - Last value in period
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        required=False,  # Not required if using config
        help='Input directory containing CSV files to resample'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default=None,
        help='Output directory (default: same as input directory)'
    )
    
    parser.add_argument(
        '-c', '--config',
        default=None,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '-p', '--profile',
        nargs='*',  # Accept multiple profiles
        default=None,
        help='Profile name(s) from config file. Can specify multiple: --profile gen_power_hourly demand_power_hourly'
    )
    
    parser.add_argument(
        '-f', '--frequency',
        default=DEFAULT_FREQUENCY,
        help=f'Resampling frequency (default: {DEFAULT_FREQUENCY}). '
             'Examples: 1H, 2H, 1D, 1W'
    )
    
    parser.add_argument(
        '-m', '--method',
        default=DEFAULT_METHOD,
        choices=VALID_METHODS,
        help=f'Aggregation method (default: {DEFAULT_METHOD})'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Determine config file to use
    config_file = args.config if args.config else (DEFAULT_CONFIG_FILE if not args.input_dir else None)
    
    # Determine profiles to process
    # Priority: command line args > DEFAULT_PROFILE > None
    if args.profile and len(args.profile) > 0:
        profiles_to_process = args.profile
    elif DEFAULT_PROFILE:
        profiles_to_process = [DEFAULT_PROFILE] if isinstance(DEFAULT_PROFILE, str) else DEFAULT_PROFILE
    else:
        profiles_to_process = [None]  # Will use 'default' from config
    
    # Print header
    if verbose:
        print("=" * 80)
        print("NZA TIME SERIES RESAMPLER")
        print("=" * 80)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Track results for all profiles
    all_results = {}
    
    # Process each profile
    for profile_idx, profile in enumerate(profiles_to_process, 1):
        if verbose and len(profiles_to_process) > 1:
            print("\n" + "=" * 80)
            print(f"PROCESSING PROFILE {profile_idx}/{len(profiles_to_process)}: {profile if profile else 'default'}")
            print("=" * 80)
        
        # Reset args for each profile (to avoid carrying over from previous profile)
        args_copy = argparse.Namespace(**vars(args))
        args_copy.input_dir = None
        args_copy.output_dir = None
        args_copy.frequency = DEFAULT_FREQUENCY
        args_copy.method = DEFAULT_METHOD
        
        # Handle configuration file if provided or using defaults
        if config_file:
            try:
                config = load_config_file(config_file, profile)
                
                # Override with config values (command-line args take precedence)
                if not args.input_dir:
                    # Get input_dir from config and make it absolute with ROOT_DIR
                    config_input = config.get('input_dir')
                    if config_input:
                        # If relative path, prepend ROOT_DIR
                        if not os.path.isabs(config_input):
                            args_copy.input_dir = os.path.join(ROOT_DIR, config_input)
                        else:
                            args_copy.input_dir = config_input
                else:
                    args_copy.input_dir = args.input_dir
                            
                if not args.output_dir:
                    # Get output_dir from config and make it absolute with ROOT_DIR
                    config_output = config.get('output_dir')
                    if config_output:
                        # If relative path, prepend ROOT_DIR
                        if not os.path.isabs(config_output):
                            args_copy.output_dir = os.path.join(ROOT_DIR, config_output)
                        else:
                            args_copy.output_dir = config_output
                else:
                    args_copy.output_dir = args.output_dir
                            
                if args.frequency == DEFAULT_FREQUENCY:  # If not specified on CLI
                    args_copy.frequency = config.get('frequency', DEFAULT_FREQUENCY)
                else:
                    args_copy.frequency = args.frequency
                    
                if args.method == DEFAULT_METHOD:  # If not specified on CLI
                    args_copy.method = config.get('method', DEFAULT_METHOD)
                else:
                    args_copy.method = args.method
                    
                if 'quiet' in config:
                    verbose = not config.get('quiet', False)
                
                if verbose:
                    profile_name = profile if profile else 'default'
                    print(f"Loaded configuration profile: {profile_name}")
                    if 'description' in config:
                        print(f"  Description: {config['description']}")
            
            except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
                print(f"Error loading configuration for profile '{profile}': {e}")
                print(f"  Tried to load: {config_file}")
                if len(profiles_to_process) > 1:
                    print(f"  Skipping this profile and continuing with remaining profiles...")
                    continue
                else:
                    if not args.input_dir:
                        print("\nPlease either:")
                        print("  1. Provide a valid config file with -c/--config")
                        print("  2. Specify input directory with -i/--input-dir")
                        print("  3. Set DEFAULT_CONFIG_FILE in the script")
                    sys.exit(1)
        
        # Validate that input_dir is provided
        if not args_copy.input_dir:
            print(f"Error: Input directory not specified for profile '{profile}'")
            if len(profiles_to_process) > 1:
                print("  Skipping this profile...")
                continue
            else:
                print("\nOptions:")
                print("  1. Use config file: --config config/nza_resample_config.yaml")
                print("  2. Specify directly: -i data/path/to/files/")
                print("  3. Set DEFAULT_CONFIG_FILE and DEFAULT_PROFILE in script")
                sys.exit(1)
        
        # If input_dir is relative, prepend ROOT_DIR
        if args_copy.input_dir and not os.path.isabs(args_copy.input_dir):
            args_copy.input_dir = os.path.join(ROOT_DIR, args_copy.input_dir)
        
        # If output_dir is relative, prepend ROOT_DIR
        if args_copy.output_dir and not os.path.isabs(args_copy.output_dir):
            args_copy.output_dir = os.path.join(ROOT_DIR, args_copy.output_dir)
        
        # Validate input directory exists
        if not os.path.isdir(args_copy.input_dir):
            print(f"Error: Input directory does not exist: {args_copy.input_dir}")
            if len(profiles_to_process) > 1:
                print("  Skipping this profile...")
                continue
            else:
                sys.exit(1)
        
        # Set output directory (default to input directory)
        output_dir = args_copy.output_dir if args_copy.output_dir else args_copy.input_dir
        
        # Validate frequency string
        if not validate_frequency_string(args_copy.frequency):
            print(f"Error: Invalid frequency string: {args_copy.frequency}")
            print("Examples of valid frequencies: 1H, 2H, 6H, 1D, 1W, 30T")
            if len(profiles_to_process) > 1:
                print("  Skipping this profile...")
                continue
            else:
                sys.exit(1)
        
        # Create output directory if needed
        if output_dir != args_copy.input_dir:
            if not ensure_directory_exists(output_dir, verbose=verbose):
                if len(profiles_to_process) > 1:
                    print("  Skipping this profile...")
                    continue
                else:
                    sys.exit(1)
        
        # Get list of CSV files
        try:
            csv_files = list_csv_files(args_copy.input_dir)
        except ValueError as e:
            print(f"Error: {e}")
            if len(profiles_to_process) > 1:
                print("  Skipping this profile...")
                continue
            else:
                sys.exit(1)
        
        if not csv_files:
            print(f"Warning: No CSV files found in {args_copy.input_dir}")
            if len(profiles_to_process) > 1:
                print("  Skipping this profile...")
                continue
            else:
                sys.exit(1)
        
        if verbose:
            print(f"Input directory:  {args_copy.input_dir}")
            print(f"Output directory: {output_dir}")
            print(f"Frequency:        {args_copy.frequency}")
            print(f"Method:           {args_copy.method}")
            print(f"Found {len(csv_files)} CSV file(s) to process\n")
            print("=" * 80)
        
        # Process each file
        success_count = 0
        error_count = 0
        
        for i, filename in enumerate(csv_files, 1):
            if verbose:
                print(f"\n[{i}/{len(csv_files)}] Processing: {filename}")
            
            input_path = os.path.join(args_copy.input_dir, filename)
            output_filename = generate_output_filename(filename, args_copy.frequency)
            output_path = os.path.join(output_dir, output_filename)
            
            success = process_file(
                input_path,
                output_path,
                args_copy.frequency,
                args_copy.method,
                verbose=verbose
            )
            
            if success:
                success_count += 1
            else:
                error_count += 1
        
        # Store results for this profile
        profile_name = profile if profile else 'default'
        all_results[profile_name] = {
            'total': len(csv_files),
            'success': success_count,
            'error': error_count
        }
        
        if verbose:
            print("\n" + "=" * 80)
            print(f"PROFILE '{profile_name}' COMPLETE")
            print("=" * 80)
            print(f"Successfully processed: {success_count} file(s)")
            if error_count > 0:
                print(f"Errors encountered:     {error_count} file(s)")
    
    # Print overall summary if multiple profiles
    if len(profiles_to_process) > 1 and verbose:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY - ALL PROFILES")
        print("=" * 80)
        
        total_files = sum(r['total'] for r in all_results.values())
        total_success = sum(r['success'] for r in all_results.values())
        total_errors = sum(r['error'] for r in all_results.values())
        
        print(f"\nProfiles processed:     {len(all_results)}")
        print(f"Total files processed:  {total_files}")
        print(f"Total successful:       {total_success}")
        print(f"Total errors:           {total_errors}")
        
        print(f"\nBy Profile:")
        for profile_name, results in all_results.items():
            success_rate = (results['success'] / results['total'] * 100) if results['total'] > 0 else 0
            print(f"  {profile_name:25s}: {results['success']:3d}/{results['total']:3d} files ({success_rate:5.1f}%)")
    
    # Print final summary
    if verbose:
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    total_errors = sum(r['error'] for r in all_results.values())
    if total_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
