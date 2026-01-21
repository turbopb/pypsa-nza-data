#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nza_convert_energy_to_power.py

Convert energy time series (MWh) to power time series (MW/MVAR) for PyPSA 
network modeling.

DESCRIPTION
-----------
This module reads consolidated energy data files (30-minute resolution) and 
converts them to average power values by dividing energy by the time interval 
(0.5 hours for 30-minute periods). The converted power data is saved to 
specified output directories with renamed files.

CONVERSION FORMULA
------------------
For 30-minute trading periods:
    Average Power (MW) = Energy (MWh) / Time Period (hours)
    Average Power (MW) = Energy (MWh) / 0.5
    Average Power (MW) = Energy (MWh) × 2.0

USAGE
-----
    # Run with default config file (for Spyder/IDE)
    python nza_convert_energy_to_power.py
    
    # Specify config file explicitly
    python nza_convert_energy_to_power.py --config config/nza_convert_config.yaml
    
    # Use specific profile
    python nza_convert_energy_to_power.py --config config/nza_convert_config.yaml --profile gen_only
    
    # For Spyder: Just press F5 to run with defaults!

DEPENDENCIES
------------
    - pandas: DataFrame operations and time series handling
    - numpy: Numerical operations (via pandas)
    - pyyaml: YAML configuration file parsing

AUTHOR
------
    Philippe Bruneau

CREATED
-------
    2025-12-04

VERSION
-------
    2.0.0 - Added YAML configuration support and file renaming

"""

import os
import sys
import time
import warnings
import yaml
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Setup path to find nza_root
ROOT_PROJECT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_PROJECT))

from nza_root import ROOT_DIR


# ============================================================================
# CONSTANTS
# ============================================================================

# Conversion factor: Energy (MWh) to Power (MW) for 30-minute periods 2x 30 mins = 1hr
ENERGY_TO_POWER_FACTOR = 2.0

# Trading period duration in hours
TRADING_PERIOD_HOURS = 0.5

# Default configuration file path (for Spyder/IDE use)
DEFAULT_CONFIG_FILE = os.path.join(ROOT_DIR, 'config', 'nza_convert_config.yaml')

# Default profile (set to None to process all types, or specify a profile name)
DEFAULT_PROFILE = None  # Options: None, "gen_only", "demand_only", "gen_demand", "all"


# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def load_config_file(config_path: str, profile: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    profile : str, optional
        Named profile to use from config. If None, processes all energy types.
        
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
    
    # Get default settings
    defaults = config.get('default', {})
    energy_types_config = config.get('energy_types', {})
    
    # Determine which energy types to process
    if profile:
        # Use named profile
        if 'profiles' not in config or profile not in config['profiles']:
            available = list(config.get('profiles', {}).keys())
            raise KeyError(
                f"Profile '{profile}' not found in config file. "
                f"Available profiles: {available}"
            )
        profile_config = config['profiles'][profile]
        energy_types_to_process = profile_config.get('energy_types', list(energy_types_config.keys()))
    else:
        # Process all defined energy types
        energy_types_to_process = list(energy_types_config.keys())
    
    # Build final configuration
    result = {
        'defaults': defaults,
        'energy_types': {},
        'types_to_process': energy_types_to_process
    }
    
    # Add configurations for types to process
    for et in energy_types_to_process:
        if et in energy_types_config:
            result['energy_types'][et] = energy_types_config[et]
    
    return result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
                    print(f"  Directory exists: {directory_path}")
                return True
            else:
                print(f"Error: Path exists but is not a directory: {directory_path}")
                return False

        os.makedirs(directory_path, exist_ok=True)
        if verbose:
            print(f"  Directory created: {directory_path}")
        return True

    except PermissionError:
        print(f"Error: Permission denied. Cannot create directory: {directory_path}")
        return False
    except OSError as e:
        print(f"Error: Could not create directory '{directory_path}'")
        print(f"Reason: {e}")
        return False


def get_csv_files_by_pattern(directory_path: str, pattern: str = "*.csv") -> List[str]:
    """
    Get list of CSV files in a directory matching a pattern.

    Parameters
    ----------
    directory_path : str
        Path to the directory to scan.
    pattern : str, optional
        Glob pattern to match files. Default is "*.csv".

    Returns
    -------
    List[str]
        List of matching filenames (not full paths).
    """
    if not os.path.isdir(directory_path):
        return []

    # Get full paths matching pattern
    full_pattern = os.path.join(directory_path, pattern)
    full_paths = glob.glob(full_pattern)
    
    # Extract just filenames
    filenames = [os.path.basename(p) for p in full_paths if os.path.isfile(p)]
    
    return sorted(filenames)


def generate_output_filename(input_filename: str, output_suffix: str) -> str:
    """
    Generate output filename from input filename.
    
    Parameters
    ----------
    input_filename : str
        Input filename (e.g., "202401_gen_md.csv")
    output_suffix : str
        Output suffix to use (e.g., "_g_MW.csv")
        
    Returns
    -------
    str
        Output filename (e.g., "202401_g_MW.csv")
        
    Examples
    --------
    >>> generate_output_filename("202401_gen_md.csv", "_g_MW.csv")
    '202401_g_MW.csv'
    
    >>> generate_output_filename("202401_demand_md.csv", "_d_MW.csv")
    '202401_d_MW.csv'
    """
    # Extract the YYYYMM prefix (first 6 characters)
    prefix = input_filename[:6]  # e.g., "202401"
    
    # Construct output filename
    output_filename = prefix + output_suffix
    
    return output_filename


def validate_energy_dataframe(
    df: pd.DataFrame,
    filename: str,
    allow_negative: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate energy DataFrame structure and content.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    filename : str
        Filename for error reporting.
    allow_negative : bool, optional
        If True, allow negative values (for reactive power).
        Default is False.

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
        - is_valid: True if DataFrame is valid
        - error_message: None if valid, otherwise description of issue
    """
    # Check if DataFrame is empty
    if df.empty:
        return False, "DataFrame is empty"

    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "Index is not DatetimeIndex"

    # Check for missing values
    if df.isna().any().any():
        num_missing = df.isna().sum().sum()
        return False, f"Contains {num_missing} missing values"

    # Check for negative values (only for active power, not reactive)
    if not allow_negative:
        if (df < 0).any().any():
            num_negative = (df < 0).sum().sum()
            return False, f"Contains {num_negative} negative values"

    # Check if all columns are numeric
    if not all(df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
        return False, "Non-numeric columns found"

    return True, None


# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

def convert_energy_to_power(
    df_energy: pd.DataFrame,
    conversion_factor: float = ENERGY_TO_POWER_FACTOR
) -> pd.DataFrame:
    """
    Convert energy values (MWh) to power values (MW).

    Parameters
    ----------
    df_energy : pd.DataFrame
        DataFrame with energy values in MWh.
    conversion_factor : float, optional
        Conversion factor from energy to power.
        Default is 2.0 for 30-minute periods (1 / 0.5 hours).

    Returns
    -------
    pd.DataFrame
        DataFrame with power values in MW.

    Notes
    -----
    For 30-minute trading periods:
        Power (MW) = Energy (MWh) / Time (hours)
        Power (MW) = Energy (MWh) / 0.5
        Power (MW) = Energy (MWh) × 2.0
    """
    df_power = df_energy.copy()

    # Multiply all numeric columns by conversion factor
    # Index (DATE) is preserved
    df_power = df_power * conversion_factor

    return df_power


def process_energy_file(
    input_filepath: str,
    output_filepath: str,
    conversion_factor: float,
    allow_negative: bool = False
) -> bool:
    """
    Process a single energy file: read, convert, validate, and save.

    Parameters
    ----------
    input_filepath : str
        Path to input energy file (MWh).
    output_filepath : str
        Path to output power file (MW or MVAR).
    conversion_factor : float
        Factor to convert energy to power.
    allow_negative : bool, optional
        Allow negative values (for reactive power). Default is False.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Read energy data
        df_energy = pd.read_csv(input_filepath, index_col='DATE', parse_dates=True)

        # Validate input data
        is_valid, error_msg = validate_energy_dataframe(
            df_energy, 
            input_filepath,
            allow_negative=allow_negative
        )
        if not is_valid:
            print(f"     Validation failed: {error_msg}")
            return False

        # Convert energy to power
        df_power = convert_energy_to_power(df_energy, conversion_factor)

        # Save power data
        df_power.to_csv(output_filepath)

        # Get file info for logging
        input_size = os.path.getsize(input_filepath) / 1024  # KB
        output_size = os.path.getsize(output_filepath) / 1024  # KB
        num_columns = len(df_power.columns)
        num_rows = len(df_power)

        print(f"     ✓ Converted: {os.path.basename(output_filepath)}")
        print(f"       Rows: {num_rows}, Columns: {num_columns}")
        print(f"       Size: {input_size:.1f} KB -> {output_size:.1f} KB")

        return True

    except FileNotFoundError:
        print(f"     ✗ Error: File not found: {input_filepath}")
        return False
    except pd.errors.EmptyDataError:
        print(f"     ✗ Error: Empty file: {input_filepath}")
        return False
    except Exception as e:
        print(f"     ✗ Error processing file: {e}")
        return False


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_energy_type(
    root_dir: str,
    energy_type: str,
    config: dict,
    conversion_factor: float
) -> Dict[str, int]:
    """
    Process all energy files for a specific energy type using config.

    Parameters
    ----------
    root_dir : str
        Root directory of the PyPSA NZA project.
    energy_type : str
        Name of energy type (key in config).
    config : dict
        Configuration dictionary for this energy type.
    conversion_factor : float
        Factor to convert energy to power.

    Returns
    -------
    Dict[str, int]
        Statistics dictionary with:
        - 'total': Total number of files found
        - 'successful': Number of files successfully processed
        - 'failed': Number of files that failed processing
    """
    stats = {'total': 0, 'successful': 0, 'failed': 0}

    # Get configuration for this energy type
    input_dir_rel = config.get('input_dir')
    output_dir_rel = config.get('output_dir')
    input_pattern = config.get('input_pattern', '*.csv')
    output_suffix = config.get('output_suffix', '.csv')
    allow_negative = config.get('allow_negative', False)
    description = config.get('description', energy_type)

    # Construct full paths
    input_dir = os.path.join(root_dir, input_dir_rel)
    output_dir = os.path.join(root_dir, output_dir_rel)

    # Determine unit
    unit = 'MVAR' if 'MVAR' in output_suffix else 'MW'

    print(f"\n{'='*80}")
    print(f"PROCESSING: {description.upper()}")
    print(f"{'='*80}")
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Pattern: {input_pattern}")
    print(f"Units:   MWh -> {unit}")
    print(f"Factor:  {conversion_factor}")
    print(f"Allow negative: {allow_negative}")

    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"  ⚠ Warning: Input directory does not exist, skipping.")
        return stats

    # Get list of CSV files matching pattern
    csv_files = get_csv_files_by_pattern(input_dir, input_pattern)
    stats['total'] = len(csv_files)

    if stats['total'] == 0:
        print(f"  ⚠ Warning: No files matching pattern '{input_pattern}' found.")
        return stats

    print(f"  Found {stats['total']} files to process")

    # Create output directory
    if not ensure_directory_exists(output_dir, verbose=True):
        print(f"  ✗ Error: Could not create output directory, skipping.")
        return stats

    # Process each file
    print(f"\n  Processing files:")
    for idx, input_filename in enumerate(csv_files, 1):
        print(f"\n  [{idx}/{stats['total']}] {input_filename}")

        # Generate output filename
        output_filename = generate_output_filename(input_filename, output_suffix)

        input_filepath = os.path.join(input_dir, input_filename)
        output_filepath = os.path.join(output_dir, output_filename)

        success = process_energy_file(
            input_filepath,
            output_filepath,
            conversion_factor,
            allow_negative=allow_negative
        )

        if success:
            stats['successful'] += 1
        else:
            stats['failed'] += 1

    # Print summary for this energy type
    print(f"\n  Summary:")
    print(f"    Total files:      {stats['total']}")
    print(f"    Successful:       {stats['successful']}")
    print(f"    Failed:           {stats['failed']}")
    if stats['total'] > 0:
        print(f"    Success rate:     {stats['successful']/stats['total']*100:.1f}%")

    return stats


def main():
    """
    Main execution function for energy to power conversion.

    Processes energy types based on configuration file.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Convert energy time series (MWh) to power (MW/MVAR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Run with default config (for Spyder/IDE)
      %(prog)s
    
      # Specify config file explicitly
      %(prog)s --config config/nza_convert_config.yaml
    
      # Use specific profile (gen only)
      %(prog)s --config config/nza_convert_config.yaml --profile gen_only
    
      # Process gen and demand
      %(prog)s --profile gen_demand
    
    Note: If no config file specified, uses: config/nza_convert_config.yaml
          If no profile specified, processes all defined energy types.
            """
    )
    
    parser.add_argument(
        '-c', '--config',
        default=None,  # Will use DEFAULT_CONFIG_FILE if not specified
        help=f'Path to YAML configuration file (default: config/nza_convert_config.yaml)'
    )
    
    parser.add_argument(
        '-p', '--profile',
        default=None,
        help='Profile name from config file (e.g., "gen_only", "all"). If not specified, uses DEFAULT_PROFILE or processes all types.'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0.0'
    )
    
    args = parser.parse_args()

    # Suppress excessive warnings
    warnings.filterwarnings('ignore')

    # Initialize program run-time
    start_time = time.time()

    print("=" * 80)
    print("NZA ENERGY TO POWER CONVERTER")
    print("=" * 80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    root_dir = ROOT_DIR
    conversion_factor = ENERGY_TO_POWER_FACTOR

    # Determine config file to use
    config_file = args.config if args.config else DEFAULT_CONFIG_FILE
    
    # Determine profile to use
    profile = args.profile if args.profile else DEFAULT_PROFILE

    # Load configuration
    try:
        print(f"Loading configuration from: {config_file}")
        if profile:
            print(f"Using profile: {profile}")
        else:
            print(f"Processing all defined energy types")
        
        config = load_config_file(config_file, profile)
        
        # Update conversion factor from config if specified
        if 'conversion_factor' in config.get('defaults', {}):
            conversion_factor = config['defaults']['conversion_factor']
        
        energy_types_config = config['energy_types']
        types_to_process = config['types_to_process']
        
        print(f"  Energy types to process: {', '.join(types_to_process)}\n")
        
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    print(f"Configuration:")
    print(f"  Root directory:     {root_dir}")
    print(f"  Conversion factor:  {conversion_factor}")

    # Verify root directory exists
    if not os.path.isdir(root_dir):
        print(f"\n✗ Error: Root directory does not exist: {root_dir}")
        sys.exit(1)

    # Process each energy type
    all_stats = {}
    for energy_type in types_to_process:
        if energy_type in energy_types_config:
            stats = process_energy_type(
                root_dir,
                energy_type,
                energy_types_config[energy_type],
                conversion_factor
            )
            all_stats[energy_type] = stats
        else:
            print(f"\n⚠ Warning: Configuration for '{energy_type}' not found, skipping.")

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_files = sum(s['total'] for s in all_stats.values())
    total_successful = sum(s['successful'] for s in all_stats.values())
    total_failed = sum(s['failed'] for s in all_stats.values())

    print(f"\nProcessing Statistics:")
    print(f"  Energy types processed:  {len(all_stats)}")
    print(f"  Total files found:       {total_files}")
    print(f"  Successfully converted:  {total_successful}")
    print(f"  Failed conversions:      {total_failed}")

    if total_files > 0:
        print(f"  Overall success rate:    {total_successful/total_files*100:.1f}%")

    print(f"\nBy Energy Type:")
    for energy_type, stats in all_stats.items():
        if stats['total'] > 0:
            success_rate = stats['successful'] / stats['total'] * 100
            print(f"  {energy_type:10s}: {stats['successful']:3d}/{stats['total']:3d} "
                  f"({success_rate:5.1f}%)")
        else:
            print(f"  {energy_type:10s}: No files found")

    # Report timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*80}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

    if total_files > 0:
        print(f"Average time per file: {elapsed_time/total_files:.2f} seconds")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        main()
        # Create success marker file for Snakemake
        log_dir = os.path.join(ROOT_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        marker_file = os.path.join(log_dir, 'conversion_complete.txt')
        with open(marker_file, 'w') as f:
            f.write('Conversion complete\n')
        sys.exit(0)  # Success
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Failure
