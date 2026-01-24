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

"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
import re
import time
import calendar
import warnings
import logging

import yaml
import pandas as pd
import numpy as np

import argparse

try:
    from importlib.resources import files as pkg_files
except Exception:  # pragma: no cover
    pkg_files = None


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    """
    Configure logging to write to both console and file.
    
    Args:
        log_dir: Directory where log files will be stored
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'nza_process_dynamic_data_{timestamp}.log'
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    root_logger.handlers.clear()
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (detailed format)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")


def print_header(title: str, char: str = '=') -> None:
    """Print a formatted header for console output."""
    width = 80
    logger.info(char * width)
    logger.info(f"{title:^{width}}")
    logger.info(char * width)


def print_section(title: str) -> None:
    """Print a section divider."""
    logger.info("")
    logger.info(f"--- {title} ---")
    logger.info("")


# ============================================================================
# CONSTANTS
# ============================================================================

# Time resolution constants
TRADING_PERIODS_PER_DAY = 48
MINUTES_PER_TRADING_PERIOD = 30

# Trading period columns to remove (daylight saving adjustment periods)
COLUMNS_TO_REMOVE = ['TP49', 'TP50']

# Default scale factor: kWh to MWh conversion
DEFAULT_SCALE_FACTOR = 1.0e-3


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================


def default_config_path() -> Path:
    """Return the packaged default config path for this processor."""
    if pkg_files is None:
        raise RuntimeError("importlib.resources.files not available; please use Python >= 3.9")
    candidate = pkg_files("pypsa_nza_data").joinpath("config/nza_process_dynamic_data.yaml")
    return Path(str(candidate))


def resolve_path(path_value: str, base_dir: Path) -> Path:
    """Resolve a path that may be absolute or relative to base_dir."""
    p = Path(str(path_value)).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()

def load_config(config_file: Path) -> Dict:
    """
    Read the YAML configuration file.

    Parameters
    ----------
    config_file : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        YAML configuration data. Returns empty dict if file not found or 
        parsing fails.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    yaml.YAMLError
        If YAML parsing fails.
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found."
        )
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config if config else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


# ============================================================================
# DIRECTORY AND FILE UTILITIES
# ============================================================================

def list_filenames_in_directory(
    directory_path: Path, 
    include_full_path: bool = True
) -> List[str]:
    """
    Return a list of filenames in the specified directory.

    Parameters
    ----------
    directory_path : Path
        Path to the directory.
    include_full_path : bool, optional
        If True, returns full file paths instead of just filenames. 
        Default is True.

    Returns
    -------
    List[str]
        A list of filenames or file paths.

    Raises
    ------
    ValueError
        If the specified path does not exist or is not a directory.
    """
    directory_path = Path(directory_path)
    
    if not directory_path.is_dir():
        raise ValueError(
            f"The specified path does not exist or is not a directory: "
            f"{directory_path}"
        )

    files = []
    for entry in directory_path.iterdir():
        if entry.is_file():
            files.append(str(entry) if include_full_path else entry.name)

    return files


def rename_files_with_prefix(
    filenames: List[str], 
    new_basename: str, 
    new_extension: Optional[str] = None
) -> List[str]:
    """
    Rename a list of files with format 'YYYYMM_*' to 'YYYYMM_<new_basename>.<ext>'.

    Parameters
    ----------
    filenames : List[str]
        List of filenames, e.g., ['202309_Generation_MD.csv'].
    new_basename : str
        The new base name to insert after the prefix (e.g., 'LoadData').
    new_extension : str, optional
        New file extension (e.g., 'txt'). Default is to preserve the original.

    Returns
    -------
    List[str]
        List of renamed files in the format 'YYYYMM_new_basename.extension'.

    Raises
    ------
    ValueError
        If any filename doesn't match the expected format.
    """
    renamed = []
    for file in filenames:
        filename = Path(file).name  # Get just the filename, not full path
        parts = filename.split('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {filename}")

        prefix = parts[0]  # YYYYMM part
        original_ext = Path(filename).suffix[1:]  # Remove the dot
        extension = new_extension if new_extension else original_ext
        new_filename = f"{prefix}_{new_basename}.{extension}"
        renamed.append(new_filename)

    return renamed


def save_dataframe_to_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save a DataFrame to a CSV file, creating directories if needed.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    filepath : Path
        The full path to the CSV file, including the filename.

    Returns
    -------
    None
    """
    filepath = Path(filepath)
    
    # Create the directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame to CSV
    df.to_csv(filepath, index=False)
    logger.info(f"  ✓ Saved: {filepath.name}")


# ============================================================================
# DATE/TIME PARSING AND EXTRACTION
# ============================================================================

def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date-time information from a filename.

    Supports multiple date formats including:
    - YYYYMMDD (e.g., 20240328)
    - YYYY-MM-DD (e.g., 2024-03-28)
    - YYMMDD (e.g., 240328)
    - MMDDYYYY (e.g., 03282024)
    - YYYYMM (e.g., 202403, assumes first day of the month)

    Parameters
    ----------
    filename : str
        Filename containing a date pattern.

    Returns
    -------
    datetime or None
        Extracted datetime object if valid date found, otherwise None.
    """
    # Define common patterns for date formats
    date_patterns = [
        (r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", "%Y-%m-%d"),  # YYYY-MM-DD or YYYYMMDD
        (r"(\d{2})[-_]?(\d{2})[-_]?(\d{4})", "%d-%m-%Y"),  # DD-MM-YYYY
        (r"(\d{4})[-_]?(\d{2})", "%Y-%m"),  # YYYYMM (assume first of the month)
        (r"(\d{2})(\d{2})(\d{4})", "%m%d%Y"),  # MMDDYYYY
        (r"(\d{2})(\d{2})(\d{2})", "%y%m%d"),  # YYMMDD (assumes 20YY)
    ]

    for pattern, date_format in date_patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                date_str = "-".join(match.groups())  # Convert to a standard format
                extracted_date = datetime.strptime(date_str, date_format)

                # Handle 2-digit years
                if "%y" in date_format and extracted_date.year < 2000:
                    extracted_date = extracted_date.replace(year=extracted_date.year + 2000)

                return extracted_date
            except ValueError:
                continue  # Ignore invalid dates and try the next pattern

    return None  # Return None if no valid date is found


def extract_detailed_date_info(date_str: str) -> Dict:
    """
    Extract comprehensive date information from a date string.

    Supports multiple date formats including:
    - ISO format: YYYY-MM-DD
    - Various delimiters: /, -, space
    - With or without time components
    - Named months: February 15, 2024

    Parameters
    ----------
    date_str : str
        Date string in various formats.

    Returns
    -------
    dict
        Dictionary containing detailed date information including:
        - Basic components: year, month, day, hour, minute, second
        - Derived values: weekday, day of year, quarter
        - Calendar info: days in month, leap year status

    Raises
    ------
    ValueError
        If date string doesn't match any recognized format.
    """
    # Define common date formats to check
    date_formats = [
        "%Y-%m-%d",  # 2024-02-15
        "%d-%m-%Y",  # 15-02-2024
        "%d/%m/%Y",  # 15/02/2024
        "%m/%d/%Y",  # 02/15/2024
        "%Y/%m/%d",  # 2024/02/15
        "%B %d, %Y",  # February 15, 2024
        "%d %B %Y",  # 15 February 2024
        "%b %d, %Y",  # Feb 15, 2024
        "%d %b %Y",  # 15 Feb 2024
        "%y-%m-%d %H:%M:%S",  # 24-01-01 08:00:00
        "%Y-%m-%d %H:%M:%S",  # 2024-01-01 08:00:00
        "%Y-%m-%d %H:%M",  # 2024-01-01 08:00
    ]

    # Try parsing the date with different formats
    date_obj = None
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            break  # Stop at the first successful match
        except ValueError:
            continue

    if date_obj is None:
        raise ValueError(
            f"Error: The provided date '{date_str}' does not match any "
            f"recognized formats."
        )

    # Extract date components
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    hour = date_obj.hour
    minute = date_obj.minute
    second = date_obj.second

    # Determine leap year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    # Number of days in the month
    days_in_month = calendar.monthrange(year, month)[1]

    # Number of hours, minutes, and seconds in the year
    total_days_in_year = 366 if is_leap else 365
    total_hours_in_year = total_days_in_year * 24

    # First and last day of the month
    first_day_of_month = datetime(year, month, 1).strftime("%Y-%m-%d (%A)")
    last_day_of_month = datetime(year, month, days_in_month).strftime(
        "%Y-%m-%d (%A)"
    )

    # Compile date information dictionary
    date_info = {
        "Original Input": date_str,
        "Parsed Date": date_obj.strftime("%Y-%m-%d %H:%M:%S"),
        "Year": year,
        "Month": month,
        "Day": day,
        "Hour": hour,
        "Minute": minute,
        "Second": second,
        "Weekday Name": date_obj.strftime("%A"),
        "Weekday Number": date_obj.weekday(),  # Monday = 0, Sunday = 6
        "ISO Weekday Number": date_obj.isoweekday(),  # Monday = 1, Sunday = 7
        "Day of Year": date_obj.timetuple().tm_yday,
        "Week Number": date_obj.strftime("%U"),
        "Quarter": (month - 1) // 3 + 1,
        "ISO Calendar (Year, Week, Weekday)": date_obj.isocalendar(),
        "Is Leap Year": is_leap,
        "Days in Month": days_in_month,
        "Total Hours in Year": total_hours_in_year,
        "First Day of Month": first_day_of_month,
        "Last Day of Month": last_day_of_month,
    }

    return date_info


# ============================================================================
# DATA CLEANING AND TRANSFORMATION
# ============================================================================

def standardize_poc_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename any column containing 'poc' (case insensitive) to 'POC'.

    Only the first match will be renamed to avoid conflicts.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with one standardized 'POC' column name.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Find columns containing 'poc' (case insensitive)
    poc_cols = [col for col in df.columns if 'poc' in col.lower()]

    if poc_cols:
        # Rename the first matching column to 'POC'
        df.rename(columns={poc_cols[0]: 'POC'}, inplace=True)
        logger.info(f"  Renamed column '{poc_cols[0]}' to 'POC'")
    
    if len(poc_cols) > 1:
        logger.warning(f"  Multiple POC-like columns found: {poc_cols}")
        logger.warning(f"  Only '{poc_cols[0]}' was renamed to 'POC'")

    return df


def aggregate_and_fill_calendar_month(
    df: pd.DataFrame, 
    year: int, 
    month: int, 
    tp_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate duplicate entries and fill missing dates for each POC.

    This function performs two key operations:
    1. Sums all trading period values for duplicate POC-date combinations
    2. Fills missing dates with zeros to create a complete calendar month

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'POC', 'Trading_Date', and 'TP1'...'TP50' columns.
    year : int
        Target year (e.g., 2025).
    month : int
        Target month (e.g., 4 for April).
    tp_columns : List[str], optional
        List of TP columns. If None, TP columns are auto-detected.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with zero-filled missing days.
        Each POC will have exactly one row per day in the month.
    """
    df = df.copy()

    # Ensure 'Trading_Date' is datetime
    df['Trading_Date'] = pd.to_datetime(df['Trading_Date'], format='%Y-%m-%d')

    # Filter to only rows matching year and month
    df = df[
        (df['Trading_Date'].dt.year == year) & 
        (df['Trading_Date'].dt.month == month)
    ]

    if df.empty:
        logger.warning(f"  No data found for {year}-{month:02d}")
        return pd.DataFrame()

    # Auto-detect TP columns if not provided
    if tp_columns is None:
        tp_columns = [col for col in df.columns if col.startswith('TP')]

    # Step 1: Aggregate by POC and Trading_Date (sum duplicates)
    df_agg = df.groupby(['POC', 'Trading_Date'])[tp_columns].sum().reset_index()

    # Step 2: Prepare full date range for the month
    days_in_month = calendar.monthrange(year, month)[1]
    full_dates = pd.date_range(
        start=f'{year}-{month:02d}-01', 
        periods=days_in_month, 
        freq='D'
    )

    all_POCs = df_agg['POC'].unique()

    final_df_list = []

    # Step 3: Fill missing dates for each POC
    for site in all_POCs:
        site_data = df_agg[df_agg['POC'] == site]

        # Create a complete date range for this POC
        full_site_df = pd.DataFrame({
            'POC': site,
            'Trading_Date': full_dates
        })

        # Merge with actual data, filling missing dates with NaN
        merged = pd.merge(
            full_site_df, 
            site_data, 
            on=['POC', 'Trading_Date'],
            how='left'
        )
        
        # Fill NaN values with zero
        merged[tp_columns] = merged[tp_columns].fillna(0)

        final_df_list.append(merged)

    # Concatenate all POCs into single DataFrame
    final_df = pd.concat(final_df_list, ignore_index=True)

    num_pocs = len(all_POCs)
    logger.info(f"  Processed {num_pocs} unique POCs with {days_in_month} days each")

    return final_df


def flatten_dataframe(df: pd.DataFrame) -> np.ndarray:
    """
    Flatten trading period columns into a single time series vector.

    Removes POC, Trading_Date, TP49, and TP50 columns, then flattens
    the remaining TP1-TP48 columns into a 1D numpy array.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing trading period columns.

    Returns
    -------
    np.ndarray
        Flattened 1D array of all trading period values.
        Length = num_days × 48 trading periods.
    """
    # Remove non-data columns
    columns_to_drop = ['POC', 'Trading_Date'] + COLUMNS_TO_REMOVE
    df_trim = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns]
    )

    # Flatten to 1D array
    vector = df_trim.to_numpy().flatten()

    return vector


def create_site_vector_dataframe(
    df: pd.DataFrame, 
    date_info: Dict
) -> pd.DataFrame:
    """
    Create a DataFrame with one column per POC containing time series vectors.

    This function orchestrates the cleaning pipeline:
    1. Aggregates duplicate POC-date entries
    2. Fills missing dates with zeros
    3. Flattens each POC into a time series vector
    4. Combines all vectors into a single DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw trading period data.
    date_info : dict
        Date information dictionary from extract_detailed_date_info().

    Returns
    -------
    pd.DataFrame
        DataFrame where each column is a POC and rows are time series values.
        Index is not yet set (will be added later as datetime).
    """
    # Extract date components
    year = date_info["Year"]
    month = date_info["Month"]
    month_name = calendar.month_name[month]
    
    logger.info(f"  Processing: {month_name} {year}")

    # Get list of all unique POCs
    unique_POCs = df['POC'].unique()
    logger.info(f"  Found {len(unique_POCs)} unique POCs")

    try:
        # Generate a clean, aggregated dataframe for the current month
        df_clean = aggregate_and_fill_calendar_month(df, year, month)

        if df_clean.empty:
            logger.warning(f"  No data after cleaning for {month_name} {year}")
            return pd.DataFrame()

        # Create a dictionary to store vectors for each site
        vector_dict = {}

        # Extract the data for each site and accumulate in the dict
        for poc in unique_POCs:
            df_site = df_clean[df_clean['POC'] == poc].copy()
            vec_energy = flatten_dataframe(df_site)
            vector_dict[poc] = vec_energy

        # Convert dictionary to DataFrame
        return pd.DataFrame.from_dict(vector_dict)

    except Exception as e:
        logger.error(f"  Error processing {month_name} {year}: {e}")
        raise


def scale_dataframe(
    df: pd.DataFrame, 
    scale_factor: float, 
    date_column: str = 'DATE'
) -> pd.DataFrame:
    """
    Scale all numeric columns by a constant factor.

    Typically used to convert kWh to MWh (scale_factor = 0.001).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with numeric columns to scale.
    scale_factor : float
        Multiplication factor for all numeric columns.
    date_column : str, optional
        Name of date column to exclude from scaling. Default is 'DATE'.

    Returns
    -------
    pd.DataFrame
        Scaled DataFrame with date column unchanged.
    """
    # Make a copy to avoid modifying the original
    df_scaled = df.copy()

    # Scale only numeric columns excluding the date column
    numeric_cols = df.columns.difference([date_column])
    df_scaled[numeric_cols] = df_scaled[numeric_cols] * scale_factor

    return df_scaled


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_energy_by_modality(
    input_dir: Path, 
    output_dir: Path, 
    file_tag: str, 
    scale_factor: float
) -> None:
    """
    Process all monthly data files for a specific energy modality.

    This is the main processing function that:
    1. Lists all input files in the directory
    2. Processes each month sequentially
    3. Applies data cleaning and transformation
    4. Scales units (kWh to MWh)
    5. Saves cleaned data to output directory

    Parameters
    ----------
    input_dir : Path
        Directory containing raw CSV files.
    output_dir : Path
        Directory to save processed CSV files.
    file_tag : str
        Tag to use in output filenames (e.g., 'generation_md').
    scale_factor : float
        Scale factor for unit conversion (typically 0.001 for kWh to MWh).

    Returns
    -------
    None
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Obtain the list of files to process
    try:
        input_file_list = list_filenames_in_directory(input_dir, False)
    except ValueError as e:
        logger.error(f"Error accessing input directory: {e}")
        return

    if not input_file_list:
        logger.warning(f"No files found in {input_dir}")
        return

    logger.info(f"Found {len(input_file_list)} files to process")
    logger.info("")

    # Create corresponding list of output files
    try:
        output_file_list = rename_files_with_prefix(
            input_file_list, 
            file_tag, 
            new_extension=None
        )
    except ValueError as e:
        logger.error(f"Error creating output filenames: {e}")
        return

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each month (each file is one month's data)
    processed_count = 0
    for index, filename in enumerate(input_file_list):
        print_section(f"File {index + 1}/{len(input_file_list)}: {filename}")

        # Full file path of the input file
        filepath = input_dir / filename

        # Extract date information from filename
        date_extracted = extract_datetime_from_filename(str(filepath))
        if date_extracted is None:
            logger.warning(f"Could not extract date from filename: {filename}")
            continue

        try:
            date_info = extract_detailed_date_info(str(date_extracted))
        except ValueError as e:
            logger.error(f"Error parsing date: {e}")
            continue

        # Read the CSV file for this specific month
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            continue

        # Validate required columns
        if df.empty:
            logger.warning(f"Empty DataFrame in file: {filename}")
            continue

        # Standardize the POC column name
        df = standardize_poc_column(df)

        if 'POC' not in df.columns:
            logger.error(f"No POC column found in {filename}")
            continue

        # Create the new dataframe of site vectors
        try:
            df_vector = create_site_vector_dataframe(df, date_info)
        except Exception as e:
            logger.error(f"Error creating site vectors: {e}")
            continue

        if df_vector.empty:
            logger.warning(f"Empty result for {filename}, skipping save")
            continue

        # Add datetime index to the dataframe
        num_days = date_info["Days in Month"]
        num_periods = num_days * TRADING_PERIODS_PER_DAY
        
        date_index = pd.date_range(
            start=date_extracted, 
            periods=num_periods, 
            freq=f'{MINUTES_PER_TRADING_PERIOD}T'
        )
        
        df_vector.index = date_index
        df_vector = df_vector.rename_axis('DATE')

        # Apply scaling if requested (e.g., kWh to MWh)
        if scale_factor not in (1, 1.0, None):
            df_vector = scale_dataframe(df_vector, scale_factor, date_column='DATE')
            logger.info(f"  Applied scale factor: {scale_factor}")

        # Save the processed dataframe
        output_filepath = output_dir / output_file_list[index]
        try:
            df_vector.to_csv(output_filepath)
            logger.info(f"  ✓ Saved: {output_file_list[index]}")
            processed_count += 1
        except Exception as e:
            logger.error(f"  Error saving file: {e}")

    logger.info("")
    logger.info(f"Successfully processed {processed_count}/{len(input_file_list)} files")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nza_process_dynamic_data",
        description="Process NZ dynamic (half-hourly) datasets into standardised time series."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. If omitted, uses packaged default (pypsa_nza_data/config/nza_process_dynamic_data.yaml).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override paths.root from the YAML (workspace/data root). Use this to point data/logs/outputs outside the repo.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    """
    Main execution function for the data processing pipeline.
    
    Reads configuration, processes all energy types, and reports timing.
    """
    # Suppress excessive warnings
    warnings.filterwarnings('ignore')

    # Initialize program run-time
    start_time = time.time()

    # Read the yaml config file data
    args = parse_args(argv)

    config_file = Path(args.config).expanduser() if args.config else default_config_path()
    try:
        config = load_config(config_file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"✗ Fatal error loading configuration: {e}")
        return 1

    # Extract configuration parameters
    workspace_root = Path(args.root).expanduser().resolve() if args.root else Path(config.get('paths', {}).get('root', Path.cwd())).expanduser().resolve()
    root = workspace_root
    log_dir = resolve_path(config.get('paths', {}).get('logs', 'logs'), root)
    grid_energy_types = config.get('grid_energy_types', [])
    
    # Set up logging
    setup_logging(log_dir)
    
    # Print startup banner
    print_header("NZA DYNAMIC DATA PROCESSING")
    logger.info("")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration: {config_file}")
    logger.info(f"Root directory: {root}")
    logger.info(f"Log directory: {log_dir}")
    logger.info("")
    
    if not grid_energy_types:
        logger.error("✗ No grid_energy_types specified in configuration")
        return 1

    logger.info(f"Energy types to process: {', '.join(grid_energy_types)}")
    logger.info("")

    # Loop over energy types, cleaning the data for each
    for grid_energy in grid_energy_types:
        print_header(f"PROCESSING: {grid_energy.upper()}", '=')
        logger.info("")

        # Get input and output directories from config
        input_dir = resolve_path(config['paths'].get(f'{grid_energy}_in', ''), root)
        output_dir = resolve_path(config['paths'].get(f'{grid_energy}_out', ''), root)
        file_tag = f"{grid_energy}_md"

        # Validate directories
        if not input_dir.exists():
            logger.error(f"✗ Input directory does not exist: {input_dir}")
            continue

        logger.info(f"Input directory:  {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"File tag:         {file_tag}")
        logger.info(f"Scale factor:     {DEFAULT_SCALE_FACTOR} (kWh → MWh)")
        logger.info("")

        # Process all files for this energy type
        process_energy_by_modality(
            input_dir, 
            output_dir, 
            file_tag, 
            DEFAULT_SCALE_FACTOR
        )

    # Report timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print_header("PROCESSING COMPLETE", '=')
    logger.info("")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
    if grid_energy_types:
        logger.info(f"Average time per energy type: {elapsed_time/len(grid_energy_types):.2f} seconds")
    logger.info("")
    logger.info(f"Log file saved to: {log_dir}")
    logger.info("")

    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code else 0)
    except KeyboardInterrupt:
        logger.error("")
        logger.error("✗ Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"✗ Application failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
