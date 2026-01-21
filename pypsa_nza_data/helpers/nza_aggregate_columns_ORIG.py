#!/usr/bin/env python
# coding: utf-8

"""
nza_aggregate_columns.py

A data processing pipeline for aggregating grid energy data columns.

WORKFLOW OVERVIEW:
==================
This script processes monthly grid energy data files and aggregates columns based on
specified identifiers. The main workflow consists of:

1. File Discovery: Scans a specified directory for monthly energy data files
2. Date Extraction: Extracts date information from filenames to understand temporal context
3. Data Loading: Reads CSV files containing grid energy measurements
4. Column Aggregation: Aggregates columns based on user-specified criteria:
   - 'poc': Aggregate by Point of Connection (first 3 characters)
   - 'volts': Aggregate by POC and voltage level (first 6 characters)
   - 'unit': Keep individual units (no aggregation)
   - 'all': Sum all columns into a single total
5. Output Generation: Writes aggregated data to new CSV files with standardized naming

AGGREGATION STRATEGIES:
=======================
- POC Aggregation: Groups columns by site identifier (e.g., ARI, BWK, CYD)
- Voltage Aggregation: Groups by site and voltage level (e.g., ARI110, BWK220)
- Unit Level: Preserves individual generator/unit granularity
- Total Sum: Collapses all energy flows into a single system-wide value

KEY FEATURES:
=============
- Flexible date format recognition from filenames
- Efficient pandas-based aggregation using concatenation
- Automatic output file naming with month prefixes (YYYYMM_)
- Comprehensive date metadata extraction for validation
- Directory-based batch processing

USE CASES:
==========
- Creating regional summaries from detailed grid data
- Simplifying analysis by reducing dimensionality
- Preparing aggregated data for capacity expansion models
- Generating system-wide energy totals for reporting

OUTPUT FORMAT:
==============
All output files maintain the format: YYYYMM_basename_agg.csv
Each file contains timestamp column plus aggregated energy columns

NOTES:
======
- Functions list_filenames_in_directory() and rename_files_with_prefix()
  should be moved to a shared utility module for reusability
- Input files must follow the YYYYMM_ naming convention
- All energy values are assumed to be in MWh

Author: Phillippe Bruneau
Created: Wed May 12 06:48:01 2025
Updated on Wed Oct  1 17:22:36 2025
"""

import pandas as pd
import os
import re
from datetime import datetime
from calendar import monthrange
from typing import Dict, List, Optional, Tuple


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract datetime information from a filename using pattern matching.

    Supports multiple date formats without prior knowledge of the format.

    Supported Formats
    -----------------
    - YYYYMMDD (e.g., 20240328)
    - YYYY-MM-DD (e.g., 2024-03-28)
    - YYMMDD (e.g., 240328)
    - MMDDYYYY (e.g., 03282024)
    - DDMMYYYY (e.g., 28032024)
    - YYYYMM (e.g., 202403, assumes first day of the month)
    - Timestamp-like numbers

    Parameters
    ----------
    filename : str
        Filename containing date information.

    Returns
    -------
    datetime or None
        Parsed datetime object if valid date found, None otherwise.

    Examples
    --------
    >>> extract_datetime_from_filename("202403_generation.csv")
    datetime.datetime(2024, 3, 1, 0, 0)

    >>> extract_datetime_from_filename("2024-03-28_data.csv")
    datetime.datetime(2024, 3, 28, 0, 0)
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
                date_str = "-".join(match.groups())
                extracted_date = datetime.strptime(date_str, date_format)

                # Handle 2-digit years by assuming 2000s
                if "%y" in date_format and extracted_date.year < 2000:
                    extracted_date = extracted_date.replace(year=extracted_date.year + 2000)

                return extracted_date
            except ValueError:
                continue  # Try next pattern if this one fails

    return None


def extract_detailed_date_info(date_str: str) -> Dict:
    """
    Extract comprehensive date information from a date string.

    Parameters
    ----------
    date_str : str
        Date string in various formats.

    Returns
    -------
    dict
        Dictionary containing detailed date components and metadata.

    Raises
    ------
    ValueError
        If the date string doesn't match any recognized format.

    Examples
    --------
    >>> info = extract_detailed_date_info("2024-03-15")
    >>> info["Year"]
    2024
    >>> info["Month"]
    3
    >>> info["Days in Month"]
    31
    """
    date_formats = [
        "%Y-%m-%d",           # 2024-02-15
        "%d-%m-%Y",           # 15-02-2024
        "%d/%m/%Y",           # 15/02/2024
        "%m/%d/%Y",           # 02/15/2024
        "%Y/%m/%d",           # 2024/02/15
        "%B %d, %Y",          # February 15, 2024
        "%d %B %Y",           # 15 February 2024
        "%b %d, %Y",          # Feb 15, 2024
        "%d %b %Y",           # 15 Feb 2024
        "%y-%m-%d %H:%M:%S",  # 24-01-01 08:00:00
        "%Y-%m-%d %H:%M:%S",  # 2024-01-01 08:00:00
        "%Y-%m-%d %H:%M",     # 2024-01-01 08:00
    ]

    # Try parsing with different formats
    date_obj = None
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue

    if date_obj is None:
        raise ValueError(
            f"Error: The provided date '{date_str}' does not match any recognized formats."
        )

    # Extract basic components
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    hour = date_obj.hour
    minute = date_obj.minute
    second = date_obj.second

    # Calculate derived information
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days_in_month = monthrange(year, month)[1]
    total_days_in_year = 366 if is_leap else 365
    total_hours_in_year = total_days_in_year * 24
    total_minutes_in_year = total_hours_in_year * 60
    total_seconds_in_year = total_minutes_in_year * 60

    # Get first and last day of month
    first_day_of_month = datetime(year, month, 1).strftime("%Y-%m-%d (%A)")
    last_day_of_month = datetime(year, month, days_in_month).strftime("%Y-%m-%d (%A)")

    # Compile comprehensive date information
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
        "Total Minutes in Year": total_minutes_in_year,
        "Total Seconds in Year": total_seconds_in_year,
        "First Day of Month": first_day_of_month,
        "Last Day of Month": last_day_of_month,
    }

    return date_info


def aggregate_columns(df: pd.DataFrame, identifier: str,
                     filename: str = "aggregated_output.csv") -> pd.DataFrame:
    """
    Aggregate DataFrame columns based on specified identifier.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime column and structured column names.
    identifier : str
        Aggregation type - one of 'poc', 'volts', 'unit', or 'all':
        - 'poc': Aggregate by Point of Connection (first 3 chars)
        - 'volts': Aggregate by POC and voltage (first 6 chars)
        - 'unit': No aggregation, keep individual units
        - 'all': Sum all columns into single total
    filename : str, optional
        Output CSV filename. Default is "aggregated_output.csv".

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with datetime column and aggregated data.

    Raises
    ------
    ValueError
        If identifier is not one of the valid options.

    Examples
    --------
    >>> df = pd.read_csv("202403_generation.csv")
    >>> agg_df = aggregate_columns(df, identifier="poc", filename="output.csv")
    """
    # Preserve the datetime column
    date_col = df.columns[0]
    result_df = df[[date_col]].copy()

    match identifier.lower():
        case 'poc':
            # Aggregate by first 3 characters (Point of Connection)
            groups = {}
            for col in df.columns[1:]:
                key = col[:3]
                groups.setdefault(key, []).append(col)
            agg_df = pd.concat(
                [df[cols].sum(axis=1).rename(key) for key, cols in groups.items()],
                axis=1
            )

        case 'volts':
            # Aggregate by POC and voltage level (first 6 characters)
            groups = {}
            for col in df.columns[1:]:
                key = col[:6]
                groups.setdefault(key, []).append(col)
            agg_df = pd.concat(
                [df[cols].sum(axis=1).rename(key) for key, cols in groups.items()],
                axis=1
            )

        case 'unit':
            # No aggregation - preserve individual units
            agg_df = df.iloc[:, 1:].copy()

        case 'all':
            # Sum all columns into a single total
            agg_df = pd.DataFrame({'SUM': df.iloc[:, 1:].sum(axis=1)})

        case _:
            raise ValueError(
                f"Invalid identifier '{identifier}'. "
                "Must be one of: 'poc', 'volts', 'unit', 'all'"
            )

    # Combine datetime with aggregated data
    result_df = pd.concat([result_df, agg_df], axis=1)
    result_df.to_csv(filename, index=False)

    return result_df


def list_filenames_in_directory(directory_path: str,
                               include_full_path: bool = True) -> List[str]:
    """
    Return a list of filenames in the specified directory.

    Parameters
    ----------
    directory_path : str
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
    if not os.path.isdir(directory_path):
        raise ValueError(
            f"The specified path does not exist or is not a directory: {directory_path}"
        )

    files = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files.append(full_path if include_full_path else entry)

    return files


def rename_files_with_prefix(filenames: List[str], new_basename: str,
                            new_extension: Optional[str] = None) -> List[str]:
    """
    Rename files with YYYYMM_ prefix using a new base name.

    Parameters
    ----------
    filenames : List[str]
        List of filenames with YYYYMM_ prefix (e.g., ['202309_Generation_MD.csv']).
    new_basename : str
        New base name to insert after the prefix (e.g., 'LoadData').
    new_extension : str, optional
        New file extension (e.g., 'txt'). If None, preserves original extension.

    Returns
    -------
    List[str]
        List of renamed files in format 'YYYYMM_new_basename.extension'.

    Raises
    ------
    ValueError
        If filename doesn't follow the expected format.

    Examples
    --------
    >>> files = ['202403_generation.csv', '202404_generation.csv']
    >>> rename_files_with_prefix(files, 'gen_agg')
    ['202403_gen_agg.csv', '202404_gen_agg.csv']
    """
    renamed = []
    for file in filenames:
        parts = file.split('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {file}")

        prefix = parts[0]
        original_ext = os.path.splitext(file)[1][1:]  # Remove the dot
        extension = new_extension if new_extension else original_ext
        new_filename = f"{prefix}_{new_basename}.{extension}"
        renamed.append(new_filename)

    return renamed


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
        return False
    except OSError as e:
        print(f"Error: Could not create directory '{directory_path}'")
        print(f"Reason: {e}")
        return False
    except Exception as e:
        print(f"Error: Unexpected error while creating directory '{directory_path}'")
        print(f"Reason: {type(e).__name__}: {e}")
        return False


def process_monthly_aggregation(input_dir: str, output_dir: str,
                                aggregation_type: str = "all",
                                output_basename: str = "aggregated") -> None:
    """
    Process all monthly files in a directory and create aggregated outputs.

    Parameters
    ----------
    input_dir : str
        Directory containing input CSV files.
    output_dir : str
        Directory where aggregated files will be saved.
    aggregation_type : str, optional
        Type of aggregation ('poc', 'volts', 'unit', 'all'). Default is 'all'.
    output_basename : str, optional
        Base name for output files. Default is 'aggregated'.
    """
    print(f"Processing files from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Aggregation type: {aggregation_type}\n")

    # Ensure output directory exists
    if not ensure_directory_exists(output_dir, verbose=True):
        print("Failed to create output directory. Exiting.")
        return

    # Get list of input files
    try:
        input_file_list = list_filenames_in_directory(input_dir, include_full_path=False)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not input_file_list:
        print("No files found in input directory.")
        return

    # Create corresponding output filenames
    output_file_list = rename_files_with_prefix(
        input_file_list,
        output_basename,
        new_extension=None
    )

    print(f"Found {len(input_file_list)} files to process\n")

    # Process each file
    for index, filename in enumerate(input_file_list):
        print(f"Processing file {index + 1}/{len(input_file_list)}: {filename}")

        # Construct full input file path
        filepath = os.path.join(input_dir, filename)

        # Extract date information
        date_extracted = extract_datetime_from_filename(filepath)
        if date_extracted:
            date_info = extract_detailed_date_info(str(date_extracted))
            print(f"  Date: {date_info['Year']}-{date_info['Month']:02d}")
            print(f"  Days in month: {date_info['Days in Month']}")

        # Read and aggregate the data
        try:
            df = pd.read_csv(filepath)
            aggregated = aggregate_columns(
                df,
                identifier=aggregation_type,
                filename=output_file_list[index]
            )

            # Save to output directory
            output_path = os.path.join(output_dir, output_file_list[index])
            aggregated.to_csv(output_path, index=False)
            print(f"  Saved: {output_file_list[index]}\n")

        except Exception as e:
            print(f"  Error processing file: {e}\n")
            continue

    print(f"Processing complete! {len(input_file_list)} files processed.")


def main():
    """Main execution function."""
    # Configuration
    input_dir = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/data/processed/2024/gen/cons_MWh"
    output_dir = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/data/processed/2024/gen/agg_MWh"

    aggregation_type = "poc"  # Options: 'poc', 'volts', 'unit', 'all'
    output_basename = "gen_agg_MWh"

    # Process the files
    process_monthly_aggregation(
        input_dir=input_dir,
        output_dir=output_dir,
        aggregation_type=aggregation_type,
        output_basename=output_basename
    )


if __name__ == '__main__':
    main()