#!/usr/bin/env python
# coding: utf-8

#******************************************************************************
#
#   grid_aggregate_columns.py
#
#   DESCRIPTION
#   aggregation of grid column data.
#
#   FUNCTIONS
#
#   LIBRARIES
#
#   DATA SOURCES
#
#   UPDATE HISTORY
#   Created on Wed May 12 06:48:01 2025
#   Author : Phillippe Bruneau
#
#******************************************************************************


# NOTE that functions
# list_filenames_in_directory
# rename_files_with_prefix
# have been copied from other programs "grid_pad_and_fill_dyn.py".
# These should be placed in a module for sharing.

import pandas as pd
import os
import re
from datetime import datetime
import calendar
from calendar import monthrange



# import data_proc as dp
# import time_proc as tme


def extract_datetime_from_filename(filename):
    """
    Extract date-time information from a filename without prior knowledge of its format.

    Supported Formats
    -----------------
    - YYYYMMDD (e.g., 20240328)
    - YYYY-MM-DD (e.g., 2024-03-28)
    - YYMMDD (e.g., 240328)
    - MMDDYYYY (e.g., 03282024)
    - DDMMYYYY (e.g., 28032024)
    - YYYYMM (e.g., 202403, assumes first day of the month)
    - Timestamp-like numbers

    Returns
    -------
        datetime object if a valid date is found, else None.
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



def extract_detailed_date_info(date_str):
    # Define common date formats to check, including pandas datetime formats
    date_formats = [
        "%Y-%m-%d",  # 2024-02-15
        "%d-%m-%Y",  # 15-02-2024
        "%d/%m/%Y",  # 15/02/2024
        "%m/%d/%Y",  # 02/15/2024
        "%Y/%m/%d",  # 2024/02/15
        "%B %d, %Y", # February 15, 2024
        "%d %B %Y",  # 15 February 2024
        "%b %d, %Y", # Feb 15, 2024
        "%d %b %Y",  # 15 Feb 2024
        "%y-%m-%d %H:%M:%S",  # 24-01-01 08:00:00 (pandas-style)
        "%Y-%m-%d %H:%M:%S",  # 2024-01-01 08:00:00 (full timestamp)
        "%Y-%m-%d %H:%M",     # 2024-01-01 08:00
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
        raise ValueError(f"Error: The provided date '{date_str}' does not match any recognized formats.")

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
    days_in_month = monthrange(year, month)[1]

    # Number of hours, minutes, and seconds in the year
    total_days_in_year = 366 if is_leap else 365
    total_hours_in_year = total_days_in_year * 24
    total_minutes_in_year = total_hours_in_year * 60
    total_seconds_in_year = total_minutes_in_year * 60

    # First and last day of the month
    first_day_of_month = datetime(year, month, 1).strftime("%Y-%m-%d (%A)")
    last_day_of_month = datetime(year, month, days_in_month).strftime("%Y-%m-%d (%A)")

    # Extract additional time-related information
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
        "Week Number": date_obj.strftime("%U"),  # Week number of the year
        "Quarter": (month - 1) // 3 + 1,  # Quarter of the year
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



def aggregate_columns(df: pd.DataFrame, identifier: str, filename: str =
                      "aggregated_output.csv") -> pd.DataFrame:
    """
    Aggregates columns of a DataFrame based on a specified identifier using efficient concatenation.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a datetime index and structured column names.
        identifier (str): One of 'poc', 'volts', 'unit', or 'all' for aggregation type.
        filename (str): Output CSV file name.

    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    # Keep the datetime column
    date_col = df.columns[0]
    result_df = df[[date_col]].copy()

    match identifier.lower():
        case 'poc':
            groups = {}
            for col in df.columns[1:]:
                key = col[:3]
                groups.setdefault(key, []).append(col)
            agg_df = pd.concat(
                [df[cols].sum(axis=1).rename(key) for key, cols in groups.items()],
                axis=1
            )

        case 'volts':
            groups = {}
            for col in df.columns[1:]:
                key = col[:6]  # POC + voltage
                groups.setdefault(key, []).append(col)
            agg_df = pd.concat(
                [df[cols].sum(axis=1).rename(key) for key, cols in groups.items()],
                axis=1
            )

        case 'unit':
            # Each column is unique; no aggregation needed
            agg_df = df.iloc[:, 1:].copy()

        case 'all':
            agg_df = pd.DataFrame({'SUM': df.iloc[:, 1:].sum(axis=1)})

        case _:
            raise ValueError("Identifier must be one of: 'poc', 'volts', 'unit', 'all'")

    result_df = pd.concat([result_df, agg_df], axis=1)
    result_df.to_csv(filename, index=False)
    return result_df


def list_filenames_in_directory(directory_path, include_full_path=True):
    """
    Return a list of filenames in the specified directory.

    Parameters
    ----------
    - directory_path (str): Path to the directory.
    - include_full_path (bool): If True, returns full file paths instead of just filenames.

    Returns
    -------
    files - List[str]: A list of filenames or file paths.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"The specified path does not exist or is not a directory: {directory_path}")

    files = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files.append(full_path if include_full_path else entry)

    return files


def rename_files_with_prefix(filenames, new_basename, new_extension=None):
    """
    Renames a list of files prefixed with 'yyyymm_'.

    Uses a new base name and  optional new extension.

    Parameters
    ----------
        filenames (list): List of filenames (e.g., ['202309_Generation_MD.csv']).
        new_basename (str): The new base name to insert after the prefix
                            (e.g., 'LoadData').
        new_extension (str, optional): New file extension (e.g., 'txt').
                            Default is to preserve the original.

    Returns
    -------
        list: List of renamed files in the format 'yyyymm_new_basename.extension'.
    """
    renamed = []
    for file in filenames:
        parts = file.split('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {file}")

        prefix = parts[0]
        original_ext = os.path.splitext(file)[1][1:]  # remove the dot
        extension = new_extension if new_extension else original_ext
        new_filename = f"{prefix}_{new_basename}.{extension}"
        renamed.append(new_filename)

    return renamed


if __name__ == '__main__':
    dirpath = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/data/processed/TEST/gen/g_MWh/"
    # dirpath = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/data/processed/2024/export/cons_MWh/"

    # Obtain the list of files to process
    input_file_list = list_filenames_in_directory(dirpath, False)

    # Define the aggregated data directory
    # diragg = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/data/processed/2024/cons/agg/"
    diragg = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/data/processed/TEST/export/cons_MWh_agg/"

    # Create a corresponding list of output files to store the cleaned up data
    # output_file_list = rename_files_with_prefix(input_file_list, 'gen_md_agg', new_extension=None)
    output_file_list = rename_files_with_prefix(input_file_list, 'export_md_agg', new_extension=None)

    # Outer loop - process each month at a time (each 'file' is a month's worth of data)
    for index, file in enumerate(input_file_list):
        # Full file path of the input file
        filepath = dirpath + file

        # Get date data info
        date_extracted = extract_datetime_from_filename(filepath)
        date_info = extract_detailed_date_info(str(date_extracted))

        year = date_info["Year"]
        month = date_info["Month"]
        num_days = date_info["Days in Month"]
        start_date = date_extracted

        # Read the csv file - specific month
        df = pd.read_csv(filepath)

        aggregated = aggregate_columns(df, identifier="all", filename=output_file_list[index])
        aggregated.to_csv(diragg + output_file_list[index], index=False)

