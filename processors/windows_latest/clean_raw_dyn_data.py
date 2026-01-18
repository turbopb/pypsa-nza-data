#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *****************************************************************************
#                                                                             *
#   clean_raw_dyn_data.py                                                     *
#                                                                             *
#   DESCRIPTION                                                               *
#   Cleans ansd reformats the raw, DYNAMIC data by aggregating and filling    *
#   missing 'POC' dates for a specified calendar month. This is a necessary   *
#   stage of the data cleaning process which transforms the data into a       *
#   uniform format.  Thereafter, generation and load vectors can be extracted *
#   for use in PyPSA.                                                         *
#                                                                             *
#                                                                             *
#   OVERVIEW                                                                  *
#   1. The headings of the dataframe are:                                     *
#   "POC, POC, Nwk_Code, Fuel_Code, Tech_Code, Gen_Code", "TP1, TP2, TP3 ...  *
#   TP48, TP49, TP50".  The 'TP' columns contain the primary i.e, load or     *
#   generation energies (as measured and provided by Transpower and the       *
#   Electricity Authority.                                                    *
#                                                                             *
#   2. The meanings of the heading are:                                       *
#   'POC' - a unique string identifier for the measurement location.          *
#   'Trading_Date' - datetime stamps for a particular day, e.g. '20-01-08'    *
#   'TPn'          - 'Trading Period', 30 min time blocks starting from       *
#                    and continuing through to '23:30:00' daily.              *
#                                                                             *
#   3. Each data row row reprensents one day of measured values over a given  *
#   month and year.
#   Typically, ther will be 30, 31, 28 or 29 rows of data that corresponds to *
#   each day (depending on the month)                                         *
#                                                                             *
#   4. There may be cases where there are additional data rows associated     *
#   with a given 'POC'. There may also be cases where there is not a          *
#   full set of data for all days for a given 'POC'.                          *
#                                                                             *
#   5. The date stamps may or may not follow a consecutive sequance or the    *
#   number of rows may not be for a full month.                               *
#                                                                             *
#   The function aggregates (sums) the data rows for each row which has the   *
#   same 'POC' and 'Trading_Date' values. The intent is create a new          *
#   datafrme that has a total of n = month days entries that correspond to    *
#   each 'POC'(aggregated if needs be) .                                      *
#   If there are are 'm' unique 'POC' values then 'n x m' entries are         *
#   expected in the new dataframe. If there is a case where there are not 'n' *
#   days of data for a given site code, then the missing data is back-filled  *
#   with zeros.                                                               *
#                                                                             *
#   WORKFLOW                                                                  *
#                                                                             *
#                                                                             *
#   DATA                                                                      *
#                                                                             *
#   OUTPUT                                                                    *
#                                                                             *
#                                                                             *
#                                                                             *
#   DATA SOURCES                                                              *
#                                                                             *
#   UPDATE HISTORY                                                            *
#   Created on Tue May 6 15:48:33 2025                                        *
#   Author : Phillippe Bruneau                                                *
#                                                                             *
# *****************************************************************************

import yaml
import re
import os
import pandas as pd
import calendar
from calendar import monthrange
from pathlib import Path
import time
from datetime import date
from datetime import datetime


def load_config(config_file: str) -> dict:
    """
    Read the YAML configuration file.

    Parameters
    ----------
        config_file (str): Path to the YAML configuration file.

    Returns
    -------
        config (dict): YAML configuration data in a dictionary.
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

    except OSError as e:
        print(f"Error: Could not create directory '{directory_path}'")
        print(f"Reason: {e}")
        return False

    except Exception as e:
        print(f"Error: Unexpected error while creating directory '{directory_path}'")
        print(f"Reason: {type(e).__name__}: {e}")
        return False

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



def standardize_poc_column(df):
    """
    Renames any column in the dataframe that contains 'poc' (case insensitive)
    to 'POC'. Only the first match will be renamed.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: DataFrame with one standardized 'POC' column name.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Find columns containing 'poc' (case insensitive)
    poc_cols = [col for col in df.columns if 'poc' in col.lower()]

    if poc_cols:
        # Rename the first matching column to 'POC'
        df.rename(columns={poc_cols[0]: 'POC'}, inplace=True)

    return df


def rename_files_with_prefix(filenames, new_basename, new_extension=None):
    """
    Renames a list of files prefixed with 'yyyymm_'.

    Uses a new base name and  optional new extension.

    Parameters
    ----------
        filenames (list): List of filenames, e.g., ['202309_Generation_MD.csv'].

        new_basename (str): The new base name to insert after the prefix (e.g.,
                            'LoadData').

        new_extension (str, optional): New file extension (e.g., 'txt'). Default
                                      is to preserve the original.

    Returns
    -------
        list: List of renamed files in the format :
                                            'yyyymm_new_basename.extension'.
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


def save_dataframe_to_csv(df, filepath):
    """
    Save a DataFrame to a CSV file.

    Parameters
    ----------
    df  - (pd.DataFrame): The DataFrame to save.
    filepath (str) - The full path to the CSV file, including the filename.

    Creating the directory if it doesn't exist.

    Returns
    -------
    None
    """
    # Create the directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame to CSV
    df.to_csv(filepath, index=False)


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


def aggregate_and_fill_calendar_month(df, year, month, tp_columns=None):
    """
    Aggregate and fill missing dates.

    Aggregate and fill missing dates for each POC for a specified
    calendar month.

    Parameters
    ----------
    - df (pd.DataFrame): Input DataFrame with 'POC', 'Trading_Date', and
        'TP1'...'TP50' columns.
    - year (int): Target year (e.g., 2025)
    - month (int): Target month (e.g., 4 for April)
    - tp_columns (list of str, optional): List of TP columns. If None, then TP
        columns are auto-detected.

    Returns
    -------
    - pd.DataFrame: Aggregated DataFrame with zero-filled missing days.
    """
    df = df.copy()

    # Ensure 'Trading_Date' is datetime
    df['Trading_Date'] = pd.to_datetime(df['Trading_Date'], format='%Y-%m-%d')

    # Filter to only rows matching year and month
    df = df[(df['Trading_Date'].dt.year == year) & (df['Trading_Date'].dt.month == month)]

    if tp_columns is None:
        tp_columns = [col for col in df.columns if col.startswith('TP')]

    # Step 1: Aggregate by POC and Trading_Date
    df_agg = df.groupby(['POC', 'Trading_Date'])[tp_columns].sum().reset_index()

    # Step 2: Prepare full date range for the month
    days_in_month = calendar.monthrange(year, month)[1]
    full_dates = pd.date_range(start=f'{year}-{month:02d}-01', periods=days_in_month, freq='D')

    all_POCs = df_agg['POC'].unique()

    final_df_list = []

    for site in all_POCs:
        site_data = df_agg[df_agg['POC'] == site]

        full_site_df = pd.DataFrame({
            'POC': site,
            'Trading_Date': full_dates
        })

        merged = pd.merge(full_site_df, site_data, on=['POC', 'Trading_Date'],
                                                                     how='left')
        merged[tp_columns] = merged[tp_columns].fillna(0)

        final_df_list.append(merged)

    # Optional: Handle POCs that were missing completely
    # (Unlikely, unless some codes have zero days entirely)

    final_df = pd.concat(final_df_list, ignore_index=True)

    return final_df


def flatten_dataframe(df:pd.DataFrame, date_dict:dict) -> pd.DataFrame:
    """
    Process a list of POC DataFrames.

    Summary
    -------

    1. Remove TP49 and TP50.
    2. Flatten into a single column.
    3. Assemble into a single DataFrame with proper datetime index.

    Parameters
    ----------
    - df: List of (POC name, DataFrame) tuples.
    - start_date: Start date of the month as a string, e.g., "2024-08-01".
    - num_days: Number of days in the month (28, 29, 30, or 31).

    Returns
    -------
    - DataFrame with POC columns and 30-min datetime index.
    """
    #
    # Construct the datetime index
    #total_half_hours = ndays * 48
    #datetime_index = pd.date_range(start=date, periods=total_half_hours, freq='30min')

    # Group by unique POC
    # unique_POCs = df['POC'].unique()
    # #print(unique_POCs)

    flattened_data = {}
    df_trim = df.drop(columns=[col for col in ["POC", "Trading_Date", "TP49",
                                               "TP50"] if col in df.columns])
    #    print(f"{df_trim.tail()}")
     #   print(sc)

        # Store the flattened series
    vector = df_trim.to_numpy().flatten()
    # print(vector, len(vector))
        # print(df_trim.shape)


    # Assemble into a single DataFrame
    #combined_df = pd.DataFrame(flattened_data)
    # print(combined_df)

    # combined_df = pd.DataFrame(flattened_data, index=datetime_index)

    return vector # combined_df


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


def site_vector_dataframe(df, date_info):
    # List of all UNIQUE sites
    unique_POCs = df['POC'].unique()

    # Start the the cleaning and reformatting process
    year = date_info["Year"]
    month = date_info["Month"]
    num_days = date_info["Days in Month"]
    #start_date = date_extracted

    try:
        # Generate a clean, aggregated dataframe for the current month
        df_clean = aggregate_and_fill_calendar_month(df, year, month, tp_columns=None)

        # Create a vector for each site and store in a dict
        vector_dict = {}  # Initialise to empty

        # Extract the data for each site and accumulate in the dict
        for sc in unique_POCs:
            df_site = df_clean[df_clean['POC'] == sc].copy()
            vec_energy = flatten_dataframe(df_site, date_info)
            vector_dict[sc] = vec_energy

        month = date_info["Month"]
        print(f"{month} ", date(1900, month, 1).strftime('%B'))
        #print('------------')
    except ValueError as e:
        print(e)

    return pd.DataFrame.from_dict(vector_dict)


def scale_dataframe(df, scale_factor, date_column='DATE'):
    # Make a copy to avoid modifying the original
    df_scaled = df.copy()

    # Scale only numeric columns excluding the date column
    numeric_cols = df.columns.difference([date_column])
    df_scaled[numeric_cols] = df_scaled[numeric_cols] * scale_factor

    return df_scaled

def energy_by_modality(input_dir, output_dir, file_tag):
    # Obtain the list of files to process
    input_file_list = list_filenames_in_directory(input_dir, False)

    # Create a corresponding list of output files to store the cleaned up data
    # output_file_list = rename_files_with_prefix(input_file_list, 'gen_md_cons', new_extension=None)
    # output_file_list = rename_files_with_prefix(input_file_list, 'import_md_cons', new_extension=None)
    output_file_list = rename_files_with_prefix(input_file_list, file_tag, new_extension=None)

    # Outer loop - process each month at a time (each 'file' is a month's worth of data)
    for index, file in enumerate(input_file_list):

        # Full file path of the input file
        filepath = input_dir + file

        # Get date data info
        date_extracted = extract_datetime_from_filename(filepath)
        date_info = extract_detailed_date_info(str(date_extracted))

        # year = date_info["Year"]
        # month = date_info["Month"]

        #start_date = date_extracted

        # Read the csv file - specific month
        df = pd.read_csv(filepath)

        # Check that the "POC"column name is correct - modify if necessary
        df = standardize_poc_column(df)

        # Crate the new dataframe of site vectors
        df_vector = site_vector_dataframe(df, date_info)

        # Tidy up the dataframe and save in the processed data directory
        num_days = date_info["Days in Month"]
        date_index = pd.date_range(start=date_extracted, periods=num_days*24/0.5, freq='30T')
        df_vector.index = date_index
        df_vector = df_vector.rename_axis('DATE')

        if scale_factor != 1 or None:
            df_vector = scale_dataframe(df_vector, scale_factor, date_column='DATE')
        df_vector.to_csv(output_dir + output_file_list[index])


# def main():
#     # Define input and output directories


if __name__ == '__main__':
    # Suppress excessive verbiage
    import warnings ; warnings.warn = lambda *args,**kwargs: None

    # Initialise program run-time
    start_time = time.time()

    # Read the yaml config file data into a dict
    config_file = '../../configs/nza_raw_data.yaml'
    config = load_config(config_file)

    with open(config_file) as f:
        config = yaml.safe_load(f)

    root = config['paths']['root']

    gen_in = root + config['paths']['gen_in']
    gen_out = root + config['paths']['gen_out']
    ensure_directory_exists(gen_out)

    export_in = root + config['paths']['export_in']
    export_out = root + config['paths']['export_out']
    ensure_directory_exists(export_out)

    import_in = root + config['paths']['import_in']
    import_out = root + config['paths']['import_out']
    ensure_directory_exists(import_out)

    print(" NODALITIES")
    modalities = config['modality']
    print(modalities)

    scale_factor = 1.0e-3   # Scale from kWh to MWh
    for madality in modalities:
        source = madality # ='export'

        match source:
            case "gen":
                print("\nGENERATION")
                input_dir = gen_in
                output_dir = gen_out
                file_tag = 'gen_md'

            case "export":
                print("\nEXPORT")
                input_dir = export_in
                output_dir = export_out
                file_tag = 'export_md'

            case "import":
                print("\nIMPORT")
                input_dir = import_in
                output_dir = import_out
                file_tag = 'import_md'

            case _:
                print(f"SOURCE type {source} not defined")

        energy_by_modality(input_dir, output_dir, file_tag)

# -----------------------------------------------------------------------------
    # Stop program timer
    end_time = time.time()
    elapsed_time = end_time-start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    #print(f"CPU Time ={dt} ")

# -----------------------------------------------------------------------------
