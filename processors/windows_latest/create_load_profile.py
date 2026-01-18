#!/usr/bin/env python
# coding: utf-8

#******************************************************************************
#                                                                             *
#   create_load_profile.py                                                    *
#                                                                             *
#   DESCRIPTION                                                               *
#   Creates and prepares bus, generation and demand load profiles for a       *
#   capital expansion planning analysis using PyPSA.                          *
#                                                                             *
#   WORKFLOW                                                                  *
#   Grid import and export data files are modified and amalgamated            *
#                                                                             *
    #   *** CHECKS ***
    #   *** NOTE ***
    # WPG in the gen_list is the same as HWB

#   FUNCTIONS                                                                 *
#                                                                             *
#   LIBRARIES                                                                 *
#                                                                             *
#   DATA SOURCES                                                              *
#                                                                             *
#   UPDATE HISTORY                                                            *
#   Created on Wed May 28 16:13:34 2025                                       *
#   Author : Phillippe Bruneau                                                *
#                                                                             *
#******************************************************************************

import numpy as np
import pandas as pd
import os
import yaml
import calendar



def load_config(config_file: str) -> dict:
    """
    Read a YAML configuration file.

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


def generate_monthly_filenames(year, months, basename, extension):
    import re

    # Normalize input
    year = str(year)
    extension = extension.strip().lstrip(".")
    basename = basename.strip()

    # Helper: convert month name/abbrev to number
    def month_to_number(m):
        m = m.strip().lower()
        for i in range(1, 13):
            if m == calendar.month_abbr[i].lower() or m == calendar.month_name[i].lower():
                return i
        if m.isdigit() and 1 <= int(m) <= 12:
            return int(m)
        raise ValueError(f"Invalid month: {m}")

    # Handle 'all'
    if isinstance(months, str) and months.strip().lower() == 'all':
        month_nums = list(range(1, 13))

    # Handle comma-separated string or list
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

    # Handle a single string like "Jan"
    else:
        month_nums = [month_to_number(months)]

    # Build the filenames
    filenames = [
        f"{year}{month:02d}_{basename}.{extension}" for month in month_nums
    ]

    return filenames

def list_filenames_in_directory(directory_path, include_full_path=True) -> list:
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


def aggregate_columns_by_prefix(csv_file_path):
    """
    Aggregate columns.

    Parameters
    ----------
    csv_file_path : TYPE
        DESCRIPTION.

    Returns
    -------
    result_df : TYPE
        DESCRIPTION.

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

def subtract_site_data(df1:pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
    """
    Subtract two dataframes.

    Parameters
    ----------
    df1 : TYPE
        DESCRIPTION.
    df2 : TYPE
        DESCRIPTION.
    output_path : TYPE, optional
        DESCRIPTION. The default is "demand.csv".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    result_df : TYPE
        DESCRIPTION.
    missing_sites : TYPE
        DESCRIPTION.
    clipped_sites : TYPE
        DESCRIPTION.

    """
    # Load both CSV files

    # df1 = pd.read_csv(file1_path)
    # df2 = pd.read_csv(file2_path)

    # Validate 'DATE' column
    if 'DATE' not in df1.columns or 'DATE' not in df2.columns:
        raise ValueError("Both files must contain a 'DATE' column.")

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

    # Save to CSV
    #result_df.to_csv(output_path, index=False)

    # Report
    if missing_sites:
        print("*** Columns in file2 not found in file1 (skipped):", missing_sites)
    if clipped_sites:
        print("*** Subtracted values clipped to zero for sites:", clipped_sites)
    #print(f"? Output written to {output_path}")

    return result_df, missing_sites, clipped_sites


def add_matching_columns_with_timestamp(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Add the values of corresponding named columns in two DataFrames.

    excluding the first column (assumed to be timestamps).
    The resulting DataFrame retains the timestamp column in the first position.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Input dataframes with the first column as a timestamp.

    Returns
    -------
    pd.DataFrame
        New dataframe with summed matched columns and all unmatched columns,
        with timestamps preserved.
    """
    # Extract timestamp columns (assumed to be first column)
    timestamp1 = df1.iloc[:, 0]
    timestamp2 = df2.iloc[:, 0]

    if not timestamp1.equals(timestamp2):
        print(timestamp1, timestamp2)
        #print(timestamp2)
        #raise ValueError("Timestamp columns do not match between dataframes.")


    # Drop the timestamp columns before processing
    df1_data = df1.iloc[:, 1:]
    df2_data = df2.iloc[:, 1:]

    # Find matched and unmatched columns
    matched_cols = df1_data.columns.intersection(df2_data.columns)
    unmatched_df1 = df1_data.columns.difference(df2_data.columns)
    unmatched_df2 = df2_data.columns.difference(df1_data.columns)

    print("Matched columns:", list(matched_cols))
    if not unmatched_df1.empty:
        print("Unmatched columns only in df1:", list(unmatched_df1))
    if not unmatched_df2.empty:
        print("Unmatched columns only in df2:", list(unmatched_df2))

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

    # Optional: sort non-timestamp columns alphabetically (except keep timestamp first)
    non_ts_cols = sorted(result.columns[1:])
    result = result[[result.columns[0]] + non_ts_cols]

    return result


def drop_and_clean_negatives(df: pd.DataFrame, drop_cols: list, verbose: bool = False) -> pd.DataFrame:
    """
    Drop named columns from a DataFrame.

    Also sets all negative values in the remaining DataFrame to 0, and
    optionally prints information about negative values found.

    Parameters
    ----------
    df         : pd.DataFrame - The input DataFrame.
    drop_cols  : list         - List of column names to drop.
    verbose    : bool         - If True, print detailed info on negative
                                values found.

    Returns
    -------
    pd.DataFrame - New DataFrame with specified columns removed and negatives
                   set to 0.
    """
    df_clean = df.copy()

    # Drop the specified columns that exist
    missing_cols = [col for col in drop_cols if col not in df_clean.columns]
    existing_cols = [col for col in drop_cols if col in df_clean.columns]

    if existing_cols:
        df_clean.drop(columns=existing_cols, inplace=True)

    if missing_cols and verbose:
        print(f"Columns not found in DataFrame and not dropped: {missing_cols}")

    # Identify negative values
    numeric_df = df_clean.select_dtypes(include=[np.number])
    neg_mask = numeric_df < 0
    num_negatives = neg_mask.sum().sum()

    if verbose:
        if num_negatives > 0:
            print(f"Found {num_negatives} negative value(s). They will be set to 0.")
            print("Negative values by column:")
            print(neg_mask.sum())
        else:
            print("No negative values found.")

    # Replace negative values with 0
    df_clean[numeric_df.columns] = numeric_df.clip(lower=0)

    return df_clean


def replace_negatives_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all negative values in a DataFrame with zero.

    Excluding the first column (assumed to be a timestamp or index column).

    Parameters
    ----------
    df : pd.DataFrame - Input DataFrame.

    Return
    ------
    pd.DataFrame - Cleaned DataFrame with negative values set to 0 (except first column).
    """
    df_clean = df.copy()

    # Preserve the first column (e.g., timestamps)
    # first_col = df_clean.columns[0]

    # Operate only on remaining columns
    numeric_cols = df_clean.columns[1:]

    # Replace negatives with 0
    df_clean[numeric_cols] = df_clean[numeric_cols].map(lambda x: max(x, 0) if pd.notnull(x) else x)

    return df_clean


def rename_files_with_prefix(filenames, new_basename, new_extension=None) -> list:
    """
    Rename a list of files prefixed with 'yyyymm_'.

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


def extract_matching_buses(df: pd.DataFrame, bus_list: list[str]) -> pd.DataFrame:
    """
    Extract rows from master dataframe based list specification.

    Extract rows from the master dataframe where the 'site' column matches
    any value in the given list.

    Parameters
    ----------
        df (pd.DataFrame): The original master dataframe.
        bus_list (list of str): List of bus names to extract.

    Returns
    -------
        pd.DataFrame: A new dataframe containing only the matching rows.
    """
    # Ensure 'site' column exists
    if 'site' not in df.columns:
        raise KeyError("The dataframe does not contain a 'site' column.")

    # Filter the dataframe using vectorized boolean masking
    filtered_df = df[df['site'].isin(bus_list)].copy()

    # Reset index (optional, remove if you want to retain original indices)
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df


def check_bus_connectivity(df, bus_list, verbose=False):
    """
    Check if each bus in bus_list appears in either 'bus0' or 'bus1' columns
    of the transmission lines file, indicating connectivity.

    Parameters:
        lines_file (str): Path to 'lines_data.csv'
        bus_list (list): List of bus names to check (e.g., ['ARI', 'BWK'])

    Returns:
        connected_buses (list): Buses that appear in either 'bus0' or 'bus1'
        disconnected_buses (list): Buses not found in the line topology
    """

    # Collect all unique connected buses
    connected_set = set(df['bus0']).union(set(df['bus1']))

    # Check each bus for presence
    connected_buses = [bus for bus in bus_list if bus in connected_set]
    disconnected_buses = [bus for bus in bus_list if bus not in connected_set]

    #print("Connected buses:", connected_buses)
    if not disconnected_buses:
        pass
    else:
        print(f"\nDisconnected buses: {disconnected_buses} \n")

    return connected_buses, disconnected_buses

#------------------------------------------------------------------------------
    def node_energy_flow(path):
        file_list = list_filenames_in_directory(path, False)

        df_list = []
        for file in file_list:
            file_path = path + file
            df = aggregate_columns_by_prefix(file_path)
            df_list.append(df)

        return df_list


def aggregate_monthly_flow_data(dirpath):
    file_list = list_filenames_in_directory(dirpath, False)

    df_list = []

    for file in file_list:
        file_path = dirpath + file
        df_f = aggregate_columns_by_prefix(file_path)
        df_list.append(df_f)

    return df_list

#------------------------------------------------------------------------------
    # df2_list = node_energy_flow(dirpath_import)
    # df1_list = node_energy_flow(dirpath_export)
#------------------------------------------------------------------------------


if __name__ == '__main__':
    # Read the yaml config file data into a dict
    config_file = '../../configs/nza_cx_config.yaml'
    config = load_config(config_file)

    root = config['paths']['root']
    dirpath_export  = root + config['paths']['dirpath_export']
    dirpath_import  = root + config['paths']['dirpath_import']
    dirpath_gen     = root + config['paths']['dirpath_gen']
    dirpath_g       = root + config['paths']['dirpath_g']
    dirpath_bus     = root + config['paths']['dirpath_bus']
    dirpath_load    = root + config['paths']['dirpath_load']
    dirpath_demand  = root + config['paths']['dirpath_demand']
    dirpath_f1f2    = root + config['paths']['dirpath_f1f2']
    dirpath_static  = root + config['paths']['dirpath_static']
    dirpath_cx      = root + config['paths']['dirpath_cx']
    year            = config['start_year']

    # Grid EXPORT and IMPORT - f1 & F2.
    # Note that df_f2 and df_f1 are lists of dataframes :
    # df_f2[0] => df_f2("January"), df_f2[1] => df_f2("February") .....
    #                               ...=> df_f2[11] => df_f2("December")
    df_f2 = aggregate_monthly_flow_data(dirpath_import)
    df_f1 = aggregate_monthly_flow_data(dirpath_export)

    # Read the master bus data set.  All possible busses and information
    # pertaining to the busses., e.g. geo location  coords, etc.
    df_bus_master = pd.read_csv(dirpath_static + "bus_data.csv")

    # Create a list of bus data files
    bus_file_list = generate_monthly_filenames(year, "all", "bus_data", "csv")

    # Read the list of generation data files
    gen_file_list = list_filenames_in_directory(dirpath_gen, False)

    # Create a list of delta (f2-f1)  data files
    delta_file_list = generate_monthly_filenames(year, "all", "delta_f1f2_md", "csv")

    # Create a list of demand data files
    demand_file_list = generate_monthly_filenames(year, "all", "demand_md", "csv")



    # Now compute the demand for each month
    for i in np.arange(0, 12):

        f2_list = df_f2[i].columns[1:]
        f1_list = df_f2[i].columns[1:]

        # Find the sites that participate in the grid energy exchange, i.e.
        # the union of import and export sites. Note that each month may have a
        # a different number of participating sites.  Some sites may be
        # dormant in one month and activated in another. New sites could also
        # be added.
        f1f2_sites = list(set(list(f1_list) + list(f2_list)))
        f1f2_sites.sort()
        # print(f"{i+1} : {len(f1f2_sites)}")


        # BUSSES : Acquire data pertainig to the participating buses from the master set
        df_filtered = extract_matching_buses(df_bus_master, f1f2_sites)

        # Check that busses have line connections
        csv_file_path = dirpath_static + "lines_data.csv"
        df_lines = pd.read_csv(csv_file_path)
        connected_buses, disconnected_buses = check_bus_connectivity(df_lines, f1f2_sites)

        # Write to file
        df_filtered.to_csv(dirpath_bus + bus_file_list[i], index=False)


        # GENERATION (monthly dispatch)
        file_path = dirpath_gen + gen_file_list[i]
        df_g = aggregate_columns_by_prefix(file_path)
        # -----------------------------------------------------------------------------
        # ANOMALIES : Join 'HWB' and 'WPG' and then drop 'WPG'
        #
        df_g['HWB'] = df_g['HWB'] + df_g['WPG']
        df_g = df_g.drop(columns = ['WPG'])

        if 'CML' in df_g.columns:
            print("CML has been found ~!!!!")
            df_g['CYD'] = df_g['CYD'] + df_g['CML']  # NOTE that CM only appears in Dec
            df_g = df_g.drop(columns = ['CML'])
        # -----------------------------------------------------------------------------


        # Write to file
        # Must create a new file list for the aggregated data  &&&&&&&&&& .....
        df_g.round(4).to_csv(dirpath_g + gen_file_list[i], index=False)

        # DEMAND
        # Comoute the difference between grid import and export energies (f1 - f2)
        df_delta, missing_sites, clipped_sites = subtract_site_data(df_f1[i], df_f2[i])
        df_delta.round(4).to_csv(dirpath_f1f2 + delta_file_list[i], index=False)

        df_d = add_matching_columns_with_timestamp(df_g, df_delta)
        df_d = replace_negatives_with_zero(df_d)
        #dem_list = df_d.columns[1:]
        df_d.to_csv(dirpath_demand + demand_file_list[i], index=False)














 #  ------------------------------------------------------------------------

    # BUSSES -  sites that participate in the grid energy exchange
    # => the union of import and export sites.
    # Unique set of sites that participate in the network. Some sites may be
    # dormant and not required for the analysis.
    # f1f2_sites = list(set(list(f1_list) + list(f2_list)))
    # f1f2_sites.sort()

#     # List of buses to extract
#     bus_names = f1f2_sites

#     # Read the master bus data set - all possible busses
#     csv_file_path = dirpath_static + "bus_data.csv"
#     df_bus_master = pd.read_csv(csv_file_path)

#     # Acquire data pertainig to the participating buses from the master set
#     df_filtered = extract_matching_buses(df_bus_master, bus_names)

#     # Check that busses have line connections
#     csv_file_path = dirpath_static + "lines_data.csv"
#     df_lines = pd.read_csv(csv_file_path)
#     connected_buses, disconnected_buses = check_bus_connectivity(df_lines, bus_names)

#     # Write to file
#     df_filtered.to_csv(dirpath_cx + '202401_buses.csv', index=False)


#     # GENERATION (monthly dispatch) - g
#     gen_file_list = list_filenames_in_directory(dirpath_gen, False)
#     csv_file_path = dirpath_gen + "202401_gen_md_cons.csv"
#     df_g = aggregate_columns_by_prefix(csv_file_path)

# # -----------------------------------------------------------------------------
#     # ANOMALIES : Join 'HWB' and 'WPG' and then drop 'WPG'
#     df_g['HWB'] = df_g['HWB'] + df_g['WPG']
#     df_g = df_g.drop(columns = ['WPG'])

#     if 'CML' in df_g.columns:
#         print("CML has been found ~!!!!")
#         df_g['CYD'] = df_g['CYD'] + df_g['CML']  # NOTE that CM only appears in Dec
#         df_g = df_g.drop(columns = ['CML'])
# # -----------------------------------------------------------------------------

    # gen_list = df_g.columns[1:]
    # df_g.to_csv(dirpath_cx + '202401_generation.csv', index=False)


    # DEMAND
    # Comoute the difference between grid import and export energies (f1 - f2)
    # df_delta, missing_sites, clipped_sites = subtract_site_data(df_f1, df_f2)
    # df_delta.to_csv(dirpath_f1f2 + '202401_delta.csv', index=False)
    # df_d = add_matching_columns_with_timestamp(df_g, df_delta)
    # df_d = replace_negatives_with_zero(df_d)
    # dem_list = df_d.columns[1:]
#     df_d = df_d.round(3)  # Maintain constant accuracy for ease of reading
#     df_d.to_csv(dirpath_cx + '202401_demand.csv', index=False)
