#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *****************************************************************************
#                                                                             *
#   nza_load_dynamic_data_from_url.py                                         *
#                                                                             *
#   DESCRIPTION                                                               *
#   MODULE : Function to download CSV files from the "Electricity Authority's *
#   web dataset using specified year and month ranges as parameters.          *
#                                                                             *
#   METHOD                                                                    *
#   Function Parameters:                                                      *
#       - `base_url`: The base URL where the dataset files are hosted.        *
#       - `save_dir`: The local directory wherein to save the downloaded      *
#       -             files
#       - `years`: A list of years to download. If `None`, all available      *
#          are downloaded.                                                    *
#       - `months`: A list of months (1-12) to download. If `None`, all       *
#          months are considered.                                             *
#                                                                             *
#   Usage Example:                                                            *
#   To download data for May and June of 2023 and 2024:                       *
#   download_grid_export_data(base_url, save_directory, years=[2023, 2024],   *
#                               months=[5, 6])                                *
#                                                                             *
#   Notes:                                                                    *
#   - The function checks if the specified `save_dir` exists and creates it   *
#     if necessary.                                                           *
#   - It loops over the specified years and months, constructing the          *
#       appropriate file names and URLs.                                      *
#   - If a file is not found (HTTP 404), it prints a message and continues.   *
#       Other errors are also caught and reported.                            *
#   - Adjust the `all_years` range in the function to match the actual        *
#       available data years.                                                 *
#                                                                             *
#   Ensure you have the `requests` library installed before running.          *
#                                                                             *
#   UPDATE HISTORY                                                            *
#   Created on Mon Mar 31 21:37:23 2025                                       *
#   Author : Phillippe Bruneau                                                *
#                                                                             *
# *****************************************************************************

import os
import requests
from itertools import product
import calendar
import yaml

ROOT_DIR = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/"

def parse_months_to_numbers(month_list):
    """
    Convert a list of month names.

    Converts a list of month names (abbreviated or full) to corresponding
    month numbers. Recognizes 'all' (in any case) to return a full list [1..12].

    Parameters:
        month_list (list of str): Input month strings.

    Returns:
        list of int: List of corresponding month numbers.
    """
    # Normalize to lowercase
    normalized = [m.lower().strip() for m in month_list]

    # If 'all' is present, return full year
    if 'all' in normalized:
        return list(range(1, 13))

    # Create lookup dict from month names
    month_lookup = {}
    for i in range(1, 13):
        full = calendar.month_name[i].lower()
        abbr = calendar.month_abbr[i].lower()
        month_lookup[full] = i
        month_lookup[abbr] = i

    # Match each input month to its number
    result = []
    for item in normalized:
        for key in month_lookup:
            if key in item:
                result.append(month_lookup[key])
                break  # Stop at the first match
        else:
            raise ValueError(f"Unrecognized month name: {item}")

    return sorted(set(result))  # Remove duplicates and sort

# if __name__== "__main__":

# Example usage
# months = ['jan', 'FEB', 'Oct']
# print(parse_months_to_numbers(months))  # Output: [1, 2, 10]

# Example Cases
# print(parse_months_to_numbers(['all']) )          #  [1, 2, ..., 12]
# print(parse_months_to_numbers(['Jan', 'march']))  #  [1, 3]
# print(parse_months_to_numbers(['May', 'JULY']) )  #  [5, 7]
# print(parse_months_to_numbers(['Sept'])        )  #  [9]


def grid_data_from_url(base_url:str, file:str, save_dir:str, years:list = None,
    months=None)->None:
    """
    Download CSV files from the EA dataset.

    The user can specify year and month ranges.

    Parameters
    ----------
    - base_url (str): Base URL of the dataset.
    - file (str) : file name
    - save_dir (str): Directory to save downloaded files.
    - years (list of int or None): List of years to download (e.g., [2023, 2024]).
      If None, all available years will be considered.
    - months (list of int or None): List of months to download (1-12).
      If None, all months will be considered.

    Returns
    -------
    - None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define all possible years and months if not specified
    all_years = range(2020, 2026)  # Adjust based on available data
    all_months = range(1, 12)

    years = years if years is not None else all_years
    months = months if months is not None else all_months

    for year, month in product(years, months):
        file_name = f"{year}{month:02d}_{file}"
        #file_name = f"{year}{month:02d}_Generation_MD.csv"
        file_url = f"{base_url}/{file_name}"
        local_path = os.path.join(save_dir, file_name)

        try:
            response = requests.get(file_url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {file_name}")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                print(f"File not found: {file_name}")
            else:
                print(f"Failed to download {file_name}: {e}")
        except Exception as e:
            print(f"An error occurred while downloading {file_name}: {e}")


def get_grid_file_specs(grid_type: str)-> dict:

    match grid_type:
        case "import":
            url_prefix = "Metered_data/Grid_import"
            filename = "Grid_import.csv"
        case "export":
            url_prefix = "Metered_data/Grid_export"
            filename = "Grid_export.csv"
        case "gen":
            url_prefix = "Generation/Generation_MD"
            filename = "Generation_MD.csv"
        case "hvdc":
            url_prefix = "Metered_data/HVDC_Flows"
            filename = "HVDC_flows.csv"
        case "react":
            url_prefix = "Metered_data/Reactive_power"
            filename = "Reactive_power.csv"
        case None:
            print("NO data type specified - must be one of import, export, gen, hvdc, or react")
        case _:
            print("Value grid_energy_flow is unknown - must be one of import, export, gen, hvdc, or react")

    grid_spec = {"url_prefix":url_prefix, "filename": filename}

    return grid_spec



if __name__ == "__main__":
    year_list = [2024]
    grid_list =["import", "export", "gen", "hvdc", "react"]
    months = parse_months_to_numbers(['all'])

    for grid_type in grid_list:
        grid_specs = get_grid_file_specs(grid_type)
        url_prefix = grid_specs["url_prefix"]
        filename = grid_specs["filename"]
        base_url = f"https://www.emi.ea.govt.nz/Wholesale/Datasets/{url_prefix}"

        # Process each year in the prescribed list
        for year in year_list:
            save_directory = ROOT_DIR + f"data/external/{year}/{grid_type}"
            grid_data_from_url(base_url, filename, save_directory, [year], months)






