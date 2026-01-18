#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *****************************************************************************
#                                                                             *
#   nza_load_gen_fleet_data_from_url.py                                       *
#                                                                             *
#   DESCRIPTION                                                               *
#   Downloads the generator fleet data CSV files from the "Electricity        *
#   Authority's" url.                                                         *
#                                                                             *
#   METHOD                                                                    *
#   Function Parameters:                                                      *
#       - `base_url`: The base URL where the dataset files are hosted.        *
#       - `save_dir`: The local directory wherein to save the downloaded      *
#       -             files                                                   *
#       - `file     : source file name on EA web-site.                        *
#                                                                             *
#   The function checks if the specified `save_dir` exists and creates it     *
#   if necessary.                                                             *
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

# from itertools import product
# import calendar
# import yaml

ROOT_DIR = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/"


def gen_data_from_url(base_url:str, file:str, save_dir:str)->None:
    """
    Download CSV files from the EA dataset.

    The user can specify year and month ranges.

    Parameters
    ----------
    - base_url (str): Base URL of the dataset.
    - file (str) : source file name on EA web-site.
    - save_dir (str): Directory to save downloaded files.

    Returns
    -------
    - None
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_url = f"{base_url}/{file}"
    local_path = os.path.join(save_dir, file)

    try:
        response = requests.get(file_url)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file} to  Directory :...\n {save_dir}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"File not found: {file}")
        else:
            print(f"Failed to download {file}: {e}")
    except Exception as e:
        print(f"An error occurred while downloading {file}: {e}")



if __name__ == "__main__":

    url_prefix = "Generation/GenerationFleet/Existing"
    base_url = f"https://www.emi.ea.govt.nz/Wholesale/Datasets/{url_prefix}"
    filename = "20250917_DispatchedGenerationPlant.csv"

    # Process
    save_directory = ROOT_DIR + "data/external/static"
    gen_data_from_url(base_url, filename, save_directory)



