#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *****************************************************************************
#                                                                             *
#   line_lengths.py                                                           *
#                                                                             *
#   DESCRIPTION                                                               *
#   Computes the lengths of the NZ electrical transmission lines.             *
#                                                                             *
#   OVERVIEW                                                                  *
#   Geospatial location data provided by Transpower is read from a csv file,  *
#   'Structures.csv' that containes NZTM coordinate points defining all power * 
#   lines in the NZ electical grid system.                                    *
#                                                                             *
#   The headings in the csv file are : 'OBJECTID', 'MXLOCATION', 'LongType',  *
#   'status', 'description', 'loctype', 'x', 'y','GlobalID'. For the purposes *
#   of line length computations, only the 'description' (or 'MXLOCATION'),    *
#   'x', and 'y' data are of interest.                                        *
#                                                                             *
#   String data under the 'description' heading denotes the line designation  *
#   and is of the form 'INV-MAN-A-0329-Structure'.  The first part 'XXX-YYY'  #
#   ('INV-MAN' in the given example) indicates the start and endpoints of the * 
#   line, where the acronymns 'INV' and MAN' represent Invercargill and       *
#   Manapouri respectively.  The second  is usually 'A', 'B' or possibly some *
#   other single capital letter. The third part is a 4 digit numeric (string) *
#   identifier such as '0329'.  Numeric strings such as '0241A' are not not   *
#   considered or of interest. The last part of the string is 'Structure',    *
#   common to all headings.                                                   *
#                                                                             *
#   The 'x', and 'y' columns are the Eastings and Northings for coordinate    *
#   points in CRS EPSG:2193 - NZGD2000 / New Zealand Transverse Mercator 2000.* 
#                                                                             *
#   WORKFLOW                                                                  *
#   Extracts the all 'x', 'y' data for a given line which is defined by the   *
#   start and and points specified in the parameter list. For example 
#       start point : 'INV-MAN-A-0001-Structure' 
#       end point :'INV-MAN-A-0355-Structure'. 
#   The new name of  the line is also specified in the function call.  
#   The function should call other 
#   functions that compute the total distance between points that define the 
#   line, 
#   i.e., total line length returned in km units and the equivalent coordinates
#   in WGS84 (lat and long).  
#                                                                             *
#   Error handling is crucial for making your code robust and user-friendly.  *
#   refined version of the function set with error handling added at key steps:
#                                                                             *
#   Error Handling Strategy                                                   *
#   1. Validation of input data (e.g., checking for existance of the CSV and  *
#     if required columns are present).                                       *
#   2. Catch and report user errors gracefully (e.g., if start or end         *
#      descriptions are missing).                                             *
#   3. Handle geospatial transformation issues with fallbacks or clear        *
#       errors.                                                               #
#   4. Use informative error messages to guide debugging.                     #
#                                                                             *
#   OVERVIEW                                                                  *
#                                                                             *
#                                                                             *
#   DATA SOURCES                                                              *
#                                                                             *
#   UPDATE HISTORY                                                            *
#   Created on Mon July 01 14:13:38 2024                                      *
#   Author : Phillippe Bruneau                                                *
#                                                                             *
# *****************************************************************************

# import os
from geopy.distance import geodesic
from pyproj import Transformer
import pandas as pd
import numpy as np
import yaml

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
    
    
def read_csv_file(filename: str, nrows: int = None):
    """
    Read a CSV file with optional row limit anderror handling.

    Parameters
    ----------
        filename (str): Path to the CSV file.
        nrows (int, optional): Number of rows to read. Reads all if None.

    Returns
    -------
        pd.DataFrame or None: The resulting DataFrame, or None if error.
    """
    try:
        # Validate nrows if given
        if nrows is not None:
            if not isinstance(nrows, int) or nrows <= 0:
                raise ValueError("!!! nrows must be a positive integer.")

        # Try UTF-8, fallback to UTF-8-SIG or ISO-8859-1 if decoding fails
        try:
            df = pd.read_csv(filename, nrows=nrows)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filename, nrows=nrows, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(filename, nrows=nrows, encoding='ISO-8859-1')

        print(f"*** Successfully read {len(df)} rows from {filename}")
        return df

    except FileNotFoundError:
        print(f"? File not found: {filename}")
    except pd.errors.EmptyDataError:
        print(f"? The file is empty or malformed: {filename}")
    except pd.errors.ParserError as e:
        print(f"? Parsing error: {e}")
    except ValueError as e:
        print(f"? Value error: {e}")
    except Exception as e:
        print(f"? Unexpected error: {e}")

    return None


def compute_cumulative_distance(df: pd.DataFrame, x_col: str = 'x', y_col: str = 'y') -> pd.Series:
    """
    Compute cumulative Euclidean distance between successive (x, y) points.
    
    Missing values are handled. 

    Parameters
    ----------
    - df: pandas DataFrame containing x and y columns
    - x_col: column name for x-coordinates
    - y_col: column name for y-coordinates

    Returns
    -------
    - A pandas Series of the same length as df with cumulative distances.
    """
    # Extract and clean coordinates
    valid_mask = df[[x_col, y_col]].notnull().all(axis=1)
    clean_df = df.loc[valid_mask].copy()
    x = clean_df[x_col].to_numpy()
    y = clean_df[y_col].to_numpy()

    # Compute deltas and distances
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    segment_distances = np.hypot(dx, dy)

    # Compute cumulative distance and align with original DataFrame
    cumulative_distance = np.insert(np.cumsum(segment_distances), 0, 0.0)
    result_series = pd.Series(index=clean_df.index, data=cumulative_distance)

    # Reindex to match original DataFrame shape, fill missing values with NaN
    full_series = pd.Series(index=df.index, dtype='float64')
    full_series.loc[result_series.index] = result_series

    return full_series


def generate_structure_names(start_desc, end_desc):
    """
    Generate a list of structure names from start_desc to end_desc (inclusive).

    Args
    ----
        start_desc (str): e.g., "INV-MAN-A-0001-Structure"
        end_desc (str): e.g., "INV-MAN-A-0355-Structure"

    Returns
    -------
        list of str: structure names from start to end.
    """
    try:
        # Extract prefix and numeric part
        start_prefix, start_number = start_desc.rsplit('-', 2)[0], start_desc.rsplit('-', 2)[1]
        end_prefix, end_number = end_desc.rsplit('-', 2)[0], end_desc.rsplit('-', 2)[1]

        # Ensure the prefixes match
        if start_prefix != end_prefix:
            raise ValueError("Start and end descriptions must have the same prefix")

        # Parse numbers
        start_num = int(start_number)
        end_num = int(end_number)

        if start_num > end_num:
            raise ValueError("Start number must be less than or equal to end number")

        # Rebuild full structure names
        structure_names = [
            f"{start_prefix}-{str(n).zfill(4)}-Structure" for n in range(start_num, end_num + 1)
        ]

        return structure_names

    except Exception as e:
        print(f"Error: {e}")
        return []


def compute_nztm_and_geodesic_distances(coord_pairs_nztm):
    """
    Given a list of (x, y) pairs in NZTM (EPSG:2193), compute:
    - Euclidean distance in projected NZTM space
    - Geodesic (WGS84) distance accounting for Earth's curvature

    Returns:
        DataFrame with:
        - segment index
        - NZTM distance (km)
        - Geodesic distance (km)
    """
    if len(coord_pairs_nztm) < 2:
        raise ValueError("At least two coordinate pairs are required.")

    # Setup transformer from NZTM to WGS84
    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)

    results = []
    for i in range(len(coord_pairs_nztm) - 1):
        x1, y1 = coord_pairs_nztm[i]
        x2, y2 = coord_pairs_nztm[i + 1]

        # NZTM Euclidean distance
        dist_nztm_km = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 1000

        # Convert to WGS84
        lon1, lat1 = transformer.transform(x1, y1)
        lon2, lat2 = transformer.transform(x2, y2)

        # Geodesic distance in km
        dist_geo_km = geodesic((lat1, lon1), (lat2, lon2)).kilometers

        results.append({
            "segment": f"{i}-{i+1}",
            "nztm_distance_km": dist_nztm_km,
            "geodesic_distance_km": dist_geo_km
        })

    return pd.DataFrame(results)


def line_of_sight_dist(P):
    N = len(P)
    #print(N)
    dx = P[-1][0] - P[0][0]
    dy = P[-1][1] - P[0][1]
    
    return np.hypot(dx, dy)/1000 
    #return np.sqrt(dx**2 +dy**2)/1000

def compute_distance(df_data, df_spec):
    start_list = df_spec["LN0"] # Extract the START points as a data list
    end_list = df_spec["LN1"]   # Extract the END points as a data list
    
    # Initialise the length vecctors for line length computed by three 
    # different apprroaches
    d_los_vec = []      # line of sight between end points, projected 
    d_proj_vec = []     # point to point integrated, projected
    d_curv_vec = []     # point to point integrated, geodesic (eartn curvature)
    
    for s, e in zip(start_list, end_list):
        # Creates a list of names for the points on the current line by 
        # incrementing the start point numeric identifier from 1 to n, where   
        # n = total number of points in the list.
        point_list = generate_structure_names(s, e) 
        
        # Extract the sub-dataframe from the master dataframe using the points 
        # in "point_list", i.e.only those points which are on the the current 
        # line.
        df_points = df_data[df_data['description'].isin(point_list)]
        
        # Form coordinate pairs for each point on the line
        coord_pairs_nztm = list(zip(df_points['x'], df_points['y']))
        #print(coords)
        #coord_pairs_nztm = coords
        
        # Compute a lists of distances between each point on the line for 
        # projected and geodesic methods.
        df_dist = compute_nztm_and_geodesic_distances(coord_pairs_nztm)
        d_seg_proj = df_dist['nztm_distance_km']        # projected
        d_seg_curv = df_dist['geodesic_distance_km']    # geodesic
                
        # Sum up the points to give an overall'distance' scale to km (/1000m)
        d_proj = d_seg_proj.sum()
        d_curv = d_seg_curv.sum()
        
        # Line of sight distance is just the differernce between the start and 
        # end points
        d_los = line_of_sight_dist(coord_pairs_nztm)
        
        # Collect the different distances into lists - one value for each line
        d_los_vec.append(d_los)
        d_proj_vec.append(d_proj)
        d_curv_vec.append(d_curv)
        
    return  d_los_vec, d_proj_vec, d_curv_vec


def pad_to_length(lst, length):
    return lst + [np.nan] * (length - len(lst))


def write_lines_file(dl, dp, dg, lines_file, update_file):
    # df_spec['d_los'] = dl
    # df_spec['d_proj'] = dp
    # df_spec['distance'] = dg
   
    df = read_csv_file(lines_file)
        
    df['length'] = pad_to_length(dl, len(df))
    df['projected'] = pad_to_length(dp, len(df))
    df['distance'] = pad_to_length(dg, len(df))

    df.to_csv(update_file)


def main():
    print("TRANSMISSION LINE LENGTHS")


if __name__ == '__main__':
    main()
    
    # Read in the master data file with ALL data points - Transpower supplied
    data_file = "Structures.csv"   
    df_data = read_csv_file(data_file)
    
    # Read start and end points for the line SEGMENTS of interest. These are 
    # manually derived by finding the start and end points between POCs.
    spec_file = "lines_data_abs.csv" 
    df_spec = read_csv_file(spec_file, 8)
    
    # Calculate the distance (length) for each line SEGMENT
    dl, dp, dg = compute_distance(df_data, df_spec)
    
    # Write the update line lengths to csv file
    lines_file = "lines_data_abs.csv" 
    update_file = "ld_abs.csv"
    
    write_lines_file(dl, dp, dg, lines_file, update_file)
 

