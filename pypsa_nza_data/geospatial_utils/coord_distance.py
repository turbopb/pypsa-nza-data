#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *****************************************************************************
#                                                                             *
#   coord_distance.py                                                         *
#                                                                             *
#   DESCRIPTION                                                               *
#   computes the straight line distance between two points on the earth's     *
#   surface uding the standard Haversine formulas.                            *
#                                                                             *
#                                                                             *
#   OVERVIEW                                                                  *
#   input data are the latitude and longitude coordinates for each point in   *
#   decimal degrees.  Output is the surface distance between the points in    *
#   kilometres, i.e, the curvature of the earth is taken into account.        *
#                                                                             *
#                                                                             *
#   DATA SOURCES                                                              *
#                                                                             *
#   UPDATE HISTORY                                                            *
#   Created on Thu Oct 24 09:16:03 2024                                      *
#   Author : Phillippe Bruneau                                                *
#                                                                             *
# *****************************************************************************

import numpy as np
import pandas as pd
import math

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Coordinate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance


def line_length(network):
    line_list = network.lines.index
    bus1_list = network.lines.loc[:,['bus1']]
    bus0_list = network.lines.loc[:,['bus0']]
    
    print(line_list)
    print(bus0_list)
    print(bus1_list)
    
    return length


if __name__ == '__main__':

    # Read in the lines and bus data
    p1 = "C://Users//Public//Documents//Thesis//analysis//PyPSA-NZA-master//"
    p2 = ".//meta_data//"        
    datapath = p1 + p2
    
    fn = "lines_dc_data.csv"
    filepath = datapath + fn
    df_lines = pd.read_csv(filepath)

    fn = "bus_dc_data.csv"
    filepath = datapath + fn
    df_bus = pd.read_csv(filepath)

    # Process each line ......
    b0 = df_lines[['bus0']].iloc[:,0].tolist() # Start bus
    b1 = df_lines[['bus1']].iloc[:,0].tolist() # Termination bus
    
    # Find the lat and longitude of the two busses defining the line
    length = []
    for idx in np.arange(len(b0)):
        lon1 = df_bus[df_bus['site'] == b0[idx]]['long'].item()
        lat1 = df_bus[df_bus['site'] == b0[idx]]['lat'].item()
        print(idx, lon1, lat1)
        
        lon2 = df_bus[df_bus['site'] == b1[idx]]['long'].item()
        lat2 = df_bus[df_bus['site'] == b1[idx]]['lat'].item()
        
        length.append(haversine(lat1, lon1, lat2, lon2))
    
    df_lines['length'] = length
    fn = "lines_dc_data.csv"
    filepath = datapath + fn
    df_lines.to_csv(filepath, index=False)
    

# -----------------------------------------------------------------------------
    # # Example of usage
    
    # lat1 = 52.2296756  # Latitude of Warsaw
    # lon1 = 21.0122287  # Longitude of Warsaw
    # lat2 = 41.8919300  # Latitude of Rome
    # lon2 = 12.5113300  # Longitude of Rome
    
    # # Calculate the distance
    # distance_km = haversine(lat1, lon1, lat2, lon2)
    
    # descriptor = "Distance between Warsaw and Rome"
    # str1 = "Latitude of Warsaw"
    # str2 = "Longitude of Warsaw"
    # str3 = "Longitude of Rome"
    # str4 = "Longitude of Rome"
    # print(f"{str1} {distance_km:.2f} km")
    # print(f"Distance: {distance_km:.2f} km")
    
    # Invercargill	INV	168.392518110513	-46.3927430946406
    # Manapouri	MAN	167.278032222251	-45.5207959984629
    
    # Invercargill (INV) -- Manapouri ()MAN
    # distance_km = haversine(168.392518110513, -46.3927430946406, 167.278032222251,-45.5207959984629)
    # print()
    # print("Invercargill (INV) -- Manapouri (MAN)")
    # print(f"Distance: {distance_km:.2f} km")
    
    # Alternative ....
    # n = df_bus[df_bus['site'] == 'MAN'].index.item() # The index in the bus frame where site = MAN 
    # df_bus.loc[n, 'lat']
    # df_bus.loc[n, 'long']
 