#!/usr/bin/env python
# coding: utf-8

# ****************************************************************************#
#                                                                             #
#   compute_geodesic_convex_hull.py                                           #
#                                                                             #
#   DESCRIPTION                                                               #
#   Computes the convex hull of a set of map coordinate points given in       #
#   latitude and longitude while accounting for the curvature of the Earth,   #
#   you can project the coordinates onto a 2D plane using a geodetic          #
#   projection system such as the Universal Transverse Mercator (UTM) or      #
#   similar. The `pyproj` library is helpful for this transformation.         #
#                                                                             #
#                                                                             #
#   WORKFLOW                                                                  #
#   1. Projection:                                                            #
#   The geographic coordinates (latitude, longitude) are transformed into     #
#   a planar Cartesian coordinate system (e.g., Web Mercator or UTM) using    # 
#   'pyproj.Transformer`.                                                     #
#                                                                             #
#   2. Convex Hull:                                                           #
#   The `scipy.spatial.ConvexHull` function computes the convex hull in the   #
#   projected 2D space.                                                       #
#                                                                             #
#   3. Inverse Transformation:                                                #
#   The convex hull vertices in the projected space are transformed back to   #
#   latitude and longitude using the inverse projection.                      #
#                                                                             #
#                                                                             #
#   INPUTS                                                                    #
#   Map coordinate points given in latitude and longitude.                    #
#                                                                             #
#   OUTPUTS                                                                   #
#   Returns an array of `[longitude, latitude]` pairs representing the convex #
#   hull in geographic coordinates.                                           #
#                                                                             #
#   Notes:                                                                    #
#   The projection system used in this example is 'Web Mercator' (EPSG:3857). #
#   Depending on your region of interest, you may use a different projection  #
#   system, such as UTM zones for better accuracy.                            #
#                                                                             #
#   For very large datasets or global computations, specialized tools like    #
#   "Shapely" or "Geopandas" can also be used to handle geospatial data more  #
#   efficiently.                                                              #
#                                                                             #
#                                                                             #
#   LIBRARIES                                                                 #
#   numpy, scipy, pyproj                                                      #
#                                                                             #
#                                                                             #
#   UPDATE HISTORY                                                            #
#   Created on Wed Jan  8 17:12:56 2025                                       #
#   Author : Phillippe Bruneau                                                #
#                                                                             #
# *****************************************************************************

import numpy as np
from scipy.spatial import ConvexHull
from pyproj import Proj, Transformer

def compute_geodesic_convex_hull(latitudes, longitudes):
    """
    Compute the geodesic convex hull of a set of latitude and longitude points,
    accounting for the Earth's curvature.

    Parameters:
    latitudes (list or array-like): List of latitude coordinates.
    longitudes (list or array-like): List of longitude coordinates.

    Returns:
    ndarray: Array of [longitude, latitude] pairs representing the vertices of 
    the convex hull.
    """
    if len(latitudes) != len(longitudes):
        raise ValueError("Latitudes and longitudes must have the same length.")

    # Convert latitudes and longitudes to a 2D Cartesian coordinate system (e.g., UTM)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)  # WGS84 to Web Mercator
    projected_points = np.array([
        transformer.transform(lon, lat) for lon, lat in zip(longitudes, latitudes)
    ])

    # Compute the convex hull in the projected Cartesian space
    hull = ConvexHull(projected_points)

    # Extract the convex hull vertices
    hull_vertices_projected = projected_points[hull.vertices]

    # Transform the convex hull vertices back to latitude and longitude
    inverse_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    hull_vertices_geodesic = np.array([
        inverse_transformer.transform(x, y) for x, y in hull_vertices_projected
    ])

    return hull_vertices_geodesic


if __name__ == "__main__":
    
    # Example usage
    latitudes = [40.7128, 34.0522, 41.8781, 29.7604, 39.7392]
    longitudes = [-74.0060, -118.2437, -87.6298, -95.3698, -104.9903]
    
    convex_hull = compute_geodesic_convex_hull(latitudes, longitudes)
    print("Geodesic Convex Hull Coordinates:")
    print(convex_hull)



