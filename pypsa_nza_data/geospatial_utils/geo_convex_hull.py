#!/usr/bin/env python
# coding: utf-8

# ****************************************************************************#
#                                                                             #
#   geo_convex_hull.py                                                        #
#                                                                             #
#   DESCRIPTION                                                               #
#   Computes the convex hull of a set of points using the 'SciPy' library,    #
#                                                                             #
#                                                                             #
#   WORKFLOW                                                                  #
#   1.  Define a set of 2D points using a NumPy array.                        #
#   2.  Compute the convex hull using`scipy.spatial.ConvexHull`.              #
#       The `hull.simplices` attribute provides pairs of indices representing #
#       the edges of the hull.                                                #
#   3.  Visualization using Matplotlib to plot the points and edges of the    #
#       convex hull. Points that are part of the convex hull are highlighted. #
#                                                                             #
#   INPUTS                                                                    #
#                                                                             #
#                                                                             #
#   OUTPUTS                                                                   #
#                                                                             #
#                                                                             #
#   UPDATE HISTORY                                                            #
#   Created on Wed Jan  8 13:15:49 2025                                       #
#   Author : Phillippe Bruneau                                                #
#                                                                             #
# *****************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


#  Function to compute the convex hull of a set of map coordinate points (given 
#  in latitude and longitude). It uses **SciPy** for the convex hull calculation 
#  and assumes the input points are in a Cartesian plane approximation (suitable 
#  for small areas). For larger areas or more precise results, you may consider geodesic calculations.
# Created on Wed Jan  8 13:15:49 2025

import numpy as np
from scipy.spatial import ConvexHull

def geo_compute_convex_hull(latitudes, longitudes):
    """
    Compute the convex hull of a set of map coordinate points (latitude and longitude).
    
    Parameters:
    latitudes (list or array-like): List of latitude coordinates.
    longitudes (list or array-like): List of longitude coordinates.

    Returns:
    ndarray: Array of coordinates representing the vertices of the convex hull.
    """
    if len(latitudes) != len(longitudes):
        raise ValueError("Latitudes and longitudes must have the same length.")
    
    # Combine latitudes and longitudes into a 2D array
    points = np.column_stack((longitudes, latitudes))
    
    # Compute the convex hull
    hull = ConvexHull(points)
    
    # Extract the coordinates of the convex hull vertices
    hull_vertices = points[hull.vertices]
    return hull_vertices

# Example usage
latitudes = [40.7128, 34.0522, 41.8781, 29.7604, 39.7392]
longitudes = [-74.0060, -118.2437, -87.6298, -95.3698, -104.9903]

convex_hull = geo_compute_convex_hull(latitudes, longitudes)
print("Convex Hull Coordinates:")
print(convex_hull)


latitudes = [-35.849032158872,
             -35.849032158872,
             -35.849032158872,
             -35.849032158872,
             -35.849032158872,
             -35.849032158872,
             -35.849032158872,
             -35.849032158872,
             -35.849032158872,
             -35.849032158872,
             -35.849032158872,
             -35.849032158872]

longitudes = [174.479885301631,
             174.479885301631,
             174.479885301631,
             174.479885301631,
             174.479885301631,
             174.479885301631,
             174.479885301631,
             174.479885301631,
             174.479885301631,
             174.479885301631,
             174.479885301631,
             174.479885301631]
convex_hull = geo_compute_convex_hull(latitudes, longitudes)
print("Convex Hull Coordinates:")
print(convex_hull)

### Explanation

# 1. **Input**:
#    - `latitudes`: A list of latitude values.
#    - `longitudes`: A list of longitude values.
#    - Both must have the same length, as each latitude corresponds to a specific longitude.

# 2. **Combining Points**:
#    - The latitude and longitude pairs are combined into a 2D array using `np.column_stack()`.

# 3. **Convex Hull Computation**:
#    - The `ConvexHull` function from `scipy.spatial` computes the convex hull for the points.

# 4. **Output**:
#    - The function returns the vertices of the convex hull as an array of `[longitude, latitude]` pairs.

# ---

# ### Example Output
# For the given `latitudes` and `longitudes`, the output might look like:
# ```
# Convex Hull Coordinates:
# [[-118.2437   34.0522]
#  [-104.9903   39.7392]
#  [ -74.006    40.7128]
#  [ -87.6298   41.8781]
#  [ -95.3698   29.7604]]
# ```

# ---

# ### Notes on Geodesic Accuracy
# - For **large areas** or global datasets, the curvature of the Earth may need to be accounted for.
# - Use geospatial libraries like **Shapely**, **Geopandas**, or **Pyproj** for geodesic hulls. Alternatively, project the coordinates (e.g., using UTM) before computing the convex hull.
# - The code above assumes the Earth's curvature has negligible impact (small geographic areas).