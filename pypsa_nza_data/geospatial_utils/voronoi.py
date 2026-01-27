# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:24:23 2024

@author: OEM
"""

# Reference
#  https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

from scipy.spatial import Delaunay
import matplotlib.path as mpltPath
from time import time
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

# Scipy tools
print('Scipy tools')

# Define data points
points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                   [2, 0], [2, 1], [2, 2]])

# Compute Voronoi points
vor = Voronoi(points)

# Plot
fig = voronoi_plot_2d(vor)
plt.show()

f = np.random.rand(10, 2)
f_vor = Voronoi(f)
fig2 = voronoi_plot_2d(f_vor)
plt.show()

# Check if a point is within a polygon
point = Point(0.5, 0.5)
polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
print(polygon.contains(point))


# Compare ray tracing and matplotllb
print('Compare ray tracing and matplotllb')

# regular polygon for testing
lenpoly = 100
polygon = [[np.sin(x)+0.5, np.cos(x)+0.5]
           for x in np.linspace(0, 2*np.pi, lenpoly)[:-1]]

# random points set of points to test
N = 10000
points = np.random.rand(N, 2)


# Ray tracing
def ray_tracing_method(x, y, poly):

    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


start_time = time()
inside1 = [ray_tracing_method(point[0], point[1], polygon) for point in points]
print("Ray Tracing Elapsed time: " + str(time()-start_time))

# Matplotlib mplPath
start_time = time()
path = mpltPath.Path(polygon)
inside2 = path.contains_points(points)
print("Matplotlib contains_points Elapsed time: " + str(time()-start_time))


# Delaunay tessellation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
#points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
points = points = np.random.rand(100, 2)

tri = Delaunay(points)

# We can plot it:

fig3 = plt.figure()
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
plt.plot(points[:, 0], points[:, 1], 'o')

plt.show()
