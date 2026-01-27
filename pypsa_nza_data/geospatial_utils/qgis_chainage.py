# QGUS CHAINAGE
### Chainage in GIS Context

# **Chainage** is a term commonly used in civil engineering, surveying, and 
# GIS to denote the cumulative distance along a linear feature, such as a 
# road, railway, pipeline, or river. It essentially represents a way to 
# measure distance along a line from a defined starting point, often used for 
# referencing specific locations along the linear feature.
# For example, a chainage of **100 meters** indicates a point 100 meters from 
# the starting point along the line.

# ---

# ### Uses of Chainage in GIS

# 1. **Infrastructure Design and Maintenance**:
#   Identify locations along roads, pipelines, or railway tracks for 
#   maintenance or monitoring.
#   Reference design elements like culverts, bridges, or intersections.

# 2. **Asset Management**:
#   Mark utility poles, signage, or other assets along a linear network.

# 3. **Hydrology and Environmental Studies**:
#   Measure distances along rivers or streams for flood modeling, sediment 
#   transport, or monitoring.

# 4. **Project Management**:
#   Communicate progress or issues by referencing chainage.

# ---

# ### How to Use Chainage in GIS

# 1. **Compute Chainage for a Line**:
#   Calculate the cumulative distance along a line geometry from its starting 
#   point.

# 2. **Create Chainage Points**:
#   Generate evenly spaced points along the line to represent chainages at 
#   specific intervals (e.g., every 10 meters).

# 3. **Label Features by Chainage**:
#    - Add labels or markers to indicate chainage values along the line.

# ### Steps to Work with Chainage in QGIS Using Python
# #### 1. **Compute and Display Chainage Along a Line**
# Here’s how you can compute chainage for a line and create chainage labels:


from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsPoint,
    QgsGeometry
)

# Function to compute chainage points
def compute_chainage(layer, interval):
    """
    Computes chainage points at regular intervals along a line layer.
    
    Parameters:
        layer (QgsVectorLayer): Line layer.
        interval (float): Interval distance for chainage points.
    
    Returns:
        list of QgsPoint: List of chainage points.
    """
    chainage_points = []
    for feature in layer.getFeatures():
        geometry = feature.geometry()
        if not geometry.isMultipart():
            length = geometry.length()
            current_distance = 0
            
            while current_distance <= length:
                point = geometry.interpolate(current_distance)  # Interpolate point at distance
                chainage_points.append((current_distance, point.asPoint()))
                current_distance += interval
    
    return chainage_points

# Load a line layer
layer_path = "/path/to/line/layer.shp"
layer = QgsVectorLayer(layer_path, "Line Layer", "ogr")

if not layer.isValid():
    print("Failed to load the layer.")
else:
    QgsProject.instance().addMapLayer(layer)
    print("Layer added.")

    # Compute chainage points
    interval = 100  # Chainage interval in map units
    chainages = compute_chainage(layer, interval)
    
    # Print chainage points
    for dist, point in chainages:
        print(f"Chainage {dist:.2f} units: ({point.x()}, {point.y()})")

# -----------------------------------------------------------------------------
#### 2. **Create Chainage Points Layer**
#To create a new point layer representing chainage locations:

from qgis.core import (
    QgsVectorLayer,
    QgsVectorFileWriter,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsPoint,
    QgsGeometry
)
from qgis.PyQt.QtCore import QVariant

# Create an output point layer for chainage points
output_path = "/path/to/output/chainage_points.shp"
fields = QgsFields()
fields.append(QgsField("Chainage", QVariant.Double))

writer = QgsVectorFileWriter(
    output_path,
    "UTF-8",
    fields,
    QgsWkbTypes.Point,
    layer.crs(),
    "ESRI Shapefile"
)

# Write chainage points to the new layer
if writer.hasError() == QgsVectorFileWriter.NoError:
    for dist, point in chainages:
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(point))
        feature.setAttributes([dist])
        writer.addFeature(feature)
    del writer
    print(f"Chainage points saved to {output_path}.")
else:
    print("Error creating output file.")


### Example Output

# - **Chainage Points**:
#   - Chainage 0: (X1, Y1)
#   - Chainage 100: (X2, Y2)
#   - Chainage 200: (X3, Y3)

# - **New Layer**:
#   - A point shapefile with attributes:
#     - `Chainage`: Cumulative distance along the line.
#     - Geometry: Location of each chainage point.

# ---

# ### Visualization in QGIS

# 1. **Load Chainage Points**:
#    - Import the generated shapefile into QGIS.

# 2. **Style Chainage Points**:
#    - Use labels to display the chainage values.

# 3. **Customize Chainage Display**:
#    - Adjust the interval or format to match project requirements.

### Advanced Applications

# - **Chainage with Attributes**:
#    - Combine chainage with other attributes like elevation, road names, or maintenance records.

# - **Dynamic Chainage Labels**:
#    - Use **Expressions** in QGIS to dynamically calculate and display chainage without creating new layers.

# - **Multi-Line Features**:
#    - Adapt scripts to handle multipart geometries or networks with multiple connected lines.

# ---

# By calculating chainage, you can effectively reference, analyze, and visualize linear features in GIS.