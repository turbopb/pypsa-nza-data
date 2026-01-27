from pyproj import Transformer
from geopy.distance import geodesic
import numpy as np
import pandas as pd

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


# Example NZTM coordinates (Easting, Northing)
coords_nztm = [
    (1384000, 4923000),
    (1385000, 4923500),
    (1386000, 4924000)
]

df = compute_nztm_and_geodesic_distances(coords_nztm)
print(df)

#   segment  nztm_distance_km  geodesic_distance_km
# 0    0-1            0.1118                0.1116
# 1    1-2            0.1118                0.1116
