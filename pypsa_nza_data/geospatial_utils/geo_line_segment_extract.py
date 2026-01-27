#!/usr/bin/env python3
"""
Geospatial Railway Line Distance Calculator

Enhanced version that handles UTM and geographic coordinate systems for
railway distance calculations. Supports coordinate transformations and
proper distance calculations for geospatial data.

Dependencies:
- pyproj: For coordinate transformations
- geopy: For geodesic distance calculations (optional)

Install with: pip install pyproj geopy

Author: Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
import sys
import csv
from pathlib import Path
from enum import Enum

# Geospatial libraries
try:
    from pyproj import Proj, transform, Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    print("Warning: pyproj not available. Install with: pip install pyproj")

try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("Note: geopy not available. Install with: pip install geopy for geodesic calculations")


class CoordinateSystem(Enum):
    """Supported coordinate systems."""
    CARTESIAN = "cartesian"  # Plain x,y coordinates
    UTM = "utm"             # UTM coordinates (meters)
    GEOGRAPHIC = "geographic"  # Latitude/Longitude (degrees)


@dataclass
class Point:
    """Represents a 2D point with coordinate system awareness."""
    x: float
    y: float
    coord_system: CoordinateSystem = CoordinateSystem.CARTESIAN
    utm_zone: Optional[int] = None
    utm_hemisphere: Optional[str] = None  # 'N' or 'S'

    def __post_init__(self):
        """Convert coordinates to float for consistency."""
        self.x = float(self.x)
        self.y = float(self.y)

    def to_array(self) -> np.ndarray:
        """Convert point to numpy array."""
        return np.array([self.x, self.y])

    def distance_to(self, other: 'Point') -> float:
        """
        Calculate distance to another point.
        Uses appropriate method based on coordinate system.
        """
        if self.coord_system == CoordinateSystem.GEOGRAPHIC and GEOPY_AVAILABLE:
            # Use geodesic distance for lat/lon coordinates
            return geodesic((self.y, self.x), (other.y, other.x)).meters
        else:
            # Use Euclidean distance for Cartesian/UTM coordinates
            return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __str__(self) -> str:
        if self.coord_system == CoordinateSystem.UTM:
            return f"Point({self.x:.2f}E, {self.y:.2f}N, UTM{self.utm_zone}{self.utm_hemisphere})"
        elif self.coord_system == CoordinateSystem.GEOGRAPHIC:
            return f"Point({self.x:.6f}°, {self.y:.6f}°)"
        else:
            return f"Point({self.x:.6f}, {self.y:.6f})"


@dataclass
class LineSegmentInfo:
    """Information about a line segment between two consecutive points."""
    start_point: Point
    end_point: Point
    length: float
    cumulative_distance: float


class GeospatialRailwayCalculator:
    """
    Calculates distances along a railway line with geospatial coordinate support.
    """

    def __init__(self, line_points: List[Point], target_coord_system: CoordinateSystem = None):
        """
        Initialize with railway line points.

        Args:
            line_points: List of points defining the railway line
            target_coord_system: Target coordinate system for calculations
        """
        if len(line_points) < 2:
            raise ValueError("Railway line must have at least 2 points")

        self.original_points = line_points
        self.coord_system = line_points[0].coord_system

        # Convert to target coordinate system if specified
        if target_coord_system and target_coord_system != self.coord_system:
            self.line_points = self._convert_coordinate_system(line_points, target_coord_system)
        else:
            self.line_points = line_points

        self.segments = self._calculate_segments()
        self.total_length = self.segments[-1].cumulative_distance if self.segments else 0

    def _convert_coordinate_system(self, points: List[Point],
                                 target_system: CoordinateSystem) -> List[Point]:
        """Convert points between coordinate systems."""
        if not PYPROJ_AVAILABLE:
            raise RuntimeError("pyproj required for coordinate system conversion")

        converted_points = []

        for point in points:
            if point.coord_system == target_system:
                converted_points.append(point)
                continue

            # Define source and target CRS
            if point.coord_system == CoordinateSystem.GEOGRAPHIC:
                source_crs = CRS.from_epsg(4326)  # WGS84
            elif point.coord_system == CoordinateSystem.UTM:
                # Construct UTM EPSG code
                if point.utm_zone and point.utm_hemisphere:
                    if point.utm_hemisphere.upper() == 'N':
                        epsg_code = 32600 + point.utm_zone
                    else:
                        epsg_code = 32700 + point.utm_zone
                    source_crs = CRS.from_epsg(epsg_code)
                else:
                    raise ValueError("UTM zone and hemisphere required for UTM coordinates")
            else:
                # Assume local coordinate system - no conversion
                converted_points.append(point)
                continue

            if target_system == CoordinateSystem.GEOGRAPHIC:
                target_crs = CRS.from_epsg(4326)
            elif target_system == CoordinateSystem.UTM:
                # Use the same UTM zone as source for simplicity
                target_crs = source_crs
            else:
                # No conversion for Cartesian
                converted_points.append(point)
                continue

            # Perform transformation
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            new_x, new_y = transformer.transform(point.x, point.y)

            converted_point = Point(
                x=new_x,
                y=new_y,
                coord_system=target_system,
                utm_zone=point.utm_zone,
                utm_hemisphere=point.utm_hemisphere
            )
            converted_points.append(converted_point)

        return converted_points

    def _calculate_segments(self) -> List[LineSegmentInfo]:
        """Calculate information for each segment of the railway line."""
        segments = []
        cumulative_distance = 0

        for i in range(len(self.line_points) - 1):
            start_point = self.line_points[i]
            end_point = self.line_points[i + 1]
            segment_length = start_point.distance_to(end_point)
            cumulative_distance += segment_length

            segments.append(LineSegmentInfo(
                start_point=start_point,
                end_point=end_point,
                length=segment_length,
                cumulative_distance=cumulative_distance
            ))

        return segments

    def _find_closest_point_on_segment(self, point: Point, segment: LineSegmentInfo) -> Tuple[Point, float, float]:
        """Find the closest point on a line segment to a given point."""
        # For geographic coordinates, we need to be more careful about projections
        if point.coord_system == CoordinateSystem.GEOGRAPHIC:
            # For small distances, we can use a local approximation
            # For larger distances, should use proper geodesic projections
            pass

        # Vector from segment start to end
        segment_vector = segment.end_point.to_array() - segment.start_point.to_array()
        segment_length_sq = np.dot(segment_vector, segment_vector)

        if segment_length_sq == 0:
            closest_point = segment.start_point
            distance = point.distance_to(closest_point)
            return closest_point, 0.0, distance

        # Vector from segment start to point
        point_vector = point.to_array() - segment.start_point.to_array()

        # Project point onto segment
        t = np.dot(point_vector, segment_vector) / segment_length_sq
        t = max(0.0, min(1.0, t))

        # Calculate closest point
        closest_array = segment.start_point.to_array() + t * segment_vector
        closest_point = Point(
            x=closest_array[0],
            y=closest_array[1],
            coord_system=point.coord_system,
            utm_zone=point.utm_zone,
            utm_hemisphere=point.utm_hemisphere
        )

        distance = point.distance_to(closest_point)

        return closest_point, t, distance

    def find_closest_point_on_line(self, point: Point) -> Tuple[Point, float, int, float]:
        """Find the closest point on the entire railway line to a given point."""
        # Convert point to same coordinate system as line if needed
        if point.coord_system != self.line_points[0].coord_system:
            # For now, assume same coordinate system
            # In production, would need coordinate conversion
            pass

        min_distance = float('inf')
        best_point = None
        best_segment_index = -1
        best_t = 0

        for i, segment in enumerate(self.segments):
            closest_point, t, distance = self._find_closest_point_on_segment(point, segment)

            if distance < min_distance:
                min_distance = distance
                best_point = closest_point
                best_segment_index = i
                best_t = t

        # Calculate distance along the line
        distance_along_line = 0
        if best_segment_index > 0:
            distance_along_line = self.segments[best_segment_index - 1].cumulative_distance

        distance_along_line += best_t * self.segments[best_segment_index].length

        return best_point, distance_along_line, best_segment_index, min_distance

    def calculate_distance_between_points(self, point_c: Point, point_d: Point,
                                        tolerance: float) -> Tuple[Point, Point, float, float, float]:
        """Calculate the distance along the railway line between points C and D."""
        closest_c, dist_c_along, seg_c, dist_c_to_line = self.find_closest_point_on_line(point_c)
        closest_d, dist_d_along, seg_d, dist_d_to_line = self.find_closest_point_on_line(point_d)

        # Check distance tolerance
        if dist_c_to_line > tolerance:
            raise ValueError(
                f"Point C is {dist_c_to_line:.2f} units from railway line, "
                f"exceeding tolerance of {tolerance:.2f}"
            )

        if dist_d_to_line > tolerance:
            raise ValueError(
                f"Point D is {dist_d_to_line:.2f} units from railway line, "
                f"exceeding tolerance of {tolerance:.2f}"
            )

        distance_cd = abs(dist_d_along - dist_c_along)

        return closest_c, closest_d, distance_cd, dist_c_along, dist_d_along

    def get_line_info(self) -> dict:
        """Get information about the railway line."""
        units = "units"
        if self.line_points[0].coord_system == CoordinateSystem.UTM:
            units = "meters"
        elif self.line_points[0].coord_system == CoordinateSystem.GEOGRAPHIC:
            units = "meters (geodesic)"

        return {
            'total_points': len(self.line_points),
            'total_segments': len(self.segments),
            'total_length': self.total_length,
            'units': units,
            'coordinate_system': self.line_points[0].coord_system,
            'start_point': self.line_points[0],
            'end_point': self.line_points[-1]
        }


def read_geospatial_line_from_file(filename: str, coord_system: CoordinateSystem,
                                 utm_zone: int = None, utm_hemisphere: str = None) -> List[Point]:
    """
    Read railway line points from a CSV file with geospatial awareness.

    Expected CSV format:
    - For UTM: easting,northing or x,y
    - For Geographic: longitude,latitude or x,y
    - For Cartesian: x,y
    """
    points = []

    try:
        with open(filename, 'r', newline='') as csvfile:
            sniffer = csv.Sniffer()
            sample = csvfile.read(1024)
            csvfile.seek(0)
            has_header = sniffer.has_header(sample)

            reader = csv.reader(csvfile)

            if has_header:
                headers = next(reader)
                print(f"Detected headers: {headers}")

            for row_num, row in enumerate(reader, start=2 if has_header else 1):
                if len(row) < 2:
                    continue

                try:
                    x = float(row[0])
                    y = float(row[1])

                    point = Point(
                        x=x,
                        y=y,
                        coord_system=coord_system,
                        utm_zone=utm_zone,
                        utm_hemisphere=utm_hemisphere
                    )
                    points.append(point)

                except ValueError:
                    print(f"Warning: Row {row_num} contains non-numeric data, skipping")
                    continue

        if not points:
            raise ValueError("No valid points found in file")

        print(f"Successfully loaded {len(points)} points from {filename}")
        print(f"Coordinate system: {coord_system.value}")
        if coord_system == CoordinateSystem.UTM:
            print(f"UTM Zone: {utm_zone}{utm_hemisphere}")

        return points

    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found")
    except Exception as e:
        raise Exception(f"Error reading file {filename}: {e}")


def create_utm_sample_file(filename: str):
    """Create a sample UTM coordinate file."""
    # Sample data for Auckland, New Zealand (UTM Zone 60S)
    # These are approximate UTM coordinates
    base_easting = 1756000
    base_northing = 5918000

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['easting', 'northing'])

        # Create a curved path
        for i in range(20):
            t = i * 0.5
            easting = base_easting + t * 1000 + 200 * np.sin(t * 0.1)
            northing = base_northing + t * 500 + 100 * np.cos(t * 0.1)
            writer.writerow([f"{easting:.2f}", f"{northing:.2f}"])

    print(f"Created sample UTM file: {filename}")


def create_geographic_sample_file(filename: str):
    """Create a sample geographic coordinate file."""
    # Sample data around Auckland, New Zealand
    base_lon = 174.7
    base_lat = -36.8

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['longitude', 'latitude'])

        # Create a curved path
        for i in range(20):
            t = i * 0.001
            lon = base_lon + t + 0.0005 * np.sin(i * 0.5)
            lat = base_lat + t * 0.5 + 0.0002 * np.cos(i * 0.5)
            writer.writerow([f"{lon:.6f}", f"{lat:.6f}"])

    print(f"Created sample geographic file: {filename}")


def get_coordinate_system_input() -> Tuple[CoordinateSystem, int, str]:
    """Get coordinate system information from user."""
    print("Select coordinate system:")
    print("1. Cartesian (x, y)")
    print("2. UTM (easting, northing)")
    print("3. Geographic (longitude, latitude)")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        return CoordinateSystem.CARTESIAN, None, None
    elif choice == "2":
        utm_zone = int(input("Enter UTM zone (1-60): "))
        utm_hemisphere = input("Enter hemisphere (N/S): ").strip().upper()
        if utm_hemisphere not in ['N', 'S']:
            raise ValueError("Hemisphere must be 'N' or 'S'")
        return CoordinateSystem.UTM, utm_zone, utm_hemisphere
    elif choice == "3":
        if not GEOPY_AVAILABLE:
            print("Warning: geopy not available. Distance calculations may be less accurate.")
        return CoordinateSystem.GEOGRAPHIC, None, None
    else:
        raise ValueError("Invalid choice")


def main():
    """Main function for geospatial railway calculator."""
    print("Geospatial Railway Line Distance Calculator")
    print("=" * 50)

    if not PYPROJ_AVAILABLE:
        print("Note: Running without pyproj - coordinate transformations not available")

    print("Choose mode:")
    print("1. Run example with UTM coordinates")
    print("2. Run example with geographic coordinates")
    print("3. Load custom data")
    print("4. Create sample files")

    try:
        choice = input("Enter choice (1-4): ").strip()

        if choice == "4":
            print("Creating sample files...")
            create_utm_sample_file("sample_utm_railway.csv")
            create_geographic_sample_file("sample_geographic_railway.csv")
            print("Sample files created. Run program again to use them.")
            return

        elif choice == "1":
            # UTM example
            coord_system = CoordinateSystem.UTM
            utm_zone = 60
            utm_hemisphere = 'S'

            # Create sample UTM points (Auckland area)
            line_points = []
            base_e, base_n = 1756000, 5918000
            for i in range(15):
                t = i * 0.5
                e = base_e + t * 800 + 150 * np.sin(t * 0.1)
                n = base_n + t * 400 + 100 * np.cos(t * 0.1)
                line_points.append(Point(e, n, coord_system, utm_zone, utm_hemisphere))

            point_c = Point(1758000, 5920000, coord_system, utm_zone, utm_hemisphere)
            point_d = Point(1762000, 5924000, coord_system, utm_zone, utm_hemisphere)
            tolerance = 500  # 500 meters

        elif choice == "2":
            # Geographic example
            coord_system = CoordinateSystem.GEOGRAPHIC

            # Create sample geographic points (Auckland area)
            line_points = []
            base_lon, base_lat = 174.7, -36.8
            for i in range(15):
                t = i * 0.001
                lon = base_lon + t + 0.0003 * np.sin(i * 0.3)
                lat = base_lat + t * 0.6 + 0.0002 * np.cos(i * 0.3)
                line_points.append(Point(lon, lat, coord_system))

            point_c = Point(174.702, -36.796, coord_system)
            point_d = Point(174.708, -36.792, coord_system)
            tolerance = 200  # 200 meters

        elif choice == "3":
            # Custom data
            coord_system, utm_zone, utm_hemisphere = get_coordinate_system_input()

            filename = input("Enter CSV filename: ").strip()
            line_points = read_geospatial_line_from_file(filename, coord_system, utm_zone, utm_hemisphere)

            print("Enter intermediate points:")
            x = float(input("Point C - X/Longitude: "))
            y = float(input("Point C - Y/Latitude: "))
            point_c = Point(x, y, coord_system, utm_zone, utm_hemisphere)

            x = float(input("Point D - X/Longitude: "))
            y = float(input("Point D - Y/Latitude: "))
            point_d = Point(x, y, coord_system, utm_zone, utm_hemisphere)

            tolerance = float(input("Distance tolerance (meters): "))

        else:
            print("Invalid choice")
            return

        # Create calculator
        calculator = GeospatialRailwayCalculator(line_points)

        # Display line information
        line_info = calculator.get_line_info()
        print(f"\nRailway Line Information:")
        print(f"Coordinate System: {line_info['coordinate_system'].value}")
        print(f"Total Points: {line_info['total_points']}")
        print(f"Total Length: {line_info['total_length']:.2f} {line_info['units']}")
        print(f"Start Point: {line_info['start_point']}")
        print(f"End Point: {line_info['end_point']}")

        # Calculate distance
        closest_c, closest_d, distance_cd, dist_c_along, dist_d_along = calculator.calculate_distance_between_points(
            point_c, point_d, tolerance
        )

        # Display results
        print(f"\nResults:")
        print("=" * 50)
        print(f"Point C: {point_c}")
        print(f"Closest point on line: {closest_c}")
        print(f"Distance along line: {dist_c_along:.2f} {line_info['units']}")
        print(f"")
        print(f"Point D: {point_d}")
        print(f"Closest point on line: {closest_d}")
        print(f"Distance along line: {dist_d_along:.2f} {line_info['units']}")
        print(f"")
        print(f"Distance CD along railway: {distance_cd:.2f} {line_info['units']}")
        print(f"Distance CD as % of total: {(distance_cd/line_info['total_length'])*100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()