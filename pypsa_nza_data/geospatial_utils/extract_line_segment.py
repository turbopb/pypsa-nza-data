# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:32:24 2025

@author: OEM
"""

#!/usr/bin/env python3
"""
Railway Line Distance Calculator

This program calculates the traveled distance along a railway line between two points.
The railway line is defined by a series of x-y coordinates, and the program finds
the closest points on the line to the given intermediate points C and D, then
calculates the distance along the line between these points.

Author: Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
import sys
import csv
from pathlib import Path


@dataclass
class Point:
    """Represents a 2D point with x and y coordinates."""
    x: float
    y: float

    def __post_init__(self):
        """Convert coordinates to float for consistency."""
        self.x = float(self.x)
        self.y = float(self.y)

    def to_array(self) -> np.ndarray:
        """Convert point to numpy array."""
        return np.array([self.x, self.y])

    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __str__(self) -> str:
        return f"Point({self.x:.6f}, {self.y:.6f})"


@dataclass
class LineSegmentInfo:
    """Information about a line segment between two consecutive points."""
    start_point: Point
    end_point: Point
    length: float
    cumulative_distance: float  # Distance from start of line to end of this segment


class RailwayLineCalculator:
    """
    Calculates distances along a railway line defined by a series of points.
    """

    def __init__(self, line_points: List[Point]):
        """
        Initialize with the railway line points.

        Args:
            line_points: List of points defining the railway line from A to B
        """
        if len(line_points) < 2:
            raise ValueError("Railway line must have at least 2 points")

        self.line_points = line_points
        self.segments = self._calculate_segments()
        self.total_length = self.segments[-1].cumulative_distance if self.segments else 0

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
        """
        Find the closest point on a line segment to a given point.

        Args:
            point: Point to find closest point to
            segment: Line segment to project onto

        Returns:
            Tuple of (closest_point, parameter_t, distance_to_segment)
            where parameter_t is between 0 and 1 (0=start, 1=end of segment)
        """
        # Vector from segment start to end
        segment_vector = segment.end_point.to_array() - segment.start_point.to_array()
        segment_length_sq = np.dot(segment_vector, segment_vector)

        if segment_length_sq == 0:
            # Degenerate case: segment has zero length
            closest_point = segment.start_point
            distance = point.distance_to(closest_point)
            return closest_point, 0.0, distance

        # Vector from segment start to point
        point_vector = point.to_array() - segment.start_point.to_array()

        # Project point onto segment
        t = np.dot(point_vector, segment_vector) / segment_length_sq

        # Clamp t to [0, 1] to stay within segment
        t = max(0.0, min(1.0, t))

        # Calculate closest point
        closest_array = segment.start_point.to_array() + t * segment_vector
        closest_point = Point(closest_array[0], closest_array[1])

        # Calculate distance
        distance = point.distance_to(closest_point)

        return closest_point, t, distance

    def find_closest_point_on_line(self, point: Point) -> Tuple[Point, float, int, float]:
        """
        Find the closest point on the entire railway line to a given point.

        Args:
            point: Point to find closest point to

        Returns:
            Tuple of (closest_point, distance_along_line, segment_index, distance_to_line)
        """
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

        # Add distance within the current segment
        distance_along_line += best_t * self.segments[best_segment_index].length

        return best_point, distance_along_line, best_segment_index, min_distance

    def calculate_distance_between_points(self, point_c: Point, point_d: Point,
                                        tolerance: float) -> Tuple[Point, Point, float, float, float]:
        """
        Calculate the distance along the railway line between points C and D.

        Args:
            point_c: First intermediate point
            point_d: Second intermediate point
            tolerance: Maximum allowed distance from points to railway line

        Returns:
            Tuple of (closest_c, closest_d, distance_cd, distance_c_along_line, distance_d_along_line)

        Raises:
            ValueError: If points are outside tolerance
        """
        # Find closest points on the railway line
        closest_c, dist_c_along, seg_c, dist_c_to_line = self.find_closest_point_on_line(point_c)
        closest_d, dist_d_along, seg_d, dist_d_to_line = self.find_closest_point_on_line(point_d)

        # Check distance tolerance
        if dist_c_to_line > tolerance:
            raise ValueError(
                f"Point C is {dist_c_to_line:.6f} units from railway line, "
                f"exceeding tolerance of {tolerance:.6f}"
            )

        if dist_d_to_line > tolerance:
            raise ValueError(
                f"Point D is {dist_d_to_line:.6f} units from railway line, "
                f"exceeding tolerance of {tolerance:.6f}"
            )

        # Calculate distance along the line between the two points
        distance_cd = abs(dist_d_along - dist_c_along)

        return closest_c, closest_d, distance_cd, dist_c_along, dist_d_along

    def get_line_info(self) -> dict:
        """Get information about the railway line."""
        return {
            'total_points': len(self.line_points),
            'total_segments': len(self.segments),
            'total_length': self.total_length,
            'start_point': self.line_points[0],
            'end_point': self.line_points[-1]
        }


def read_line_from_file(filename: str) -> List[Point]:
    """
    Read railway line points from a CSV file.

    Args:
        filename: Path to CSV file with columns 'x' and 'y'

    Returns:
        List of Point objects
    """
    points = []

    try:
        with open(filename, 'r', newline='') as csvfile:
            # Try to detect if file has headers
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
                    print(f"Warning: Row {row_num} has fewer than 2 columns, skipping")
                    continue

                try:
                    x = float(row[0])
                    y = float(row[1])
                    points.append(Point(x, y))
                except ValueError:
                    print(f"Warning: Row {row_num} contains non-numeric data, skipping")
                    continue

        if not points:
            raise ValueError("No valid points found in file")

        print(f"Successfully loaded {len(points)} points from {filename}")
        return points

    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found")
    except Exception as e:
        raise Exception(f"Error reading file {filename}: {e}")


def create_sample_line_file(filename: str):
    """Create a sample railway line file for demonstration."""
    # Create a curved railway line (like a gentle S-curve)
    t = np.linspace(0, 4*np.pi, 50)
    x = t
    y = 2 * np.sin(t/2) + 0.5 * np.sin(t)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])  # Header
        for xi, yi in zip(x, y):
            writer.writerow([f"{xi:.6f}", f"{yi:.6f}"])

    print(f"Created sample railway line file: {filename}")


def create_visualization(calculator: RailwayLineCalculator, point_c: Point, point_d: Point,
                        closest_c: Point, closest_d: Point, tolerance: float) -> None:
    """
    Create a visualization of the railway line and distance calculation.
    """
    plt.figure(figsize=(18, 16))

    # Plot railway line
    x_coords = [p.x for p in calculator.line_points]
    y_coords = [p.y for p in calculator.line_points]
    plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Railway Line', alpha=0.8)

    # Plot railway line points
    plt.plot(x_coords, y_coords, 'b.', markersize=4, alpha=0.6)

    # Mark start and end points
    plt.plot(calculator.line_points[0].x, calculator.line_points[0].y,
             'go', markersize=10, label='Start (A)')
    plt.plot(calculator.line_points[-1].x, calculator.line_points[-1].y,
             'rs', markersize=10, label='End (B)')

    # Plot original points C and D
    plt.plot(point_c.x, point_c.y, 'ro', markersize=8, label='Point C')
    plt.plot(point_d.x, point_d.y, 'mo', markersize=8, label='Point D')

    # Plot closest points on railway line
    plt.plot(closest_c.x, closest_c.y, 'r^', markersize=8, label='Closest to C')
    plt.plot(closest_d.x, closest_d.y, 'm^', markersize=8, label='Closest to D')

    # Draw lines from original points to closest points
    plt.plot([point_c.x, closest_c.x], [point_c.y, closest_c.y],
             'r--', alpha=0.7, linewidth=1)
    plt.plot([point_d.x, closest_d.x], [point_d.y, closest_d.y],
             'm--', alpha=0.7, linewidth=1)

    # Highlight the segment CD on the railway line
    # Find the portion of the line between closest_c and closest_d
    _, dist_c_along, _, _ = calculator.find_closest_point_on_line(point_c)
    _, dist_d_along, _, _ = calculator.find_closest_point_on_line(point_d)

    # Ensure C comes before D along the line
    if dist_c_along > dist_d_along:
        dist_c_along, dist_d_along = dist_d_along, dist_c_along
        closest_c, closest_d = closest_d, closest_c

    # Extract the portion of the line between C and D
    segment_x = [closest_c.x]
    segment_y = [closest_c.y]

    current_dist = 0
    for segment in calculator.segments:
        seg_start_dist = current_dist
        seg_end_dist = current_dist + segment.length

        if seg_end_dist >= dist_c_along and seg_start_dist <= dist_d_along:
            # This segment overlaps with our region of interest
            if seg_start_dist >= dist_c_along:
                segment_x.append(segment.start_point.x)
                segment_y.append(segment.start_point.y)

            if seg_end_dist <= dist_d_along:
                segment_x.append(segment.end_point.x)
                segment_y.append(segment.end_point.y)

        current_dist = seg_end_dist

    segment_x.append(closest_d.x)
    segment_y.append(closest_d.y)

    plt.plot(segment_x, segment_y, 'g-', linewidth=4, alpha=0.7, label='Distance CD')

    # Add tolerance circles
    circle_c = plt.Circle((point_c.x, point_c.y), tolerance,
                         fill=False, color='red', alpha=0.3, linestyle='--')
    circle_d = plt.Circle((point_d.x, point_d.y), tolerance,
                         fill=False, color='magenta', alpha=0.3, linestyle='--')
    plt.gca().add_patch(circle_c)
    plt.gca().add_patch(circle_d)

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title('Railway Line Distance Calculation', fontsize=22)
    plt.xlabel('X Coordinate', fontsize=22)
    plt.ylabel('Y Coordinate', fontsize=22)
    plt.tight_layout()
    plt.show()


def get_user_input() -> Tuple[List[Point], Point, Point, float]:
    """
    Get user input for railway line data, points, and tolerance.
    """
    print("Railway Line Distance Calculator")
    print("=" * 50)

    # Get railway line data
    print("Railway Line Data:")
    print("1. Load from file")
    print("2. Enter points manually")
    print("3. Use sample data")

    choice = input("Choose option (1-3): ").strip()

    line_points = []

    if choice == "1":
        filename = input("Enter CSV filename: ").strip()
        line_points = read_line_from_file(filename)

    elif choice == "2":
        print("Enter railway line points (empty line to finish):")
        point_num = 1
        while True:
            try:
                x_input = input(f"Point {point_num} - X coordinate (or press Enter to finish): ").strip()
                if not x_input:
                    break
                x = float(x_input)
                y = float(input(f"Point {point_num} - Y coordinate: "))
                line_points.append(Point(x, y))
                point_num += 1
            except ValueError:
                print("Invalid input, please enter numeric values")

        if len(line_points) < 2:
            raise ValueError("Need at least 2 points for railway line")

    elif choice == "3":
        # Create sample curved line
        t = np.linspace(0, 10, 20)
        x = t
        y = 2 * np.sin(t/2)
        line_points = [Point(xi, yi) for xi, yi in zip(x, y)]
        print(f"Using sample railway line with {len(line_points)} points")

    else:
        raise ValueError("Invalid choice")

    # Get intermediate points
    print(f"\nRailway line loaded with {len(line_points)} points")
    print("Total line length will be calculated...")

    try:
        print("\nIntermediate Points:")
        cx = float(input("Point C - X coordinate: "))
        cy = float(input("Point C - Y coordinate: "))
        dx = float(input("Point D - X coordinate: "))
        dy = float(input("Point D - Y coordinate: "))

        print("\nTolerance:")
        tolerance = float(input("Distance tolerance: "))

        if tolerance <= 0:
            raise ValueError("Tolerance must be positive")

        return line_points, Point(cx, cy), Point(dx, dy), tolerance

    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)


def run_example():
    """Run a predefined example."""
    print("Running Example with Sample Railway Line")
    print("=" * 50)

    # Create a sample curved railway line
    t = np.linspace(0, 10, 25)
    x = t
    y = 2 * np.sin(t/2) + 0.3 * np.cos(t)
    line_points = [Point(xi, yi) for xi, yi in zip(x, y)]

    # Intermediate points
    point_c = Point(3.0, 1.5)
    point_d = Point(7.0, -1.2)
    tolerance = 1.0

    print(f"Railway line: {len(line_points)} points")
    print(f"Point C: {point_c}")
    print(f"Point D: {point_d}")
    print(f"Tolerance: {tolerance}")

    return line_points, point_c, point_d, tolerance


def main():
    """Main function."""
    print("Railway Line Distance Calculator")
    print("Choose mode:")
    print("1. Run example")
    print("2. Enter custom data")
    print("3. Create sample file and exit")

    try:
        choice = input("Enter choice (1-3): ").strip()

        if choice == "3":
            filename = input("Enter filename for sample data (default: sample_railway.csv): ").strip()
            if not filename:
                filename = "sample_railway.csv"
            create_sample_line_file(filename)
            return

        elif choice == "1":
            line_points, point_c, point_d, tolerance = run_example()
        elif choice == "2":
            line_points, point_c, point_d, tolerance = get_user_input()
        else:
            print("Invalid choice. Using example data.")
            line_points, point_c, point_d, tolerance = run_example()

        # Create calculator
        calculator = RailwayLineCalculator(line_points)

        # Display line information
        line_info = calculator.get_line_info()
        print(f"\nRailway Line Information:")
        print(f"Total Points: {line_info['total_points']}")
        print(f"Total Segments: {line_info['total_segments']}")
        print(f"Total Length: {line_info['total_length']:.6f}")
        print(f"Start Point: {line_info['start_point']}")
        print(f"End Point: {line_info['end_point']}")

        # Calculate distance between C and D
        closest_c, closest_d, distance_cd, dist_c_along, dist_d_along = calculator.calculate_distance_between_points(
            point_c, point_d, tolerance
        )

        # Display results
        print(f"\nResults:")
        print("=" * 50)
        print(f"Point C: {point_c}")
        print(f"Closest point on line to C: {closest_c}")
        print(f"Distance along line to C: {dist_c_along:.6f}")
        print(f"")
        print(f"Point D: {point_d}")
        print(f"Closest point on line to D: {closest_d}")
        print(f"Distance along line to D: {dist_d_along:.6f}")
        print(f"")
        print(f"Distance CD along railway line: {distance_cd:.6f}")
        print(f"Distance CD as % of total line: {(distance_cd/line_info['total_length'])*100:.2f}%")

        # Create visualization
        print("\nGenerating visualization...")
        create_visualization(calculator, point_c, point_d, closest_c, closest_d, tolerance)

    except ValueError as e:
        print(f"\nError: {e}")
        print("Program terminated.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()