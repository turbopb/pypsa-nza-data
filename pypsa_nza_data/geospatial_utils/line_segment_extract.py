# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:08:58 2025

@author: OEM
"""

#!/usr/bin/env python3
"""
Line Segment Extractor

This program extracts a line segment from line AB that corresponds to the projection
of points C and D onto the line, subject to distance tolerance constraints.

Author: Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from dataclasses import dataclass
import sys


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


class LineSegmentExtractor:
    """
    Extracts line segments from a reference line based on projected points
    with distance tolerance constraints.
    """

    def __init__(self, point_a: Point, point_b: Point):
        """
        Initialize with the reference line segment AB.

        Args:
            point_a: Starting point of reference line
            point_b: Ending point of reference line
        """
        self.point_a = point_a
        self.point_b = point_b
        self.line_vector = self._calculate_line_vector()
        self.line_length = point_a.distance_to(point_b)

        if self.line_length == 0:
            raise ValueError("Points A and B cannot be identical (zero-length line)")

    def _calculate_line_vector(self) -> np.ndarray:
        """Calculate the direction vector of line AB."""
        return self.point_b.to_array() - self.point_a.to_array()

    def _project_point_onto_line(self, point: Point) -> Tuple[Point, float, float]:
        """
        Project a point onto the infinite line defined by AB.

        Args:
            point: Point to project onto the line

        Returns:
            Tuple of (projected_point, parameter_t, distance_to_line)
            where parameter_t indicates position along AB (0=A, 1=B)
        """
        # Vector from A to the point
        ap_vector = point.to_array() - self.point_a.to_array()

        # Project AP onto AB using dot product
        # t = (AP · AB) / |AB|²
        t = np.dot(ap_vector, self.line_vector) / np.dot(self.line_vector, self.line_vector)

        # Calculate the projected point: A + t * AB
        projected_array = self.point_a.to_array() + t * self.line_vector
        projected_point = Point(projected_array[0], projected_array[1])

        # Calculate perpendicular distance from original point to line
        distance_to_line = point.distance_to(projected_point)

        return projected_point, t, distance_to_line

    def extract_segment(self, point_c: Point, point_d: Point,
                       tolerance: float) -> Tuple[Point, Point, float]:
        """
        Extract line segment from AB based on projections of C and D.

        Args:
            point_c: First reference point
            point_d: Second reference point
            tolerance: Maximum allowed distance from C/D to line AB

        Returns:
            Tuple of (start_point, end_point, segment_length)

        Raises:
            ValueError: If points are outside tolerance or other constraints violated
        """
        # Project both points onto the line
        proj_c, t_c, dist_c = self._project_point_onto_line(point_c)
        proj_d, t_d, dist_d = self._project_point_onto_line(point_d)

        # Check distance tolerance
        if dist_c > tolerance:
            raise ValueError(
                f"Point C is {dist_c:.6f} units from line AB, "
                f"exceeding tolerance of {tolerance:.6f}"
            )

        if dist_d > tolerance:
            raise ValueError(
                f"Point D is {dist_d:.6f} units from line AB, "
                f"exceeding tolerance of {tolerance:.6f}"
            )

        # Check if projected points are within the line segment AB
        if not (0 <= t_c <= 1):
            raise ValueError(
                f"Projection of point C (t={t_c:.6f}) falls outside line segment AB "
                f"(valid range: 0 ≤ t ≤ 1)"
            )

        if not (0 <= t_d <= 1):
            raise ValueError(
                f"Projection of point D (t={t_d:.6f}) falls outside line segment AB "
                f"(valid range: 0 ≤ t ≤ 1)"
            )

        # Ensure consistent ordering (smaller t parameter first)
        if t_c > t_d:
            proj_c, proj_d = proj_d, proj_c
            t_c, t_d = t_d, t_c

        # Calculate segment length
        segment_length = proj_c.distance_to(proj_d)

        return proj_c, proj_d, segment_length

    def get_line_info(self) -> dict:
        """Get information about the reference line AB."""
        return {
            'point_a': self.point_a,
            'point_b': self.point_b,
            'length': self.line_length,
            'direction_vector': self.line_vector
        }


def create_visualization(extractor: LineSegmentExtractor, point_c: Point, point_d: Point,
                        start_point: Point, end_point: Point, tolerance: float) -> None:
    """
    Create a visualization of the line extraction process.

    Args:
        extractor: LineSegmentExtractor instance
        point_c: Original point C
        point_d: Original point D
        start_point: Extracted segment start point
        end_point: Extracted segment end point
        tolerance: Distance tolerance used
    """
    plt.figure(figsize=(12, 8))

    # Plot reference line AB
    plt.plot([extractor.point_a.x, extractor.point_b.x],
             [extractor.point_a.y, extractor.point_b.y],
             'b-', linewidth=2, label='Reference Line AB')

    # Plot reference points A and B
    plt.plot(extractor.point_a.x, extractor.point_a.y, 'bo', markersize=8, label='Point A')
    plt.plot(extractor.point_b.x, extractor.point_b.y, 'bs', markersize=8, label='Point B')

    # Plot original points C and D
    plt.plot(point_c.x, point_c.y, 'ro', markersize=8, label='Point C')
    plt.plot(point_d.x, point_d.y, 'rs', markersize=8, label='Point D')

    # Plot extracted segment
    plt.plot([start_point.x, end_point.x], [start_point.y, end_point.y],
             'g-', linewidth=3, label='Extracted Segment')

    # Plot projected points
    plt.plot(start_point.x, start_point.y, 'go', markersize=8, label='Projected Points')
    plt.plot(end_point.x, end_point.y, 'go', markersize=8)

    # Draw projection lines (from original to projected points)
    plt.plot([point_c.x, start_point.x], [point_c.y, start_point.y],
             'r--', alpha=0.7, label='Projection Lines')
    plt.plot([point_d.x, end_point.x], [point_d.y, end_point.y], 'r--', alpha=0.7)

    # Add tolerance circles around C and D
    circle_c = plt.Circle((point_c.x, point_c.y), tolerance,
                         fill=False, color='red', alpha=0.3, linestyle='--')
    circle_d = plt.Circle((point_d.x, point_d.y), tolerance,
                         fill=False, color='red', alpha=0.3, linestyle='--')
    plt.gca().add_patch(circle_c)
    plt.gca().add_patch(circle_d)

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title('Line Segment Extraction from Reference Line')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    plt.show()


def get_user_input() -> Tuple[Point, Point, Point, Point, float]:
    """
    Get user input for points and tolerance.

    Returns:
        Tuple of (point_a, point_b, point_c, point_d, tolerance)
    """
    print("Line Segment Extractor")
    print("=" * 50)
    print("Enter coordinates for the reference line and target points.")
    print()

    try:
        # Get reference line points
        print("Reference Line AB:")
        ax = float(input("Point A - X coordinate: "))
        ay = float(input("Point A - Y coordinate: "))
        bx = float(input("Point B - X coordinate: "))
        by = float(input("Point B - Y coordinate: "))

        print("\nTarget Points CD:")
        cx = float(input("Point C - X coordinate: "))
        cy = float(input("Point C - Y coordinate: "))
        dx = float(input("Point D - X coordinate: "))
        dy = float(input("Point D - Y coordinate: "))

        print("\nTolerance:")
        tolerance = float(input("Distance tolerance: "))

        if tolerance <= 0:
            raise ValueError("Tolerance must be positive")

        return (Point(ax, ay), Point(bx, by), Point(cx, cy), Point(dx, dy), tolerance)

    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)


def run_example():
    """Run a predefined example to demonstrate the functionality."""
    print("Running Example:")
    print("Reference Line: A(0, 0) to B(10, 0)")
    print("Target Points: C(2, 1) and D(8, -0.5)")
    print("Tolerance: 2.0")
    print()

    # Example data
    point_a = Point(0, 0)
    point_b = Point(10, 0)
    point_c = Point(2, 1)
    point_d = Point(8, -0.5)
    tolerance = 2.0

    return point_a, point_b, point_c, point_d, tolerance


def main():
    """Main function to run the line segment extraction."""

    # Choose between example and user input
    print("Choose mode:")
    print("1. Run example")
    print("2. Enter custom data")

    try:
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            point_a, point_b, point_c, point_d, tolerance = run_example()
        elif choice == "2":
            point_a, point_b, point_c, point_d, tolerance = get_user_input()
        else:
            print("Invalid choice. Using example data.")
            point_a, point_b, point_c, point_d, tolerance = run_example()

        # Create extractor
        extractor = LineSegmentExtractor(point_a, point_b)

        # Display input information
        print(f"\nInput Data:")
        print(f"Reference Line: {point_a} to {point_b}")
        print(f"Line Length: {extractor.line_length:.6f}")
        print(f"Target Points: {point_c}, {point_d}")
        print(f"Distance Tolerance: {tolerance:.6f}")
        print()

        # Extract the segment
        start_point, end_point, segment_length = extractor.extract_segment(
            point_c, point_d, tolerance
        )

        # Display results
        print("Results:")
        print("=" * 50)
        print(f"Extracted Segment Start: {start_point}")
        print(f"Extracted Segment End:   {end_point}")
        print(f"Segment Length: {segment_length:.6f}")
        print(f"Segment as % of AB: {(segment_length/extractor.line_length)*100:.2f}%")

        # Create visualization
        print("\nGenerating visualization...")
        create_visualization(extractor, point_c, point_d, start_point, end_point, tolerance)

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