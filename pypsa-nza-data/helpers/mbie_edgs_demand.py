#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
electricity_demand_mbie.py

Analysis and visualization of MBIE electricity demand growth scenarios.
Reads 2023/24 MBIE electricity demand growth scenario data from CSV file,
groups by scenario, and plots growth curves over time. Historical data from
1990 provides context for future predictions starting from reference year 2023.

Author: Phillippe Bruneau
Created: Mon July 27 21:24:38 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import root_scalar
import plot_multi_line as pml


def parametric_inverse_interpolation(t, y, y_target, s_tol=1e-4):
    """
    Find t-values where y equals y_target using parametric interpolation.

    Uses arc-length parameterization with cubic splines to find all points
    where the curve crosses a target y-value.

    Parameters
    ----------
    t : array-like
        Independent variable (e.g., time/years)
    y : array-like
        Dependent variable (e.g., energy demand)
    y_target : float
        Target y-value to find
    s_tol : float, optional
        Tolerance for arc length calculation (default: 1e-4)

    Returns
    -------
    ndarray
        Array of t-values where y equals y_target
    """
    # Compute cumulative arc length
    dt = np.diff(t)
    dy = np.diff(y)
    ds = np.sqrt(dt**2 + dy**2)
    s = np.insert(np.cumsum(ds), 0, 0)

    # Fit splines: t(s) and y(s)
    t_spline = CubicSpline(s, t)
    y_spline = CubicSpline(s, y)

    # Find roots of y(s) - y_target
    roots = []
    for i in range(1, len(s)):
        s0, s1 = s[i-1], s[i]
        y0, y1 = y_spline(s0), y_spline(s1)

        # Check for sign change (potential root)
        if (y0 - y_target) * (y1 - y_target) <= 0:
            try:
                sol = root_scalar(
                    lambda s_val: y_spline(s_val) - y_target,
                    bracket=[s0, s1],
                    method='brentq'
                )
                if sol.converged:
                    roots.append(sol.root)
            except ValueError:
                continue

    # Evaluate t at found arc length values
    t_matches = t_spline(roots)
    return np.array(t_matches)


def interpolate_tabulated_data(t, y, t_interp):
    """
    Interpolate y-values at specified t-values.

    Parameters
    ----------
    t : array-like
        Original independent variable values
    y : array-like
        Original dependent variable values
    t_interp : array-like
        New t-values at which to interpolate

    Returns
    -------
    ndarray
        Interpolated y-values
    """
    f = interp1d(t, y, kind='linear')
    return f(t_interp)


def extract_scenario_data(path, csv_file, value_column='Value'):
    """
    Extract and separate MBIE demand scenarios into individual CSV files.

    Reads the master CSV file, groups by scenario, and writes each scenario
    to a separate file named 'mbie_scenario_{scenario_name}.csv'.

    Parameters
    ----------
    csv_file : str
        Path to the master CSV file
    value_column : str, optional
        Name of the column containing numeric values (default: 'Value')
    """
    df = pd.read_csv(csv_file)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    grouped = df.groupby('Scenario')
    for scenario, group in grouped:
        filename = path + f"mbie_scenario_{scenario}.csv"
        group.to_csv(filename, index=False)
        print(f"Created: {filename}")


def plot_all_scenarios(csv_file, value_column='Value', title="MBIE Scenario Trends"):
    """
    Plot all scenarios from the master CSV file on a single graph.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file
    value_column : str, optional
        Column name for y-axis values (default: 'Value')
    title : str, optional
        Plot title
    """
    df = pd.read_csv(csv_file)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    grouped = df.groupby('Scenario')

    plt.figure(figsize=(14, 10))
    for scenario, group in grouped:
        plt.plot(group['Year'], group[value_column], label=scenario, lw=3)

    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Energy (GWh)", fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(title="Scenario", title_fontsize=16, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


def analyze_scenario(path, scenario_name, year_range=(2024, 2050, 2)):
    """
    Analyze a specific scenario and create detailed visualization.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario (e.g., 'Reference', 'Growth')
    year_range : tuple, optional
        (start, stop, step) for year analysis (default: 2024 to 2050, step 2)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    year_list : ndarray
        Array of analyzed years
    E_year : ndarray
        Energy demand values for analyzed years
    """
    # Load scenario data
    filename = path + f"mbie_scenario_{scenario_name}.csv"
    df = pd.read_csv(filename)

    E = df['Value'].to_numpy()
    t = df['Year'].to_numpy()

    # Generate year list
    start, stop, step = year_range
    year_list = np.arange(start, stop + 1, step)

    # Interpolate energy demand for selected years
    E_year = interpolate_tabulated_data(t, E, year_list)

    # Save interpolated data
    output_filename = f"scenario_{scenario_name}_energy.csv"
    output_df = pd.DataFrame({'year': year_list, 'energy': E_year})
    output_df.to_csv(output_filename, index=False)
    print(f"Saved interpolated data to: {output_filename}")

    # Plot settings
    plot_settings = {
        'title': f"MBIE Electricity Demand Forecast - {scenario_name} Scenario",
        'xlabel': "Year",
        'ylabel': "Energy Demand (TWh)",
        'xlabel_fontsize': 16,
        'ylabel_fontsize': 16,
        'tick_labelsize': 14,
        'title_fontsize': 18,
        'x_min': min(t) - 2,
        'x_max': max(year_list) + 3,
        'y_min': min(E) * 0.9,
        'y_max': max(E_year) * 1.1,
        'grid': True,
        'grid_kwargs': {'which': 'both', 'linestyle': '-', 'linewidth': 0.5, 'color':'black',
                        'alpha': 0.5},
        'legend': True,
        'legend_title': "Scenario",
        'legend_loc': 'upper left',
        'line_styles': {
            scenario_name: {
                'color': 'black',
                'linestyle': '-',
                'linewidth': 3.5,
                'marker': 'o',
                'markersize': 0
            }
        }
    }

    # Create plot
    y_data = {scenario_name: E}
    fig, ax = pml.plot_multiple_lines(t, y_data, plot_settings)

    # Add annotations for key years
    for i, year in enumerate(year_list):
        # Horizontal and vertical reference lines
        ax.axhline(y=E_year[i], color='g', linestyle='--', lw=1.6, alpha=0.4)
        ax.axvline(x=year, color='g', linestyle='--', lw=1.6, alpha=0.4)

        # Marker points
        ax.plot(year, E_year[i], 'ok', markersize=14,
                markerfacecolor='white', markeredgewidth=2.2)
        ax.plot(year, E_year[i], '+k', markersize=20,
                markerfacecolor='k', markeredgewidth=2.5)

        # Text annotations
        ax.annotate(f"    {year}", xy=(year, E_year[i] + 0.15), fontsize=15)
        ax.annotate(f"{E_year[i]:.1f} TWh", xy=(year - 6.6, E_year[i] + 0.15),
                   fontsize=15)

    plt.tight_layout()

    return fig, ax, year_list, E_year


def main():
    """Main execution function."""
    # Configuration
    path = "/home/pbrun/pypsa-nza/data/external/mbie_edgs/"
    master_csv = path+"electricity-demand-scenarios-total-2024-results.csv"
    scenario_to_analyze = 'Reference'
    year_range = (2024, 2050, 1)

    # Step 1: Extract individual scenario files
    print("Extracting scenario data...")
    extract_scenario_data(path, master_csv)

    # Step 2: Plot all scenarios for comparison
    print("\nPlotting all scenarios...")
    plot_all_scenarios(master_csv)

    # Step 3: Detailed analysis of selected scenario
    print(f"Analyzing {scenario_to_analyze} scenario...")
    fig, ax, years, energy = analyze_scenario(path, scenario_to_analyze, year_range)

    # Step 4: Compute energy factors for selected scenario
    print(f"Compute energy factors {scenario_to_analyze} scenario...")
    energy_factor = [f/energy[0] for f in energy]


    # Display summary
    print(f"\n{scenario_to_analyze} Scenario Summary:")
    print("-" * 50)
    print("Year \tDemand (TWh) \tEnergy factor")
    print("-" * 50)
    for year, E, energy_factor in zip(years, energy, energy_factor):
        print(f"{year} \t{E:.2f} \t\t\t{energy_factor:.3f}")

    plt.show()

    return energy


if __name__ == "__main__":
    energy = main()