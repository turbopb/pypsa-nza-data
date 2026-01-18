#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *****************************************************************************
#                                                                             *
#   electricity_demand_mbie.py                                                *
#                                                                             *
#   DESCRIPTION                                                               *
#   Reads the 2023/24 MBIE electricity demand growth sccenario data from CSV  *
#   file.  The data is grouped by 'Scenario' and plotted as a series of       *
#   growth curves as a function of time (years). Historical data from 1990 is *
#   included to provide context for the growth curves. The MBIE predictions   *
#   for several future growth scenarios are plotted from the reference year,  *
#   2023.                                                                     *
#                                                                             *
#                                                                             *
#   For smoother functions:
#   If your function is smooth and you want a spline rather than piecewise    *
#   linear interpolation: scipy.interpolate.CubicSpline (more advanced and    *
#   smoother):                                                                *
#   from scipy.interpolate import CubicSpline                                 *
#                                                                             *
#                                                                             *
#   WORKFLOW                                                                  *
#                                                                             *
#   OVERVIEW                                                                  *
#                                                                             *
#   KEY FEATURES                                                              *
#                                                                             *
#                                                                             *
#   DATA SOURCES                                                              *
#                                                                             *
#   UPDATE HISTORY                                                            *
#   Created on Mon July 27 21:24:38 2025                                      *
#   Author : Phillippe Bruneau                                                *
#                                                                             *
# *****************************************************************************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

from plot_graph_line_2d import plot_multiple_lines
#from multi_plot import plot_curves



def parametric_inverse_interpolation(t, y, y_target, s_tol=1e-4):
    """
    parametric_inverse_interpolation.

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    y_target : TYPE
        DESCRIPTION.
    s_tol : TYPE, optional
        DESCRIPTION. The default is 1e-4.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Step 1: Compute cumulative arc length
    dt = np.diff(t)
    dy = np.diff(y)
    ds = np.sqrt(dt**2 + dy**2)
    s = np.insert(np.cumsum(ds), 0, 0)  # arc length parameter

    # Step 2: Fit splines t(s), y(s)
    t_spline = CubicSpline(s, t)
    y_spline = CubicSpline(s, y)

    # Step 3: Find roots of y(s) - y_target
    roots = []
    for i in range(1, len(s)):
        s0, s1 = s[i-1], s[i]
        y0, y1 = y_spline(s0), y_spline(s1)

        if (y0 - y_target) * (y1 - y_target) <= 0:
            try:
                sol = root_scalar(lambda s_: y_spline(s_) - y_target,
                                  bracket=[s0, s1], method='brentq')
                if sol.converged:
                    roots.append(sol.root)
            except ValueError:
                continue  # no root in interval

    # Step 4: Evaluate t(s) at found s values
    t_matches = t_spline(roots)
    return np.array(t_matches)



def interpolate_tabulated_data(t, y, t_interp):
    """
    interpolate_tabulated_data.

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    y_interp : TYPE
        DESCRIPTION.

    """
    # Creates a callable function from tabulated data, providing a means to
    # to interpolate yy at any desired value of tt, including non-tabulated
    # points.

    # Example tabulated data
    # t = np.array([0, 1, 2, 3, 4])
    # y = np.array([0, 1, 4, 9, 16])

    # Create an interpolation function
    f = interp1d(t, y, kind='linear')  # Options: 'linear', 'quadratic', 'cubic', etc.

    # Evaluate at a new point
    y_interp = f(t_interp)  # Interpolates between y[2] and y[3]

    return y_interp


def extract_scenario_data(csv_file, value_column):
    """
    Extract MBIE demand scenario data.

    Reads CSV file, groups data by 'Scenario'. Plots the numeric value against
    'Year'.

    Parameters
    ----------
        - csv_file (str): Path to the CSV file.
        - value_column (str): Name of the column containing numeric values to
            plot.
        - title (str, optional): Title of the plot. Default is 'Scenario Trends
            Over Time'.
        - **kwargs: Additional keyword arguments for customization
            (e.g., line styles, colors, markers).
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the dtring 'Year' is treated as numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Group by 'Scenario'
    grouped = df.groupby('Scenario')

    for scenario, group in grouped:
        filename = f"mbie_scenario_{scenario}.csv"
        group.to_csv(filename, index=False)



def plot_scenario_data(csv_file, value_column, title="MBIE Scenario Trends", **kwargs):
    """
    Read a CSV file, groups data by 'Scenario', and plots the numeric value against 'Year'.

    Parameters
    ----------
    - csv_file (str): Path to the CSV file.
    - value_column (str): Name of the column containing numeric values to plot.
    - title (str, optional): Title of the plot. Default is 'Scenario Trends Over Time'.
    - **kwargs: Additional keyword arguments for customization (e.g., line styles, colors, markers).
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Ensure string 'Year' is treated as numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Group by 'Scenario'
    grouped = df.groupby('Scenario')

    # Plot each scenario
    plt.figure(figsize=(26, 18))

    for scenario, group in grouped:
        plt.plot(group['Year'], group[value_column], label=scenario, lw = 5, **kwargs)

    # Customization
    plt.xlabel("Year", fontsize=26)
    plt.ylabel("Energy (GWh)", fontsize=26)
    plt.title(title, fontsize=32)
    plt.legend(fontsize=18)
    plt.legend(title="Scenario", title_fontsize=21, fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)

    # Show plot
    plt.show()



# Example usage
if __name__=="__main__":

    # Extract all scenario data from the MBIE file and reformat. Files for each
    # scenario are written to a separate csv file, eg. "mbie_scenario_Growth.csv",
    # "mbie_scenario_Reference.csv"  etc.
    path_data="/home/pbrun/pypsa-nza/data/external/mbie_edgs/"
    extract_scenario_data(path_data+"electricity-demand-scenarios-total-2024-results.csv", "Value")

    # Plot all the scenario demand curves to show variation
    plot_scenario_data("electricity-demand-scenarios-total-2024-results.csv", "Value", marker=None, linestyle='-')

    # Load the scenario CSV data file
    #scenario_name_list = ['Constraint', 'Reference', 'Growth', 'Environment', 'Innovation']
    scenario_name = 'Reference'
    filename = f"mbie_scenario_{scenario_name}.csv"

    # Read and extract the data from file
    df = pd.read_csv(filename)
    E = df['Value'].tolist()
    t = df['Year'].tolist()

    # Define years of interest
    # year_list = [2024, 2025, 2030, 2035, 2040, 2045, 2050] # 5 year steps
    start = 2024; stop = 2050; step = 2
    year_list = np.arange(start, stop + 1, step)

    column_index = df.columns.get_loc("Scenario")
    scenario_label = df.iat[0, column_index]

    # Energy demand for selected years
    E_year = interpolate_tabulated_data(t, E, year_list)

    # Write E data to file
    filename = f"scenario_{scenario_name}_energy.csv"
    data = list(zip(year_list, E_year))     # Combine the lists into a list of tuples
    df = pd.DataFrame(data, columns=['year', 'energy']) # Create the DataFrame, specifying column names
    df.to_csv(path_data+filename, index=False)


    e_data = {
        f"{scenario_name}": E,
        #"Scenario B": [1, 4, 6, 8, 10]
    }

    # E_target = 42
    # E_cut = parametric_inverse_interpolation(t, E, E_target)

    plot_settings = {
        'title': "MBIE Electricity Demand Growth Forecast",
        'xlabel': "Year",
        'ylabel': "Energy Demand (TWh)",
        'line_width': 4,
        'legend': True,
        'legend_title': "Scenario",
        'legend_title_fontsize': 11,
        'tick_labelsize': 10,
        'x_min': 1988,
        'x_max': 2053,
        'y_min': 20,
        'y_max': 80,
        'xlabel_size': 12,
        'ylabel_size': 12,
        'grid': True,
        'grid_kwargs': {'which': 'both', 'linestyle': '-', 'linewidth': 2, 'alpha': 1},

        'line_styles': {
            f"{scenario_name}": {'color': 'black', 'linestyle': '-', 'linewidth': 3.5, 'marker': 'o', 'markersize': 0},
            #"Scenario B": {'color': 'g', 'linestyle': '--', 'linewidth': 2, 'marker': 's'}
                }
    }

    #y_data = {scenario_label: E}
    fig, ax = plot_multiple_lines(t, e_data, plot_settings)
    ax.legend(loc='upper left')
    plt.show()

    for i, y in enumerate(year_list):
        ax.annotate(f"    {y}", xy=(y, E_year[i]+0.15), fontsize=15)
        ax.annotate(f"{E_year[i]} TWh", xy=(y-6.6, E_year[i]+0.15), fontsize=15)
        ax.plot(y, E_year[i], 'ok', markersize=14, markerfacecolor='white', markeredgewidth=2.2)    # , label='Limit year'
        ax.plot(y, E_year[i], '+k', markersize=20, markerfacecolor='k', markeredgewidth=2.5)
        ax.axhline(y=E_year[i], color='k', linestyle='--', lw=0.6, alpha=0.6)
        ax.axvline(x=y, color='k', linestyle='-', lw=0.6)

