# -*- coding: utf-8 -*-
"""
pypsa_nza_data.analysis

Post-processing and analysis tools for NZ electricity market data.

Modules
-------
nza_poc_aggregator
    Aggregates half-hourly POC-level EA data to site level with energy
    conservation checking. Processes a single monthly CSV file.

nza_process_annual
    Orchestrates full-year processing of monthly export, import, or
    generation dispatch files. Produces annual site-aggregated time
    series CSVs.

nza_build_site_registry
    Builds the definitive PyPSA bus registry from the union of all sites
    appearing in the annual export and import datasets. Merges with
    nodes.csv to add coordinates and metadata.

nza_validate_gen_import
    Systematic five-check validation of the discrepancy between annual
    generation dispatch and grid import totals. Documents the
    investigation for reproducibility.

nza_plot_annual_overview
    Plots annual overview of generation dispatch, grid import, and grid
    export time series. Produces three groups of plots: all-three
    dataset overview, import/export model inputs, and validation plots.
"""

from .nza_poc_aggregator import (
    load_poc_file,
    aggregate_by_site_all_voltages,
    energy_check,
    build_poc_inventory,
    sub_threshold_fraction,
    get_site,
    get_voltage,
)

from .nza_process_annual import (
    process_annual,
)

from .nza_build_site_registry import (
    main as build_site_registry,
)

__all__ = [
    "load_poc_file",
    "aggregate_by_site_all_voltages",
    "energy_check",
    "build_poc_inventory",
    "sub_threshold_fraction",
    "get_site",
    "get_voltage",
    "process_annual",
    "build_site_registry",
]