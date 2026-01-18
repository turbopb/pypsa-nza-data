#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nza_cx_modality_standard.py

Standardize energy carrier/modality names in generator data for PyPSA network
modeling and capacity expansion analysis.

DESCRIPTION
-----------
This utility standardizes inconsistent energy carrier/modality names in CSV
files to a consistent set of PyPSA-compatible identifiers. It handles common
variations, abbreviations, and synonyms for energy technologies, ensuring
uniform naming across datasets for network modeling.

The standardization is critical for:
    - PyPSA network model compatibility
    - Consistent carrier definitions across datasets
    - Capacity expansion optimization accuracy
    - Technology-specific constraint application
    - Economic dispatch calculations

PROBLEM ADDRESSED
-----------------
Generator data from various sources (Electricity Authority, MBIE, etc.) uses
inconsistent naming conventions for energy technologies:
    - Different abbreviations: "OCGT", "COG", "gas"
    - Case variations: "Wind", "WIND", "wind"
    - Descriptive variations: "solar", "solar pv", "photovoltaic"
    - Whitespace issues: "pumped storage", "pumped_storage"

This tool maps all variations to a consistent standard set.

MAPPING LOGIC
-------------
The MODALITY_MAP dictionary defines the standardization rules:

Standard Carriers (Target Names):
    - wind: Onshore wind generation
    - offshore_wind: Offshore wind generation (future)
    - solar_pv: Solar photovoltaic generation
    - hydro: Hydroelectric (dams, run-of-river)
    - pumped_storage: Pumped hydro storage
    - geothermal: Geothermal generation
    - gas: Gas turbines (OCGT, CCGT, cogeneration)
    - coal: Coal-fired generation
    - biomass: Biomass/biogas generation
    - diesel: Diesel/fuel oil backup generation
    - battery: Battery energy storage systems (BESS)

Each standard carrier has multiple input variations that map to it.
The mapping is case-insensitive and strips whitespace.

INPUT FORMAT
------------
CSV file with a 'carrier' column containing technology names.
Example input:
    name,site,carrier,power
    Clyde,CYD,HYDRO,432.0
    Te Apiti,NAP,WND,90.6
    Huntly,HTU,OCGT,403.0

OUTPUT FORMAT
-------------
Same CSV structure with standardized 'carrier' column.
Example output:
    name,site,carrier,power
    Clyde,CYD,hydro,432.0
    Te Apiti,NAP,wind,90.6
    Huntly,HTU,gas,403.0

USAGE
-----
As a standalone script:
    python nza_cx_modality_standard.py

As a module:
    from nza_cx_modality_standard import standardize_carrier_column
    standardize_carrier_column("input.csv", "output.csv")

CONFIGURATION
-------------
Reads YAML configuration file for input/output paths:
    - Config: config/nza_cx_config.yaml
    - Default input: data/processed/static/generators.csv
    - Default output: data/processed/static/generators_ref.csv

VALIDATION
----------
The tool performs comprehensive validation:
    - File existence and format checks
    - 'carrier' column presence verification
    - Empty value detection
    - Unmapped value identification (fails with clear error)
    - Pre-flight validation before any modifications

LOGGING
-------
- Creates timestamped log files in logs/ directory
- Logs to both console (simple format) and file (detailed format)
- Log filename: nza_cx_modality_standard_YYYYMMDD_HHMMSS.log

ERROR HANDLING
--------------
Returns clear error messages for:
    - Missing input files
    - Non-CSV file formats
    - Missing 'carrier' column
    - Empty carrier values
    - Unmapped carrier values (lists all unmapped)
    - File I/O errors

DESIGN DECISION: YAML vs. HARDCODED MAPPING
--------------------------------------------
The MODALITY_MAP is intentionally kept in Python code rather than YAML because:
    1. Core business logic, not configuration
    2. Performance - no file I/O overhead per row
    3. Type safety - IDE autocomplete and validation
    4. Self-contained - works without external files
    5. Low change frequency - energy technologies are stable

If you need:
    - Non-programmers to modify mappings
    - Multiple mapping scenarios
    - Dynamic mapping updates
Consider moving to YAML in future versions.

AUTHOR
------
    Philippe Bruneau

CREATED
-------
    2025-11-18
    Created on Thu Dec 18 16:39:20 2025


MODIFIED
--------
    2025-12-17 - Refactored with professional logging and formatting
               - Added comprehensive documentation
               - Implemented cross-platform path handling
               - Enhanced error handling and validation
               - Added detailed summary reporting

VERSION
-------
    2.0.0
"""

from pathlib import Path
from datetime import datetime
import sys
import logging
from typing import Dict, List

import pandas as pd
import yaml

from nza_root import ROOT_DIR


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> Path:
    """
    Configure logging to write to both console and file.
    
    Args:
        log_dir: Directory where log files will be stored
        
    Returns:
        Path to the log file
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'nza_cx_modality_standard_{timestamp}.log'
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    root_logger.handlers.clear()
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (detailed format)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def print_header(title: str, char: str = '=') -> None:
    """Print a formatted header for console output."""
    width = 80
    logger.info(char * width)
    logger.info(f"{title:^{width}}")
    logger.info(char * width)


def print_section(title: str) -> None:
    """Print a section divider."""
    logger.info("")
    logger.info(f"--- {title} ---")
    logger.info("")


# ============================================================================
# MODALITY MAPPING
# ============================================================================

# Standard carrier/modality mapping
# This maps various input names to consistent PyPSA-compatible carriers
MODALITY_MAP: Dict[str, str] = {
    # Wind - Onshore
    "wind": "wind",
    "wnd": "wind",
    "win": "wind",
    "onw": "wind",
    "onshore": "wind",
    "on-shore": "wind",
    "onshore wind": "wind",
    
    # Wind - Offshore (future)
    "offshore": "offshore_wind",
    "off-shore": "offshore_wind",
    "offshore wind": "offshore_wind",
    
    # Solar Photovoltaic
    "solar": "solar_pv",
    "solar pv": "solar_pv",
    "pv": "solar_pv",
    "photovoltaic": "solar_pv",
    "sol": "solar_pv",
    
    # Hydro
    "hydro": "hydro",
    "hyd": "hydro",
    "hydroelectric": "hydro",
    "hydro dam": "hydro",
    "run-of-river": "hydro",
    "ror": "hydro",
    "water": "hydro",
    
    # Pumped Hydro Storage
    "pumped storage": "pumped_storage",
    "pumped hydro": "pumped_storage",
    "pumped_hydro": "pumped_storage",
    "psh": "pumped_storage",
    "pumped": "pumped_storage",
    
    # Geothermal
    "geothermal": "geothermal",
    "geo-thermal": "geothermal",
    "geo": "geothermal",
    "geo thermal": "geothermal",
    
    # Gas (all types)
    "ocgt": "gas",
    "cog": "gas",
    "open cycle gas turbine": "gas",
    "ccgt": "gas",
    "closed cycle gas turbine": "gas",
    "combined cycle gas turbine": "gas",
    "combined cycle": "gas",
    "gas": "gas",
    "natural gas": "gas",
    "gas turbine": "gas",
    
    # Coal
    "coal": "coal",
    "lignite": "coal",
    
    # Biomass
    "biomass": "biomass",
    "bio": "biomass",
    "biogas": "biomass",
    "wood": "biomass",
    "waste": "biomass",
    
    # Diesel / Fuel Oil
    "diesel": "diesel",
    "oil": "diesel",
    "fuel oil": "diesel",
    "fuel-oil": "diesel",
    "fuel_oil": "diesel",
    "dsl": "diesel",
    "distillate": "diesel",
    
    # Battery Storage
    "bat": "battery",
    "battery": "battery",
    "battery storage": "battery",
    "battery-storage": "battery",
    "battery_storage": "battery",
    "bess": "battery",
    "batteries": "battery",
}


def standard_modality(modality: str) -> str:
    """
    Map non-standard energy modality descriptor to a consistent standard.
    
    Performs case-insensitive lookup with whitespace stripping to find
    the appropriate standardized carrier name.
    
    Parameters
    ----------
    modality : str
        Energy modality/carrier descriptor (can be any case, with whitespace)
        
    Returns
    -------
    str
        Standardized energy carrier name (lowercase, underscored)
        
    Raises
    ------
    KeyError
        If the modality is not in the mapping dictionary
        
    Examples
    --------
    >>> standard_modality("OCGT")
    'gas'
    >>> standard_modality("  Solar PV  ")
    'solar_pv'
    >>> standard_modality("run-of-river")
    'hydro'
    """
    # Normalize: lowercase and strip whitespace
    modality_normalized = modality.lower().strip()
    
    # Look up in mapping
    if modality_normalized not in MODALITY_MAP:
        raise KeyError(f"Unknown modality: '{modality}'")
    
    return MODALITY_MAP[modality_normalized]


# ============================================================================
# STANDARDIZATION PROCESSING
# ============================================================================

def standardize_carrier_column(
    input_file: Path,
    output_file: Path = None,
    delimiter: str = ',',
    encoding: str = 'utf-8'
) -> Path:
    """
    Standardize the 'carrier' column in a CSV file using standard_modality.
    
    Reads a CSV file, validates the 'carrier' column, standardizes all values
    using the MODALITY_MAP, and writes the result to an output file.
    
    Performs comprehensive validation before any modifications to prevent
    data corruption.
    
    Parameters
    ----------
    input_file : Path
        Path to the input CSV file
    output_file : Path, optional
        Path to the output CSV file. If None, overwrites the input file.
    delimiter : str, optional
        CSV delimiter (default: ',')
    encoding : str, optional
        File encoding (default: 'utf-8')
        
    Returns
    -------
    Path
        Path to the output file
        
    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist
    ValueError
        If the 'carrier' column doesn't exist, is empty, or contains 
        unmapped values
    IOError
        If there are file read/write errors
    """
    # Convert to Path objects
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file else input_path
    
    print_section("Validating Input")
    
    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Input file:  {input_path}")
    logger.info(f"Output file: {output_path}")
    
    # Check if input file is a CSV
    if input_path.suffix.lower() not in ['.csv', '.txt']:
        raise ValueError(
            f"Input file must be a CSV file, got: {input_path.suffix}"
        )
    
    # Read the CSV file
    print_section("Reading Data")
    
    try:
        # Treat empty strings as actual empty strings, not NaN
        df = pd.read_csv(
            input_path,
            delimiter=delimiter,
            encoding=encoding,
            keep_default_na=False,
            na_values=['']
        )
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {input_path} is empty")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError(f"The file {input_path} contains no data")
    
    # Check if 'carrier' column exists
    if 'carrier' not in df.columns:
        raise ValueError(
            f"Column 'carrier' not found in {input_path.name}. "
            f"Available columns: {', '.join(df.columns)}"
        )
    
    logger.info(f"Columns: {', '.join(df.columns)}")
    
    # Validate carrier column
    print_section("Validating Carrier Column")
    
    # Strip whitespace from carrier column
    df['carrier'] = df['carrier'].astype(str).str.strip()
    
    # Check for empty strings or 'nan' strings after stripping
    empty_mask = (df['carrier'] == '') | (df['carrier'].str.lower() == 'nan')
    empty_count = empty_mask.sum()
    
    if empty_count > 0:
        problem_indices = df[empty_mask].index.tolist()
        raise ValueError(
            f"The 'carrier' column contains {empty_count} empty or 'nan' "
            f"values at row indices: {problem_indices}. "
            f"Please clean the data before standardization."
        )
    
    # Get unique carriers
    unique_carriers = df['carrier'].unique()
    logger.info(f"Found {len(unique_carriers)} unique carrier values:")
    for carrier in sorted(unique_carriers):
        count = (df['carrier'] == carrier).sum()
        logger.info(f"  {carrier}: {count} records")
    
    # Identify unmapped values before attempting transformation
    print_section("Checking Mapping Coverage")
    
    unmapped_values = []
    for value in unique_carriers:
        try:
            standard_modality(value)
        except KeyError:
            unmapped_values.append(value)
    
    if unmapped_values:
        raise ValueError(
            f"The following carrier values are not in the mapping: "
            f"{unmapped_values}. Please update the MODALITY_MAP or "
            f"correct these values in your data."
        )
    
    logger.info("✓ All carrier values have valid mappings")
    
    # Show mapping preview
    logger.info("")
    logger.info("Mapping preview:")
    for carrier in sorted(unique_carriers):
        standardized = standard_modality(carrier)
        logger.info(f"  {carrier} → {standardized}")
    
    # Apply the standardization
    print_section("Standardizing Carrier Values")
    
    try:
        df['carrier'] = df['carrier'].apply(standard_modality)
        logger.info(f"✓ Standardized {len(df)} rows")
    except Exception as e:
        raise ValueError(f"Error during standardization: {str(e)}")
    
    # Summary of standardized values
    standardized_carriers = df['carrier'].unique()
    logger.info(f"Result: {len(standardized_carriers)} unique standard carriers:")
    for carrier in sorted(standardized_carriers):
        count = (df['carrier'] == carrier).sum()
        logger.info(f"  {carrier}: {count} records")
    
    # Write the output file
    print_section("Writing Output")
    
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding=encoding)
        logger.info(f"✓ Saved: {output_path}")
    except Exception as e:
        raise IOError(f"Error writing output file {output_path}: {str(e)}")
    
    return output_path


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config(config_file: Path) -> Dict:
    """
    Read the YAML configuration file.
    
    Parameters
    ----------
    config_file : Path
        Path to the YAML configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
        
    Raises
    ------
    FileNotFoundError
        If configuration file does not exist
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config if config else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    
    # Convert ROOT_DIR to Path and setup paths
    root_path = Path(ROOT_DIR)
    log_dir = root_path / "logs"
    
    # Set up logging
    log_file = setup_logging(log_dir)
    
    # Print startup banner
    print_header("NZA CARRIER MODALITY STANDARDIZATION")
    logger.info("")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    try:
        # Load configuration (optional - defines default paths)
        config_file = root_path / "config" / "nza_cx_config.yaml"
        
        # Define file paths
        # Using hardcoded paths as in original, but with Path objects
        input_file = root_path / "data/processed/static/generators.csv"
        output_file = root_path / "data/processed/static/generators.csv"
        
        logger.info(f"Root directory: {root_path}")
        logger.info("")
        
        # Process the data
        print_header("STANDARDIZING CARRIER COLUMN", '=')
        logger.info("")
        
        result = standardize_carrier_column(input_file, output_file)
        
        # Final message
        print_header("STANDARDIZATION COMPLETE", '=')
        logger.info("")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info("")
        logger.info(f"Output file: {result}")
        logger.info(f"Log saved to: {log_file}")
        logger.info("")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"✗ File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"✗ Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"✗ Application failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code else 0)
    except KeyboardInterrupt:
        logger.error("")
        logger.error("✗ Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        sys.exit(1)