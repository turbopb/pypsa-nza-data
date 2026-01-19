# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:07:56 2025

@author: OEM
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nza_process_generation_data.py

Process and consolidate New Zealand power grid generator data for PyPSA network
modeling and capacity expansion analysis.

DESCRIPTION
-----------
This module transforms raw generator registry data from the Electricity Authority
into a consolidated, site-level dataset suitable for PyPSA (Python for Power 
System Analysis). The processing pipeline handles duplicate consolidation, 
decommissioned unit removal, site aggregation, data standardization, and 
enrichment with geographic coordinates and site metadata.

INPUT DATA SOURCE
-----------------
Electricity Authority - Dispatched Generation Plant Registry
URL: https://www.emi.ea.govt.nz/Wholesale/Datasets/Generation
File: DispatchedGenerationPlant.csv (updated quarterly)

The registry contains all grid-connected generation units in New Zealand with:
    - Plant identification (name, POC code, unit code)
    - Technical specifications (capacity, technology, fuel type)
    - Connection details (voltage level, commissioned date)
    - Operational status (commissioned, decommissioned dates)

PROCESSING WORKFLOW
-------------------
1. **Load Raw Data**
   - Read CSV with all registered generation units
   - Parse POC codes to extract voltage levels
   
2. **Consolidate Duplicate Operators**
   - Same physical plant may have multiple operator/date entries
   - Keep most recent entry (by EffectiveStartDate) for each plant/unit
   
3. **Remove Decommissioned Units**
   - Filter out generators with DateDecommissioned populated
   - Keep only active generators (DateDecommissioned = null/NaN)
   
4. **Simplify Site Codes**
   - Extract 3-letter site code from POC codes (e.g., ARG1101 → ARG)
   - This groups units at the same substation together
   
5. **Aggregate by Site**
   - Sum nameplate capacity across all units at each site
   - Keep name and technology from highest voltage unit
   - Result: One row per generation site

6. **Enrich with Geographic Data**
   - Merge with sites.csv to get coordinates and site metadata
   - Fallback to pocs.csv for sites not in sites.csv (e.g., MKE)
   - Add X, Y (NZTM2000), lat, long (WGS84), and site type
   - Add zone and island from pocs.csv for all sites
   
7. **Standardize Output**
   - Rename columns to PyPSA conventions
   - Select relevant fields for modeling
   - Sort by site code for readability

OUTPUT FORMAT
-------------
CSV file with columns:
    - name: Plant/site name (from sites.csv reference)
    - site: 3-letter site code (e.g., ARG, BEN, MAN)
    - type: Site type (ACSTN, DCSTN, etc.)
    - carrier: Technology code (e.g., HYDRO, WIND, GEOTHERMAL)
    - tech: Full technology description
    - peaker: Peak generation flag (Y/N)
    - commissioned: Date commissioned (YYYY-MM-DD)
    - power: Total nameplate capacity in MW (aggregated by site)
    - volts: Connection voltage in kV (maximum at site)
    - X: NZTM2000 easting coordinate (meters)
    - Y: NZTM2000 northing coordinate (meters)
    - long: WGS84 longitude (degrees)
    - lat: WGS84 latitude (degrees)
    - zone: Grid zone/region (e.g., UNI, CAN, etc.)
    - island: Island location (NI = North Island, SI = South Island)

TYPICAL CONSOLIDATION EXAMPLES
-------------------------------
Example 1: Branch River (ARG)
    - Raw data: 3 separate entries (ARG110G1, ARG110G2, ARG110G3)
    - Each: 8.5 MW hydro units
    - Consolidated: 1 entry with 25.5 MW total (8.5 × 3)

Example 2: Clyde/Teviot (CYD)
    - Raw data: 4 units at CYD220 (Clyde dam) + 2 units at CYD033 (Teviot)
    - Clyde: 4 × 120 MW = 480 MW at 220 kV
    - Teviot: 2 × 30 MW = 60 MW at 33 kV
    - Consolidated: 1 entry "CYD" with 540 MW at 220 kV

CONFIGURATION
-------------
Reads YAML configuration file for paths:
    - Input: data/external/static/DispatchedGenerationPlant.csv
    - Output: data/processed/static/generators.csv
    - Reference: data/processed/static/sites.csv
    - Fallback: data/processed/static/pocs.csv
    - Config: config/nza_raw_static_data.yaml

LOGGING
-------
- Creates timestamped log files in logs/ directory
- Logs to both console (simple format) and file (detailed format)
- Log filename: nza_process_generation_data_YYYYMMDD_HHMMSS.log

USAGE
-----
    python nza_process_generation_data.py

DEPENDENCIES
------------
    - pandas: DataFrame operations
    - pyyaml: Configuration file parsing
    - pathlib: Cross-platform path handling

NOTES
-----
    - Decommissioned generators are completely removed from output
    - Site aggregation uses 3-letter codes (first 3 chars of POC code)
    - Voltage extracted from POC code chars 3-5 (e.g., ARG110 → 110 kV)
    - Capacity is summed across all active units at each site
    - Technology/name taken from highest voltage unit at each site
    - Geographic coordinates prioritize sites.csv, fallback to pocs.csv
    - Zone and island data obtained from pocs.csv
    - For duplicate site entries in pocs.csv, first entry is used
    - Missing data filled with NaN (numeric) or 'null' (string)

AUTHOR
------
    Philippe Bruneau

CREATED
-------
    2025-09-17

MODIFIED
--------
    2025-12-17 - Refactored with professional logging and formatting
               - Added comprehensive documentation
               - Implemented cross-platform path handling
               - Enhanced error handling and validation
               - Added geographic data enrichment from sites.csv/pocs.csv
               - Added zone and island fields from pocs.csv

VERSION
-------
    2.2.0
"""

from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import sys
import logging

import pandas as pd
import numpy as np
import yaml

from nza_root import ROOT_DIR


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    """
    Configure logging to write to both console and file.
    
    Args:
        log_dir: Directory where log files will be stored
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'nza_process_generation_data_{timestamp}.log'
    
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
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config(config_file: Path) -> Dict:
    """
    Read the YAML configuration file.

    Parameters
    ----------
    config_file : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        YAML configuration data. Returns empty dict if file not found or 
        parsing fails.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    yaml.YAMLError
        If YAML parsing fails.
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found."
        )
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config if config else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


# ============================================================================
# DATA PROCESSING
# ============================================================================

class GeneratorDataProcessor:
    """Processes and consolidates generator registry data."""
    
    # Column mappings for output
    COLUMN_MAPPING = {
        'PlantName': 'name_original',  # Will be replaced with sites.csv name
        'TechnologyCode': 'carrier',
        'PeakingPlantFlag': 'peaker',
        'site_simplified': 'site',
        'NameplateMegawatts': 'power',
        'DateCommissioned': 'commissioned',
        'Technology': 'tech'
    }
    
    # Final output columns
    OUTPUT_COLUMNS = [
        'name', 'site', 'type', 'carrier', 'tech', 
        'peaker', 'commissioned', 'power', 'volts',
        'X', 'Y', 'long', 'lat', 'zone', 'island'
    ]
    
    def __init__(self, input_path: Path, output_path: Path, 
                 sites_path: Path, pocs_path: Path):
        """
        Initialize the generator data processor.
        
        Args:
            input_path: Path to input CSV file (DispatchedGenerationPlant.csv)
            output_path: Path to output CSV file (generators.csv)
            sites_path: Path to sites reference file (sites.csv)
            pocs_path: Path to POCs reference file (pocs.csv)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.sites_path = Path(sites_path)
        self.pocs_path = Path(pocs_path)
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def process(self) -> pd.DataFrame:
        """
        Execute the complete processing pipeline.
        
        Returns:
            Processed generator DataFrame
        """
        # Load raw data
        df = self._load_raw_data()
        
        # Extract voltage information
        df = self._extract_voltage(df)
        
        # Consolidate duplicate operators
        df = self._consolidate_operators(df)
        
        # Remove decommissioned generators
        df = self._remove_decommissioned(df)
        
        # Create simplified site codes
        df = self._simplify_site_codes(df)
        
        # Aggregate by site
        df = self._aggregate_by_site(df)
        
        # Standardize output format (initial)
        df = self._standardize_output(df)
        
        # Enrich with geographic data
        df = self._enrich_with_geographic_data(df)
        
        # Final column selection and ordering
        df = self._finalize_output(df)
        
        # Save to file
        self._save_output(df)
        
        return df
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw generator data from CSV file."""
        print_section("Loading Raw Data")
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        df = pd.read_csv(self.input_path)
        logger.info(f"Loaded {len(df)} raw generator entries")
        logger.info(f"Input file: {self.input_path.name}")
        
        return df
    
    def _extract_voltage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract voltage from PointOfConnectionCode.
        
        POC code format: XXXYYY#, where:
            - XXX = 3-letter site code
            - YYY = voltage in kV (characters 3-5)
            - # = unit/circuit identifier
        
        Examples:
            - ARG1101 → 110 kV
            - ASB0661 → 66 kV
            - BEN2201 → 220 kV
        """
        print_section("Extracting Voltage Information")
        
        df = df.copy()
        df['volts'] = df['PointOfConnectionCode'].str[3:6].astype(int)
        
        voltage_summary = df['volts'].value_counts().sort_index()
        logger.info("Voltage distribution:")
        for voltage, count in voltage_summary.items():
            logger.info(f"  {voltage:3d} kV: {count:3d} units")
        
        return df
    
    def _consolidate_operators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidate duplicate generators from different operators.
        
        Same physical plant may have multiple entries with different:
            - Network participants (operators)
            - Effective date ranges
        
        Keep only the most recent entry (by EffectiveStartDate) for each
        unique plant/POC/unit combination.
        """
        print_section("Consolidating Duplicate Operators")
        
        initial_count = len(df)
        
        # Group by physical generator identity
        group_cols = ['PlantName', 'PointOfConnectionCode', 'UnitCode']
        
        # Sort by date descending (most recent first)
        df = df.sort_values('EffectiveStartDate', ascending=False)
        
        # Keep first (most recent) entry in each group
        df = df.groupby(group_cols, as_index=False).first()
        
        removed = initial_count - len(df)
        logger.info(f"Removed {removed} duplicate operator entries")
        logger.info(f"Remaining entries: {len(df)}")
        
        return df
    
    def _remove_decommissioned(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove decommissioned generators from dataset.
        
        Keep only active generators where DateDecommissioned is 'null' or NaN.
        """
        print_section("Removing Decommissioned Generators")
        
        initial_count = len(df)
        
        # Keep only active generators
        df = df[
            (df['DateDecommissioned'] == 'null') | 
            (df['DateDecommissioned'].isna())
        ]
        
        removed = initial_count - len(df)
        logger.info(f"Removed {removed} decommissioned generators")
        logger.info(f"Active generators: {len(df)}")
        
        return df
    
    def _simplify_site_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simplified 3-letter site codes.
        
        Extract first 3 characters from POC code to create site-level
        identifier that groups all units at the same substation.
        
        Examples:
            - ARG1101, ARG1102, ARG1103 → ARG
            - CYD2201, CYD2202, CYD0331 → CYD
        """
        print_section("Creating Simplified Site Codes")
        
        df = df.copy()
        df['site_simplified'] = df['PointOfConnectionCode'].str[:3]
        
        unique_sites = df['site_simplified'].nunique()
        logger.info(f"Created {unique_sites} unique site codes")
        
        return df
    
    def _aggregate_by_site(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate generators to site level.
        
        For each site (3-letter code):
            - Sum nameplate capacity across all units
            - Keep name from highest voltage unit
            - Keep maximum voltage level
            - Keep technology from highest voltage unit
        
        This consolidates multiple units at the same substation into a
        single site-level entry suitable for network modeling.
        """
        print_section("Aggregating Generators by Site")
        
        initial_count = len(df)
        consolidated = []
        
        for site_code, group in df.groupby('site_simplified'):
            # Sort by voltage descending (highest first)
            group = group.sort_values('volts', ascending=False)
            
            # Use highest voltage entry as template
            result = group.iloc[0].copy()
            
            # Aggregate capacity across all units
            result['NameplateMegawatts'] = group['NameplateMegawatts'].sum()
            
            # Keep maximum voltage
            result['volts'] = group['volts'].max()
            
            consolidated.append(result)
        
        df = pd.DataFrame(consolidated)
        
        logger.info(f"Consolidated {initial_count} generators into {len(df)} sites")
        logger.info(f"Average units per site: {initial_count / len(df):.1f}")
        
        return df
    
    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize output format with renamed columns and sorting.
        
        Apply column renaming per COLUMN_MAPPING, select relevant columns,
        and sort by site code for readability.
        """
        print_section("Standardizing Output Format")
        
        # Rename columns
        df = df.rename(columns=self.COLUMN_MAPPING)
        
        # Sort by site code
        df = df.sort_values('site').reset_index(drop=True)
        
        logger.info(f"Standardized dataset: {len(df)} generator sites")
        
        return df
    
    def _enrich_with_geographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich generator data with geographic coordinates and site metadata.
        
        Process:
        1. Load sites.csv reference data (for coordinates and site info)
        2. Load pocs.csv reference data (for zone, island, and fallback coordinates)
        3. Merge with sites.csv for primary geographic data
        4. For unmatched sites, look in pocs.csv for coordinates
        5. Merge with pocs.csv for zone and island data (all sites)
        6. Fill remaining missing data with NaN/null
        
        Returns:
            DataFrame with added columns: name, type, X, Y, long, lat, zone, island
        """
        print_header("ENRICHING WITH GEOGRAPHIC DATA", '=')
        logger.info("")
        
        # Load reference data
        print_section("Loading Reference Data")
        
        if not self.sites_path.exists():
            raise FileNotFoundError(f"Sites reference file not found: {self.sites_path}")
        
        sites_df = pd.read_csv(self.sites_path)
        logger.info(f"Loaded {len(sites_df)} sites from: {self.sites_path.name}")
        
        if not self.pocs_path.exists():
            raise FileNotFoundError(f"POCs reference file not found: {self.pocs_path}")
        
        pocs_df = pd.read_csv(self.pocs_path)
        logger.info(f"Loaded {len(pocs_df)} POCs from: {self.pocs_path.name}")
        logger.info("")
        
        # Select columns needed from sites.csv
        sites_cols = ['site', 'name', 'type', 'X', 'Y', 'long', 'lat']
        sites_ref = sites_df[sites_cols].copy()
        
        # Select columns needed from pocs.csv
        pocs_cols = ['site', 'name', 'X', 'Y', 'long', 'lat', 'zone', 'island']
        pocs_ref = pocs_df[pocs_cols].copy()
        
        # Remove duplicate sites in pocs (keep first entry - zone/island are same)
        pocs_ref = pocs_ref.drop_duplicates(subset='site', keep='first')
        logger.info(f"POCs after removing duplicates: {len(pocs_ref)} unique sites")
        logger.info("")
        
        print_section("Merging with Sites Reference")
        
        # Merge with sites.csv for primary geographic data
        df_merged = df.merge(
            sites_ref,
            on='site',
            how='left',
            suffixes=('_gen', '_sites')
        )
        
        # Count matches
        matched_sites = df_merged['name'].notna().sum()
        unmatched_sites = df_merged['name'].isna().sum()
        
        logger.info(f"Matched with sites.csv:     {matched_sites} sites")
        logger.info(f"Not found in sites.csv:     {unmatched_sites} sites")
        
        # For unmatched sites, try POCs for coordinates
        if unmatched_sites > 0:
            print_section("Checking POCs for Missing Sites (Coordinates)")
            
            # Get list of unmatched sites
            unmatched_mask = df_merged['name'].isna()
            unmatched_site_codes = df_merged.loc[unmatched_mask, 'site'].unique()
            
            logger.info(f"Looking up {len(unmatched_site_codes)} sites in POCs:")
            
            poc_matches = 0
            still_missing = 0
            
            for site_code in unmatched_site_codes:
                # Check if site exists in POCs
                poc_match = pocs_ref[pocs_ref['site'] == site_code]
                
                if not poc_match.empty:
                    # Found in POCs - copy coordinate data
                    poc_data = poc_match.iloc[0]
                    mask = (df_merged['site'] == site_code) & df_merged['name'].isna()
                    
                    df_merged.loc[mask, 'name'] = poc_data['name']
                    df_merged.loc[mask, 'X'] = poc_data['X']
                    df_merged.loc[mask, 'Y'] = poc_data['Y']
                    df_merged.loc[mask, 'long'] = poc_data['long']
                    df_merged.loc[mask, 'lat'] = poc_data['lat']
                    df_merged.loc[mask, 'type'] = 'POC'  # Mark as POC-sourced
                    
                    logger.info(f"  ⚠ {site_code}: Coordinates copied from pocs.csv")
                    poc_matches += 1
                else:
                    logger.warning(f"  ✗ {site_code}: NOT FOUND in sites.csv or pocs.csv")
                    still_missing += 1
            
            logger.info("")
            logger.info(f"Found in pocs.csv:          {poc_matches} sites")
            logger.info(f"Still missing:              {still_missing} sites")
        
        # Now merge zone and island from POCs for ALL sites
        print_section("Adding Zone and Island Data from POCs")
        
        # Select only site, zone, island from pocs_ref
        pocs_zone_island = pocs_ref[['site', 'zone', 'island']].copy()
        
        # Merge with current dataframe
        df_merged = df_merged.merge(
            pocs_zone_island,
            on='site',
            how='left'
        )
        
        # Count matches for zone/island
        zone_matched = df_merged['zone'].notna().sum()
        zone_missing = df_merged['zone'].isna().sum()
        
        logger.info(f"Sites with zone/island data:    {zone_matched}")
        logger.info(f"Sites missing zone/island:      {zone_missing}")
        
        if zone_missing > 0:
            missing_zone_sites = df_merged[df_merged['zone'].isna()]['site'].tolist()
            logger.info(f"  Sites without zone/island: {', '.join(missing_zone_sites)}")
        
        # Fill remaining missing data
        print_section("Filling Missing Data")
        
        missing_count = 0
        
        # Fill missing names with 'null'
        if df_merged['name'].isna().any():
            missing_mask = df_merged['name'].isna()
            missing_sites = df_merged.loc[missing_mask, 'site'].tolist()
            df_merged.loc[missing_mask, 'name'] = 'null'
            df_merged.loc[missing_mask, 'type'] = 'null'
            logger.info(f"Name/type: Filled 'null' for sites: {', '.join(missing_sites)}")
            missing_count += len(missing_sites)
        
        # Fill missing zone/island with 'null'
        if df_merged['zone'].isna().any():
            missing_mask = df_merged['zone'].isna()
            missing_sites = df_merged.loc[missing_mask, 'site'].tolist()
            df_merged.loc[missing_mask, 'zone'] = 'null'
            df_merged.loc[missing_mask, 'island'] = 'null'
            logger.info(f"Zone/island: Filled 'null' for sites: {', '.join(missing_sites)}")
        
        # Numeric fields (X, Y, long, lat) are already NaN from merge
        if df_merged[['X', 'Y', 'long', 'lat']].isna().any(axis=1).any():
            logger.info("Coordinates: Numeric fields (X, Y, long, lat) remain as NaN where missing")
        
        logger.info("")
        logger.info("✓ Geographic enrichment complete")
        
        return df_merged
    
    def _finalize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize output with column selection and ordering.
        
        Select OUTPUT_COLUMNS in the correct order.
        """
        print_section("Finalizing Output")
        
        # Select and order columns
        df = df[self.OUTPUT_COLUMNS]
        
        logger.info(f"Output columns: {', '.join(self.OUTPUT_COLUMNS)}")
        logger.info(f"Final dataset: {len(df)} generator sites")
        
        return df
    
    def _save_output(self, df: pd.DataFrame) -> None:
        """Save processed data to CSV file."""
        print_section("Saving Output")
        
        df.to_csv(self.output_path, index=False)
        logger.info(f"✓ Saved: {self.output_path.name}")
        logger.info(f"  Records: {len(df)}")
        logger.info(f"  Total capacity: {df['power'].sum():.0f} MW")


# ============================================================================
# SUMMARY REPORTING
# ============================================================================

def print_processing_summary(df: pd.DataFrame) -> None:
    """
    Print comprehensive summary of processed generator data.
    
    Args:
        df: Processed generator DataFrame
    """
    print_header("PROCESSING SUMMARY", '=')
    logger.info("")
    
    # Overall statistics
    total_generators = len(df)
    total_capacity = df['power'].sum()
    
    logger.info(f"Total generator sites:     {total_generators}")
    logger.info(f"Total nameplate capacity:  {total_capacity:.0f} MW")
    logger.info(f"Average capacity per site: {total_capacity/total_generators:.1f} MW")
    logger.info("")
    
    # Data completeness
    print_section("Data Completeness")
    
    complete_coords = df[['X', 'Y', 'long', 'lat']].notna().all(axis=1).sum()
    missing_coords = total_generators - complete_coords
    
    complete_zone = (df['zone'] != 'null').sum()
    missing_zone = total_generators - complete_zone
    
    logger.info(f"Sites with complete coordinates:   {complete_coords}")
    logger.info(f"Sites with missing coordinates:    {missing_coords}")
    
    if missing_coords > 0:
        missing_sites = df[df[['X', 'Y']].isna().any(axis=1)]['site'].tolist()
        logger.info(f"  Missing coordinates: {', '.join(missing_sites)}")
    
    logger.info(f"Sites with zone/island data:       {complete_zone}")
    logger.info(f"Sites missing zone/island:         {missing_zone}")
    
    if missing_zone > 0:
        missing_sites = df[df['zone'] == 'null']['site'].tolist()
        logger.info(f"  Missing zone/island: {', '.join(missing_sites)}")
    
    logger.info("")
    
    # Island distribution
    print_section("Distribution by Island")
    
    island_summary = df[df['island'] != 'null'].groupby('island').agg({
        'site': 'count',
        'power': 'sum'
    }).round(1)
    island_summary.columns = ['Sites', 'Total MW']
    
    logger.info(f"{'Island':<10} {'Sites':>8} {'Total MW':>12}")
    logger.info("-" * 40)
    for island, row in island_summary.iterrows():
        logger.info(f"{island:<10} {row['Sites']:>8.0f} {row['Total MW']:>12.1f}")
    logger.info("")
    
    # Technology breakdown
    print_section("Capacity by Technology")
    
    tech_summary = df.groupby('tech').agg({
        'site': 'count',
        'power': 'sum'
    }).round(1)
    tech_summary.columns = ['Sites', 'Total MW']
    tech_summary['Avg MW'] = (tech_summary['Total MW'] / tech_summary['Sites']).round(1)
    tech_summary = tech_summary.sort_values('Total MW', ascending=False)
    
    # Format as table
    logger.info(f"{'Technology':<30} {'Sites':>8} {'Total MW':>12} {'Avg MW':>10}")
    logger.info("-" * 80)
    for tech, row in tech_summary.iterrows():
        logger.info(
            f"{tech:<30} {row['Sites']:>8.0f} "
            f"{row['Total MW']:>12.1f} {row['Avg MW']:>10.1f}"
        )
    logger.info("")
    
    # Voltage level breakdown
    print_section("Capacity by Voltage Level")
    
    voltage_summary = df.groupby('volts').agg({
        'site': 'count',
        'power': 'sum'
    }).round(1)
    voltage_summary.columns = ['Sites', 'Total MW']
    voltage_summary = voltage_summary.sort_index()
    
    logger.info(f"{'Voltage (kV)':<15} {'Sites':>10} {'Total MW':>15}")
    logger.info("-" * 50)
    for voltage, row in voltage_summary.iterrows():
        logger.info(
            f"{voltage:<15} {row['Sites']:>10.0f} {row['Total MW']:>15.1f}"
        )
    logger.info("")
    
    # Largest sites
    print_section("Top 10 Generation Sites by Capacity")
    
    top10 = df.nlargest(10, 'power')[['site', 'name', 'island', 'tech', 'power', 'volts']]
    
    logger.info(f"{'Site':<6} {'Name':<25} {'Island':<8} {'Technology':<15} {'MW':>8} {'kV':>6}")
    logger.info("-" * 80)
    for _, row in top10.iterrows():
        logger.info(
            f"{row['site']:<6} {row['name']:<25} {row['island']:<8} "
            f"{row['tech']:<15} {row['power']:>8.0f} {row['volts']:>6}"
        )
    logger.info("")


def print_consolidation_examples(df: pd.DataFrame) -> None:
    """
    Print examples of site consolidation for verification.
    
    Args:
        df: Processed generator DataFrame
    """
    print_header("CONSOLIDATION EXAMPLES", '=')
    logger.info("")
    
    # Example 1: ARG (Branch River) - multiple small hydro units
    print_section("Example 1: ARG (Branch River)")
    arg = df[df['site'] == 'ARG']
    if not arg.empty:
        row = arg.iloc[0]
        logger.info(f"Site:       {row['site']}")
        logger.info(f"Name:       {row['name']}")
        logger.info(f"Type:       {row['type']}")
        logger.info(f"Zone:       {row['zone']}")
        logger.info(f"Island:     {row['island']}")
        logger.info(f"Technology: {row['tech']}")
        logger.info(f"Capacity:   {row['power']:.1f} MW")
        logger.info(f"Voltage:    {row['volts']} kV")
        logger.info(f"Location:   {row['lat']:.4f}°, {row['long']:.4f}°")
        logger.info("")
        logger.info("Note: Multiple small hydro units aggregated into single site entry")
    else:
        logger.info("ARG site not found in processed data")
    logger.info("")
    
    # Example 2: CYD (Clyde/Teviot) - multiple voltage levels
    print_section("Example 2: CYD (Clyde + Teviot)")
    cyd = df[df['site'] == 'CYD']
    if not cyd.empty:
        row = cyd.iloc[0]
        logger.info(f"Site:       {row['site']}")
        logger.info(f"Name:       {row['name']}")
        logger.info(f"Type:       {row['type']}")
        logger.info(f"Zone:       {row['zone']}")
        logger.info(f"Island:     {row['island']}")
        logger.info(f"Technology: {row['tech']}")
        logger.info(f"Capacity:   {row['power']:.1f} MW")
        logger.info(f"Voltage:    {row['volts']} kV (maximum)")
        logger.info(f"Location:   {row['lat']:.4f}°, {row['long']:.4f}°")
        logger.info("")
        logger.info("Note: Clyde (220 kV) and Teviot (33 kV) units combined")
    else:
        logger.info("CYD site not found in processed data")
    logger.info("")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    
    # Convert ROOT_DIR to Path and setup paths
    root_path = Path(ROOT_DIR)
    config_file = root_path / "config" / "nza_raw_static_data.yaml"
    log_dir = root_path / "logs"
    
    # Set up logging
    setup_logging(log_dir)
    
    # Print startup banner
    print_header("NZA GENERATOR DATA PROCESSING")
    logger.info("")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    try:
        # Load configuration
        print_header("LOADING CONFIGURATION", '=')
        logger.info("")
        logger.info(f"Configuration file: {config_file}")
        
        config = load_config(config_file)
        
        if not config:
            logger.error("✗ Failed to load configuration")
            return 1
        
        # Define file paths
        input_file = root_path / config['inpdir'] / config['gen_inp']
        output_file = root_path / config['outdir'] / config['gen_out']
        sites_file = root_path / config['outdir'] / config['sites_out']
        pocs_file = root_path / config['outdir'] / config['pocs_out']
        
        logger.info(f"Input:      {input_file}")
        logger.info(f"Output:     {output_file}")
        logger.info(f"Sites ref:  {sites_file}")
        logger.info(f"POCs ref:   {pocs_file}")
        logger.info("")
        
        # Process the data
        print_header("PROCESSING GENERATOR DATA", '=')
        logger.info("")
        
        processor = GeneratorDataProcessor(
            input_file, 
            output_file,
            sites_file,
            pocs_file
        )
        df = processor.process()
        
        # Print summaries
        print_processing_summary(df)
        print_consolidation_examples(df)
        
        # Final message
        print_header("PROCESSING COMPLETE", '=')
        logger.info("")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info("")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Log saved to: {log_dir}")
        logger.info("")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"✗ File not found: {e}")
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