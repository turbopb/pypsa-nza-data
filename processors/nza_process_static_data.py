#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    nza_process_static_data
    
    Preprocesses, filters and reformats static raw data pertaining to the
    NZ power network. Static data includes fixed assets, infrastructure
    specs, geographic locations, etc. which change infrequently (> 12 months).
    It is assumed that the raw .csv data files have already been downloaded and
    are available in the directories specified in the appropriate .yaml file.
    At the time of writing, the config directory and file were:
    /home/pbrun/pypsa-nza/config/nza_raw_static_data.yaml"
    
    NOTE:
    The Transpower data (Sites and Transmission Lines, Cook Straight Protection
    Zone, etc. ...) must be downloaded manually as no direct web access is available
    without a license.
    
    Electricity Authority data are generally directly accessible on their website 
    within the "Wholesale Datasets" section. An exception is the "Network Supply 
    Points Table" which must also be downloaded manually at:
    https://www.emi.ea.govt.nz/Wholesale/Reports/R_NSPL_DR?_si=v|3
    
    
    Data sources include:
    - Network Supply Points (POCs) from EA/EMI
    - Transpower Sites (AC/DC stations, Tees) https://data-transpower.opendata.arcgis.com/
    - Transmission Lines data
    
    The cleaned data is written to new files in the processed data directory for
    PyPSA model input:
    - pocs.csv: Processed POC data (Points of Connection)
    - sites.csv: Processed Sites data (Substations) - columns: 'site', 'name'
    - trans_lines.csv: Processed transmission line data
    
    LOGGING:
    - Creates timestamped log files in the logs/ directory
    - Logs to both console (simple format) and file (detailed format)
    - Log files include full processing details for debugging
    
    PATH HANDLING:
    - Uses pathlib.Path throughout for cross-platform compatibility
    - Works identically on Windows and Linux
    - YAML config should use forward slashes (/) - works on both platforms
    - All paths converted to Path objects at initialization
    
    Author: Phillippe Bruneau
    Created on Sat Sep 20 19:33:57 2025
    Modified: Dec 17, 2025 - Unified path handling, logging, and column naming
"""


from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys

import logging
import numpy as np
import pandas as pd
import yaml

from utils.geospatial_utils import nztm_geod as gd
from nza_root import ROOT_DIR

# ROOT_DIR should be a Path object or string - will be converted to Path below

# Logging will be configured in setup_logging() after paths are established
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
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'nza_process_static_data_{timestamp}.log'
    
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


class CoordinateConverter:
    """Handles coordinate conversion between NZTM and latitude/longitude."""

    def __init__(self):
        """Initialize the Transverse Mercator projection."""
        self.tm = gd.TransverseMercator()
        gd.init_tm_proj(self.tm)

    def nztm_to_latlong(self, eastings: np.ndarray, northings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert NZTM coordinates to latitude and longitude in degrees.

        Args:
            eastings: NZTM easting coordinates in meters
            northings: NZTM northing coordinates in meters

        Returns:
            Tuple of (latitude, longitude) arrays in degrees
        """
        try:
            lat_rad, long_rad = gd.tm_geod(self.tm, eastings, northings)
            return np.rad2deg(lat_rad), np.rad2deg(long_rad)
        except Exception as e:
            logger.error(f"Coordinate conversion failed: {e}")
            raise


class POCProcessor:
    """Processes Points of Connection (POC) data."""

    # Columns to drop from POC data
    COLUMNS_TO_DROP = [
        'Current flag', 'NSP', 'NSP replaced by', 'Network participant',
        'Embedded under POC code', 'Embedded under network participant',
        'Reconciliation type', 'Network reporting region ID',
        'Network reporting region', 'Start date', 'Start TP', 'End date',
        'End TP', 'SB ICP', 'Balancing code', 'MEP', 'Responsible participant',
        'Certification expiry', 'Metering information exemption expiry date'
    ]

    # Column mapping for renaming
    COLUMN_MAPPING = {
        "NZTM easting": "X",
        'NZTM northing': "Y",
        "Zone": "zone",
        "Island": "island"
    }

    def __init__(self, coordinate_converter: CoordinateConverter):
        self.coord_converter = coordinate_converter

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process POC dataframe with cleaning, coordinate conversion, and restructuring.

        Args:
            df: Raw POC dataframe

        Returns:
            Processed POC dataframe
        """
        print_section("Processing POC (Points of Connection) Data")
        
        logger.info(f"Input records: {len(df)}")

        # Remove embedded POCs and unused columns
        df_clean = self._remove_embedded_pocs(df)
        logger.info(f"After removing embedded POCs: {len(df_clean)} records")
        
        df_clean = self._drop_unused_columns(df_clean)

        # Convert coordinates
        logger.info("Converting NZTM coordinates to latitude/longitude...")
        df_clean = self._add_coordinates(df_clean)

        # Remove duplicates and add parsed fields
        df_clean = df_clean.drop_duplicates(subset=['POC code'], keep='first')
        df_clean = self._parse_poc_fields(df_clean)

        # Clean up description field
        df_clean = self._clean_descriptions(df_clean)

        # Final cleanup and column ordering
        df_clean = self._finalize_dataframe(df_clean)

        logger.info(f"Successfully processed {len(df_clean)} POC records")
        logger.info(f"Output columns: {', '.join(df_clean.columns.tolist())}")
        
        return df_clean.reset_index(drop=True)

    def _remove_embedded_pocs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove embedded POCs, keeping only non-embedded ones."""
        return df[df['Embedded under POC code'].isna()].copy()

    def _drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unused columns from dataframe."""
        existing_columns = [col for col in self.COLUMNS_TO_DROP if col in df.columns]
        return df.drop(columns=existing_columns)

    def _add_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add latitude and longitude coordinates."""
        eastings = np.array(df['NZTM easting'].tolist())
        northings = np.array(df['NZTM northing'].tolist())

        lat, long = self.coord_converter.nztm_to_latlong(eastings, northings)

        df = df.copy()
        df['lat'] = lat
        df['long'] = long
        return df

    def _parse_poc_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse POC code into component fields."""
        df = df.copy()
        poc_codes = df["POC code"].tolist()

        df['poc'] = poc_codes
        df['site'] = [code[:3] for code in poc_codes]
        df['volts'] = [code[3:6] for code in poc_codes]
        df['poc_count'] = [code[6] if len(code) > 6 else '' for code in poc_codes]

        return df

    def _clean_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format description field."""
        df = df.copy()
        descriptions = [str.title(desc) for desc in df["Description"].tolist()]
        df['name'] = descriptions
        return df

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and column reordering."""
        # Drop redundant columns
        df = df.drop(columns=["POC code", "Description", "X flow", "I flow"])

        # Rename columns
        df = df.rename(columns=self.COLUMN_MAPPING, errors="raise")

        # Remove any remaining duplicates
        df = df.drop_duplicates()

        # Reorder columns
        column_order = ['name', 'poc', 'site', 'zone', 'island', 'volts',
                       'X', 'Y', 'long', 'lat']
        return df[column_order]


class SitesProcessor:
    """Processes Transpower Sites data."""

    def __init__(self, coordinate_converter: CoordinateConverter):
        self.coord_converter = coordinate_converter

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process sites dataframe.

        Args:
            df: Raw sites dataframe

        Returns:
            Processed sites dataframe with renamed columns
        """
        print_section("Processing Sites Data")

        # Get unique site codes and convert coordinates
        site_codes = df["MXLOCATION"].unique()
        logger.info(f"Found {len(site_codes)} unique sites in input data")
        
        lat, long = self._extract_coordinates(df, site_codes)

        # Clean up dataframe
        df_clean = df.drop(columns=['status', 'GlobalID', 'OBJECTID', 'MXLOCATION'])

        # Create sites dataframe with renamed columns
        # Rename 'poc' -> 'site' for consistency
        df_sites = pd.DataFrame({'site': site_codes, 'long': long, 'lat': lat})
        df_result = pd.merge(df_clean, df_sites, left_index=True, right_index=True)
        
        # If 'description' column exists, rename it to 'name'
        if 'description' in df_result.columns:
            df_result = df_result.rename(columns={'description': 'name'})
            logger.info("Renamed column: 'description' -> 'name'")
        
        logger.info(f"Successfully processed {len(df_result)} site records")
        logger.info(f"Output columns: {', '.join(df_result.columns.tolist())}")
        
        return df_result

    def _extract_coordinates(self, df: pd.DataFrame, site_codes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and convert substation coordinates."""
        eastings = np.array(df.X.tolist())
        northings = np.array(df.Y.tolist())
        logger.info("Converting NZTM coordinates to latitude/longitude...")
        return self.coord_converter.nztm_to_latlong(eastings, northings)


class TransmissionLinesProcessor:
    """Processes transmission lines data."""

    COLUMNS_TO_DROP = ['GlobalID', 'OBJECTID', 'status']
    COLUMN_MAPPING = {
        "MXLOCATION": "name",
        "designvolt": "volts",
        "Symbol": "tag",
        "Shape__Length": "length"
    }

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process transmission lines dataframe.

        Args:
            df: Raw transmission lines dataframe

        Returns:
            Processed transmission lines dataframe
        """
        print_section("Processing Transmission Lines Data")
        
        logger.info(f"Input records: {len(df)}")

        # Clean up dataframe
        df_clean = df.drop(columns=self.COLUMNS_TO_DROP)
        df_clean = df_clean.sort_values(by=['designvolt'])
        logger.info("Sorted by voltage level")

        # Parse line names to extract bus connections
        logger.info("Parsing line names to extract bus connections...")
        df_clean = self._parse_line_connections(df_clean)

        # Rename columns
        df_clean = df_clean.rename(columns=self.COLUMN_MAPPING)

        logger.info(f"Successfully processed {len(df_clean)} transmission line records")
        logger.info(f"Output columns: {', '.join(df_clean.columns.tolist())}")
        
        return df_clean.reset_index(drop=True)

    def _parse_line_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse line names to extract bus connections."""
        df = df.copy()

        bus0_list = []
        bus1_list = []

        for line_name in df["MXLOCATION"]:
            try:
                parts = line_name.split("-")
                if len(parts) >= 2:
                    bus0_list.append(parts[0])
                    bus1_list.append(parts[1])
                else:
                    logger.warning(f"Unexpected line name format: {line_name}")
                    bus0_list.append("")
                    bus1_list.append("")
            except Exception as e:
                logger.error(f"Error parsing line name {line_name}: {e}")
                bus0_list.append("")
                bus1_list.append("")

        df['bus0'] = bus0_list
        df['bus1'] = bus1_list

        return df


class GridDataProcessor:
    """Main class for processing grid static data."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the grid data processor.

        Args:
            config: Configuration dictionary containing file paths and settings
        """
        self.config = config
        self.coord_converter = CoordinateConverter()
        
        # Convert all path strings to Path objects at initialization (cross-platform)
        # Use .resolve() to get absolute paths - works identically on Windows and Linux
        self.root_dir = Path(config['rootdir']).resolve()
        self.input_dir = self.root_dir / config['inpdir']
        self.output_dir = self.root_dir / config['outdir']
        self.log_dir = self.root_dir / "logs"
        
        # Set up logging to both console and file
        setup_logging(self.log_dir)
        
        # Print startup banner
        print_header("NZA STATIC DATA PROCESSING")
        logger.info("")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log the paths being used (helpful for debugging)
        logger.info(f"Root directory:   {self.root_dir}")
        logger.info(f"Input directory:  {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Log directory:    {self.log_dir}")
        logger.info("")
        
        # Initialize processors
        self.poc_processor = POCProcessor(self.coord_converter)
        self.sites_processor = SitesProcessor(self.coord_converter)
        self.transmission_processor = TransmissionLinesProcessor()

    def process_pocs(self) -> None:
        """Process POC data and save to output file."""
        try:
            # Use Path / operator for cross-platform path construction
            input_path = self.input_dir / self.config['pocs_inp']
            output_path = self.output_dir / self.config['pocs_out']

            print_header("PROCESSING POCs (POINTS OF CONNECTION)", '=')
            logger.info(f"Input file:  {input_path}")
            logger.info(f"Output file: {output_path}")
            logger.info("")
            
            # Read data
            df_poc = pd.read_csv(input_path)

            # Process the data
            df_processed = self.poc_processor.process(df_poc)

            # Save to output file
            logger.info("")
            logger.info(f"Writing {len(df_processed)} records to: {output_path}")
            df_processed.to_csv(output_path, index=True)
            
            logger.info("✓ POC processing completed successfully")
            logger.info("")

        except Exception as e:
            logger.error(f"✗ Error processing POC data: {e}")
            raise

    def process_sites(self) -> None:
        """Process sites data and save to output file."""
        try:
            input_path = self.input_dir / self.config['sites_inp']
            output_path = self.output_dir / self.config['sites_out']

            print_header("PROCESSING SITES (SUBSTATIONS)", '=')
            logger.info(f"Input file:  {input_path}")
            logger.info(f"Output file: {output_path}")
            logger.info("")

            # Read data
            df_sites = pd.read_csv(input_path)

            # Process the data
            df_processed = self.sites_processor.process(df_sites)

            # Save to output file
            logger.info("")
            logger.info(f"Writing {len(df_processed)} records to: {output_path}")
            df_processed.to_csv(output_path, index=True)
            
            logger.info("✓ Sites processing completed successfully")
            logger.info("")

        except Exception as e:
            logger.error(f"✗ Error processing Sites data: {e}")
            raise

    def process_transmission_lines(self) -> None:
        """Process transmission lines data and save to output file."""
        try:
            input_path = self.input_dir / self.config['trans_inp']
            output_path = self.output_dir / self.config['trans_out']

            print_header("PROCESSING TRANSMISSION LINES", '=')
            logger.info(f"Input file:  {input_path}")
            logger.info(f"Output file: {output_path}")
            logger.info("")

            # Read data
            df_trans = pd.read_csv(input_path)

            # Process the data
            df_processed = self.transmission_processor.process(df_trans)

            # Save to output file
            logger.info("")
            logger.info(f"Writing {len(df_processed)} records to: {output_path}")
            df_processed.to_csv(output_path, index=True)
            
            logger.info("✓ Transmission lines processing completed successfully")
            logger.info("")

        except Exception as e:
            logger.error(f"✗ Error processing Transmission Lines data: {e}")
            raise

    def process_all(self) -> None:
        """Process all data types."""
        self.process_pocs()
        self.process_sites()
        self.process_transmission_lines()

        print_header("ALL PROCESSING COMPLETED SUCCESSFULLY", '=')
        logger.info("")
        logger.info("Output files created:")
        logger.info(f"  • {self.output_dir / self.config['pocs_out']}")
        logger.info(f"  • {self.output_dir / self.config['sites_out']}")
        logger.info(f"  • {self.output_dir / self.config['trans_out']}")
        logger.info("")
        logger.info(f"Log file saved to: {self.log_dir}")
        logger.info("")


def load_config(config_file: Path) -> dict:
    """
    Read the YAML configuration file.

    Parameters
    ----------
        config_file: Path to the YAML configuration file (Path object or string)

    Returns
    -------
        config (dict): YAML configuration data in a dictionary.
    """
    # Convert to Path object if string
    config_path = Path(config_file)
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"✗ Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"✗ Error parsing YAML file: {e}")
        return {}


def main():
    """Main function to run the grid data processing."""
    try:
        # Convert ROOT_DIR to Path object and construct config file path
        # Use Path / operator instead of string concatenation
        root_path = Path(ROOT_DIR)
        config_file = root_path / "config" / "nza_raw_static_data.yaml"
        
        # Load configuration (logging not set up yet, so just use print for errors)
        config = load_config(config_file)
        
        if not config:
            print(f"✗ Failed to load configuration from: {config_file}")
            print("Exiting.")
            return 1

        # Initialize processor (sets up logging)
        processor = GridDataProcessor(config)

        # Process all data
        processor.process_pocs()
        processor.process_sites()
        processor.process_transmission_lines()
        
        return 0

    except KeyboardInterrupt:
        logger.error("")
        logger.error("✗ Processing interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"✗ Application failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code if exit_code else 0)