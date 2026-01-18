#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nza_process_static_data

Preprocesses, filters and reformats static raw data pertaining to the
NZ power network. Static data includes fixed assets, infrastructure
specs, geographic locations, etc. which change infrequently (> 12 months).
it is assumed that the raw .csv data files have already been downloaded and
are available in the directories specified in the appropriate .yaml file.
At the time of writing, the config directory and file were:
/home/pbrun/pypsa-nza/config/nza_raw_static_data.yaml"

NOTE:
The Transpower data (Sites and Transmission Lines, Cook Straight Protection
Zone, etc. ...) must be downloaded manually as no direct web access is available
without a license.

Electricity Authority data are generally directly accessible on their website on
the"Wholesale Datasets" section. An exception is the "Network Supply Points Table"
which must also be downloaded manually at:
https://www.emi.ea.govt.nz/Wholesale/Reports/R_NSPL_DR?_si=v|3


Data sources include:
- Network Supply Points (POCs) from EA/EMI
- Transpower Sites (AC/DC stations, Tees)
- Transmission Lines data

The cleaned data is written to new files in the processed data directory for
PyPSA model input.

Author: Phillippe Bruneau
Created on Sat Sep 20 19:33:57 2025
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import nztm_geod as gd
import yaml


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        logger.info("Processing POC data...")

        # Remove embedded POCs and unused columns
        df_clean = self._remove_embedded_pocs(df)
        df_clean = self._drop_unused_columns(df_clean)

        # Convert coordinates
        df_clean = self._add_coordinates(df_clean)

        # Remove duplicates and add parsed fields
        df_clean = df_clean.drop_duplicates(subset=['POC code'], keep='first')
        df_clean = self._parse_poc_fields(df_clean)

        # Clean up description field
        df_clean = self._clean_descriptions(df_clean)

        # Final cleanup and column ordering
        df_clean = self._finalize_dataframe(df_clean)

        logger.info(f"Processed {len(df_clean)} POC records")
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
            Processed sites dataframe
        """
        logger.info("Processing Sites data...")

        # Get unique POCs and convert coordinates
        pocs = df["MXLOCATION"].unique()
        lat, long = self._extract_coordinates(df, pocs)

        # Clean up dataframe
        df_clean = df.drop(columns=['status', 'GlobalID', 'OBJECTID', 'MXLOCATION'])

        # Create sites dataframe and merge
        df_sites = pd.DataFrame({'poc': pocs, 'long': long, 'lat': lat})
        df_result = pd.merge(df_clean, df_sites, left_index=True, right_index=True)

        logger.info(f"Processed {len(df_result)} site records")
        return df_result

    def _extract_coordinates(self, df: pd.DataFrame, pocs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and convert substation coordinates."""
        eastings = np.array(df.X.tolist())
        northings = np.array(df.Y.tolist())
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
        logger.info("Processing Transmission Lines data...")

        # Clean up dataframe
        df_clean = df.drop(columns=self.COLUMNS_TO_DROP)
        df_clean = df_clean.sort_values(by=['designvolt'])

        # Parse line names to extract bus connections
        df_clean = self._parse_line_connections(df_clean)

        # Rename columns
        df_clean = df_clean.rename(columns=self.COLUMN_MAPPING)

        logger.info(f"Processed {len(df_clean)} transmission line records")
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
        self.poc_processor = POCProcessor(self.coord_converter)
        self.sites_processor = SitesProcessor(self.coord_converter)
        self.transmission_processor = TransmissionLinesProcessor()

    def process_pocs(self) -> None:
        """Process POC data and save to output file."""
        try:
            input_path = self._get_input_path(self.config['pocs_inp'])
            output_path = self._get_output_path(self.config['pocs_out'])

            logger.info(f"Reading POC data from: {input_path}")
            df_poc = pd.read_csv(input_path)

            # Process the data
            df_processed = self.poc_processor.process(df_poc)

            # Save to output file
            logger.info(f"Writing POC data to: {output_path}")
            df_processed.to_csv(output_path, index=True)

        except Exception as e:
            logger.error(f"Error processing POC data: {e}")
            raise

    def process_sites(self) -> None:
        """Process sites data and save to output file."""
        try:
            input_path = self._get_input_path(self.config['sites_inp'])
            output_path = self._get_output_path(self.config['sites_out'])

            logger.info(f"Reading Sites data from: {input_path}")
            df_sites = pd.read_csv(input_path)

            # Process the data
            df_processed = self.sites_processor.process(df_sites)

            # Save to output file
            logger.info(f"Writing Sites data to: {output_path}")
            df_processed.to_csv(output_path, index=True)

        except Exception as e:
            logger.error(f"Error processing Sites data: {e}")
            raise

    def process_transmission_lines(self) -> None:
        """Process transmission lines data and save to output file."""
        try:
            input_path = self._get_input_path(self.config['trans_inp'])
            output_path = self._get_output_path(self.config['trans_out'])

            logger.info(f"Reading Transmission Lines data from: {input_path}")
            df_trans = pd.read_csv(input_path)

            # Process the data
            df_processed = self.transmission_processor.process(df_trans)

            # Save to output file
            logger.info(f"Writing Transmission Lines data to: {output_path}")
            df_processed.to_csv(output_path, index=True)

        except Exception as e:
            logger.error(f"Error processing Transmission Lines data: {e}")
            raise

    def process_all(self) -> None:
        """Process all data types."""
        logger.info("Starting grid data processing...")

        self.process_pocs()
        self.process_sites()
        self.process_transmission_lines()

        logger.info("Grid data processing completed successfully!")

    def _get_input_path(self, filename: str) -> Path:
        """Get full input file path."""
        return Path(self.config['rootdir']) / self.config['inpdir'] / filename

    def _get_output_path(self, filename: str) -> Path:
        """Get full output file path."""
        output_path = Path(self.config['rootdir']) / self.config['outdir'] / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        return output_path


def load_config(config_file: str) -> dict:
    """
    Read the YAML configuration file.

    Parameters
    ----------
        config_file (str): Path to the YAML configuration file.

    Returns
    -------
        config (dict): YAML configuration data in a dictionary.
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}

def main():
    """Main function to run the grid data processing."""
    try:
        # Load configuration
        config = load_config("/home/pbrun/pypsa-nza/config/nza_raw_static_data.yaml")

        # Initialize processor
        processor = GridDataProcessor(config)

        # Process all data
        #processor.process_all()
        processor.process_pocs()
        processor.process_sites()
        processor.process_transmission_lines()


    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise


if __name__ == '__main__':
    main()