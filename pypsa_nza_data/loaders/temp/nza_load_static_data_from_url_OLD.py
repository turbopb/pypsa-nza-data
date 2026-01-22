#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nza_download_static_data_from_url.py

Simple data downloader for PyPSA-NZA - ALL configuration is in YAML file.

DESCRIPTION
-----------
This script downloads datasets specified in config/nza_data_sources.yaml.
All configuration (URLs, directories, filenames, datasets) is in the YAML file,
making it easy to modify without touching this code.

Always downloads fresh data (files are small, want latest versions).

WORKFLOW
--------
The script follows this process:

1. INITIALIZATION
   - Load configuration from YAML file (nza_download_data_from_source.yaml)
   - Validate required sections (directories, data_sources, datasets)
   - Set up logging to both console and timestamped log file
   - Create output directories if they don't exist

2. CONFIGURATION LOADING
   - Parse directory paths (relative to ROOT_DIR)
   - Load data source URLs (Transpower ArcGIS, Electricity Authority)
   - Identify enabled datasets to download
   - Set validation parameters (minimum rows, file formats)

3. DOWNLOAD PROCESS (for each enabled dataset)
   - Check dataset source (Transpower vs Electricity Authority)
   - Construct appropriate download URL
   - Attempt direct download:
     * For Transpower: Try direct CSV/GeoJSON download
     * For EA: Direct file download from Azure blob storage
   - On failure (404): Fall back to ArcGIS REST API query
   - Stream download to output file
   - Log progress and any errors

4. VALIDATION
   - Check file exists and is not empty
   - Validate file format (CSV or GeoJSON)
   - Count rows/features and columns
   - Compare against minimum expected rows
   - Compute file size statistics
   - Report validation results

5. ARCHIVING
   - Create timestamped copy in archive directory
   - Preserves download history for auditing
   - Format: filename_YYYYMMDD_HHMMSS.extension

6. REPORTING
   - Track statistics (total, downloaded, failed)
   - Generate summary table with rows and file sizes
   - Write detailed log file
   - Report total elapsed time

ERROR HANDLING
   - Network errors: Logged with details, download marked as failed
   - File errors: Validation catches empty/corrupt files
   - Configuration errors: Clear error messages about missing sections
   - Graceful handling of keyboard interrupts

DATA SOURCES
   - Transpower: ArcGIS Open Data portal with REST API fallback
   - Electricity Authority: EMI datasets on Azure blob storage

CONFIGURATION
-------------
All settings are in: config/nza_data_sources.yaml
    - directories: Where to save files
    - data_sources: URLs for Transpower and EA
    - datasets: What to download (each can be enabled/disabled)

USAGE
-----
Download all enabled datasets:
    python nza_download_static_data_from_url.py

Download only specific dataset:
    python nza_download_static_data_from_url.py --dataset sites

See what would be downloaded (dry run):
    python nza_download_static_data_from_url.py --dry-run

OUTPUTS
-------
Downloaded files:
    - Saved to directories specified in YAML config
    - Files are always refreshed with latest data
    
Archive copies:
    - Timestamped copies saved to archive directory
    - Format: filename_YYYYMMDD_HHMMSS.extension
    - Useful for tracking changes over time

Log files:
    - Saved to logs/ directory
    - Format: nza_download_YYYYMMDD_HHMMSS.log
    - Contains detailed download and validation information

AUTHOR
------
    Philippe Bruneau

CREATED
-------
    2025-12-17

VERSION
-------
    2.1.0 - Removed --force option; always downloads fresh data
            Added workflow description and user guide
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys
import logging
import argparse
import hashlib
import shutil

import requests
import pandas as pd
import yaml

from pypsa_nza_data.config.project_paths import load_paths_config

PATHS = load_paths_config()

raw_dir = PATHS["raw_data_dir"]
processed_dir = PATHS["processed_data_dir"]



# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> tuple[Path, logging.FileHandler]:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'nza_download_{timestamp}.log'
    
    # Detailed formatter for file
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simple formatter for console
    console_formatter = logging.Formatter('%(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(console_formatter)
    root_logger.addHandler(console)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    return log_file, file_handler


def print_header(title: str, char: str = '=') -> None:
    """Print formatted header."""
    width = 80
    logger.info(char * width)
    logger.info(f"{title:^{width}}")
    logger.info(char * width)


def print_section(title: str) -> None:
    """Print section divider."""
    logger.info("")
    logger.info(f"--- {title} ---")
    logger.info("")


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(config_file: Path) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to YAML configuration
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['directories', 'data_sources', 'datasets']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in config: {section}")
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}")


# ============================================================================
# DIRECTORY MANAGER
# ============================================================================

class DirectoryManager:
    """Manages output directories from config."""
    
    def __init__(self, root_dir: Path, config: Dict):
        """
        Initialize directory manager.
        
        Args:
            root_dir: Root directory (from ROOT_DIR)
            config: Configuration dictionary with 'directories' section
        """
        self.root = Path(root_dir)
        self.dirs = {}
        
        # Create all configured directories
        dir_config = config.get('directories', {})
        
        for name, path in dir_config.items():
            full_path = self.root / path
            full_path.mkdir(parents=True, exist_ok=True)
            self.dirs[name] = full_path
            logger.debug(f"Directory '{name}': {full_path}")
    
    def get(self, name: str) -> Path:
        """
        Get directory by name.
        
        Args:
            name: Directory name from config
            
        Returns:
            Path object
        """
        if name not in self.dirs:
            raise ValueError(f"Unknown directory: {name}")
        return self.dirs[name]


# ============================================================================
# DATA DOWNLOADER
# ============================================================================

class DataDownloader:
    """Downloads data from configured sources."""
    
    def __init__(self, config: Dict):
        """
        Initialize downloader.
        
        Args:
            config: Configuration dictionary with 'data_sources' section
        """
        self.sources = config.get('data_sources', {})
        
        # Setup HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_transpower_dataset(self, dataset_config: Dict, output_path: Path) -> bool:
        """
        Download from Transpower ArcGIS.
        
        Args:
            dataset_config: Dataset configuration
            output_path: Where to save file
            
        Returns:
            True if successful
        """
        dataset_id = dataset_config['dataset_id']
        format_type = dataset_config['format']
        
        transpower = self.sources['transpower']
        base_url = transpower['base_url']
        
        # Construct download URL
        if format_type == 'csv':
            url = f"{base_url}/datasets/{dataset_id}.csv"
        elif format_type == 'geojson':
            url = f"{base_url}/datasets/{dataset_id}.geojson"
        else:
            logger.error(f"Unsupported format: {format_type}")
            return False
        
        logger.info(f"URL: {url}")
        
        try:
            response = self.session.get(url, stream=True, timeout=60)
            
            if response.status_code == 200:
                # Download successful
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            
            elif response.status_code == 404:
                # Try API fallback
                logger.warning("Direct download failed, trying API...")
                return self._download_via_api(dataset_config, output_path)
            
            else:
                logger.error(f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def _download_via_api(self, dataset_config: Dict, output_path: Path) -> bool:
        """Fallback: Download via ArcGIS REST API."""
        dataset_id = dataset_config['dataset_id']
        
        transpower = self.sources['transpower']
        api_url = transpower['api_url']
        
        # Remove _0 suffix for service name
        service_name = dataset_id.replace('_0', '')
        query_url = f"{api_url}/{service_name}/FeatureServer/0/query"
        
        params = {
            'where': '1=1',
            'outFields': '*',
            'f': 'json',
            'returnGeometry': 'true'
        }
        
        try:
            response = self.session.get(query_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Convert JSON response to CSV
            data = response.json()
            if 'features' in data:
                records = [f['attributes'] for f in data['features']]
                df = pd.DataFrame(records)
                df.to_csv(output_path, index=False)
                return True
            else:
                logger.error("No features in API response")
                return False
                
        except Exception as e:
            logger.error(f"API error: {e}")
            return False
    
    def download_ea_dataset(self, dataset_config: Dict, output_path: Path) -> bool:
        """
        Download from Electricity Authority.
        
        Args:
            dataset_config: Dataset configuration
            output_path: Where to save file
            
        Returns:
            True if successful
        """
        url_path = dataset_config['url_path']
        
        ea = self.sources['electricity_authority']
        base_url = ea['base_url']
        
        url = f"{base_url}/{url_path}"
        
        logger.info(f"URL: {url}")
        
        try:
            response = self.session.get(url, stream=True, timeout=60)
            
            if response.status_code == 404:
                logger.warning("File not found (may not be published yet)")
                return False
            
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def download(self, dataset_name: str, dataset_config: Dict, 
                 output_path: Path) -> bool:
        """
        Download a dataset (routes to appropriate method).
        
        Args:
            dataset_name: Name of dataset
            dataset_config: Dataset configuration
            output_path: Where to save file
            
        Returns:
            True if successful
        """
        source = dataset_config['source']
        
        if source == 'transpower':
            return self.download_transpower_dataset(dataset_config, output_path)
        elif source == 'electricity_authority':
            return self.download_ea_dataset(dataset_config, output_path)
        else:
            logger.error(f"Unknown source: {source}")
            return False


# ============================================================================
# DATA VALIDATOR
# ============================================================================

class DataValidator:
    """Validates downloaded files."""
    
    @staticmethod
    def validate(filepath: Path, min_rows: int = 1, file_format: str = 'csv') -> Dict:
        """
        Validate a downloaded file.
        
        Args:
            filepath: Path to file
            min_rows: Minimum expected rows/features
            file_format: File format ('csv', 'geojson', etc.)
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'exists': False,
            'size_bytes': 0,
            'rows': 0,
            'columns': 0,
            'errors': []
        }
        
        # Check existence
        if not filepath.exists():
            result['errors'].append("File does not exist")
            return result
        
        result['exists'] = True
        result['size_bytes'] = filepath.stat().st_size
        
        # Check size
        if result['size_bytes'] == 0:
            result['errors'].append("File is empty")
            return result
        
        # Format-specific validation
        if file_format == 'csv':
            return DataValidator._validate_csv(filepath, min_rows, result)
        elif file_format == 'geojson':
            return DataValidator._validate_geojson(filepath, min_rows, result)
        else:
            # For unknown formats, just check size
            result['valid'] = True
            result['rows'] = 0
            result['columns'] = 0
            return result
    
    @staticmethod
    def _validate_csv(filepath: Path, min_rows: int, result: Dict) -> Dict:
        """Validate CSV file."""
        try:
            df = pd.read_csv(filepath)
            result['rows'] = len(df)
            result['columns'] = len(df.columns)
            
            if result['rows'] < min_rows:
                result['errors'].append(
                    f"Too few rows: {result['rows']} < {min_rows}"
                )
            else:
                result['valid'] = True
                
        except Exception as e:
            result['errors'].append(f"CSV read error: {e}")
        
        return result
    
    @staticmethod
    def _validate_geojson(filepath: Path, min_rows: int, result: Dict) -> Dict:
        """Validate GeoJSON file."""
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # GeoJSON should have 'type' and 'features'
            if data.get('type') != 'FeatureCollection':
                result['errors'].append("Not a valid GeoJSON FeatureCollection")
                return result
            
            features = data.get('features', [])
            result['rows'] = len(features)
            
            # Count properties (columns) from first feature
            if features:
                properties = features[0].get('properties', {})
                result['columns'] = len(properties)
            
            if result['rows'] < min_rows:
                result['errors'].append(
                    f"Too few features: {result['rows']} < {min_rows}"
                )
            else:
                result['valid'] = True
                
        except json.JSONDecodeError as e:
            result['errors'].append(f"JSON parse error: {e}")
        except Exception as e:
            result['errors'].append(f"GeoJSON read error: {e}")
        
        return result
    
    @staticmethod
    def compute_checksum(filepath: Path) -> str:
        """Compute MD5 checksum."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class DownloadOrchestrator:
    """Orchestrates the complete download process."""
    
    def __init__(self, root_dir: Path, config: Dict):
        """
        Initialize orchestrator.
        
        Args:
            root_dir: Root directory
            config: Configuration dictionary
        """
        self.root = Path(root_dir)
        self.config = config
        
        # Initialize components
        self.directories = DirectoryManager(root_dir, config)
        self.downloader = DataDownloader(config)
        self.validator = DataValidator()
        
        # Statistics
        self.stats = {
            'total': 0,
            'downloaded': 0,
            'failed': 0,
            'results': []
        }
    
    def download_dataset(self, name: str, config: Dict) -> bool:
        """
        Download a single dataset.
        
        Args:
            name: Dataset name
            config: Dataset configuration
            
        Returns:
            True if successful
        """
        print_section(f"Dataset: {name}")
        
        # Display info
        logger.info(f"Description: {config.get('description', 'N/A')}")
        logger.info(f"Source:      {config['source']}")
        logger.info(f"Format:      {config['format']}")
        logger.info(f"Filename:    {config['filename']}")
        
        # Get output directory
        dir_name = config['directory']
        output_dir = self.directories.get(dir_name)
        output_path = output_dir / config['filename']
        
        logger.info(f"Save to:     {output_path}")
        
        # Note if replacing existing file
        if output_path.exists():
            logger.info(f"Replacing existing file: {config['filename']}")
        
        # Download
        logger.info("Downloading...")
        success = self.downloader.download(name, config, output_path)
        
        if not success:
            logger.error("✗ Download failed")
            self.stats['failed'] += 1
            return False
        
        # Validate
        min_rows = config.get('validate_min_rows', 1)
        file_format = config.get('format', 'csv')
        validation = self.validator.validate(output_path, min_rows, file_format)
        
        if not validation['valid']:
            logger.error(f"✗ Validation failed: {validation['errors']}")
            self.stats['failed'] += 1
            return False
        
        # Success
        logger.info("✓ Downloaded successfully")
        logger.info(f"  Size:    {validation['size_bytes']:,} bytes")
        
        if file_format == 'geojson':
            logger.info(f"  Features: {validation['rows']:,}")
        else:
            logger.info(f"  Rows:    {validation['rows']:,}")
        
        if validation['columns'] > 0:
            logger.info(f"  Columns: {validation['columns']}")
        
        # Archive if configured
        if self.config.get('validation', {}).get('archive_downloads', True):
            self._archive_file(output_path)
        
        # Record stats
        self.stats['downloaded'] += 1
        self.stats['results'].append({
            'name': name,
            'file': config['filename'],
            'rows': validation['rows'],
            'size': validation['size_bytes']
        })
        
        return True
    
    def _archive_file(self, filepath: Path) -> None:
        """Create timestamped archive copy."""
        try:
            archive_dir = self.directories.get('archive')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create archive filename
            stem = filepath.stem
            suffix = filepath.suffix
            archive_name = f"{stem}_{timestamp}{suffix}"
            archive_path = archive_dir / archive_name
            
            # Copy file
            shutil.copy2(filepath, archive_path)
            logger.info(f"  Archived: {archive_name}")
            
        except Exception as e:
            logger.warning(f"Could not archive file: {e}")
    
    def download_all(self, dataset_filter: Optional[str] = None) -> None:
        """
        Download all enabled datasets.
        
        Args:
            dataset_filter: Optional dataset name to download only that one
        """
        datasets = self.config.get('datasets', {})
        
        for name, dataset_config in datasets.items():
            # Skip if filter specified and doesn't match
            if dataset_filter and name != dataset_filter:
                continue
            
            # Skip if disabled
            if not dataset_config.get('enabled', True):
                logger.info(f"Skipping disabled dataset: {name}")
                continue
            
            self.stats['total'] += 1
            self.download_dataset(name, dataset_config)
    
    def print_summary(self) -> None:
        """Print download summary."""
        print_header("SUMMARY", '=')
        logger.info("")
        
        logger.info(f"Total datasets:         {self.stats['total']}")
        logger.info(f"Successfully downloaded: {self.stats['downloaded']}")
        logger.info(f"Failed:                 {self.stats['failed']}")
        logger.info("")
        
        if self.stats['results']:
            print_section("Downloaded Files")
            
            logger.info(f"{'Dataset':<30} {'Rows':>10} {'Size':>15}")
            logger.info("-" * 60)
            
            for result in self.stats['results']:
                logger.info(
                    f"{result['name']:<30} {result['rows']:>10,} "
                    f"{result['size']:>12,} bytes"
                )
            logger.info("")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Download PyPSA-NZA datasets (config in YAML file)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Download only specific dataset'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without downloading'
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    # Setup paths
    root_path = Path(ROOT_DIR)
    config_file = root_path / "config" / "nza_download_data_from_source.yaml"
    log_dir = root_path / "logs"
    
    # Setup logging - get both log file and handler
    log_file, file_handler = setup_logging(log_dir)
    
    # Print header
    print_header("PYPSA-NZA DATA DOWNLOADER")
    logger.info("")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config:     {config_file}")
    if args.dataset:
        logger.info(f"Filter:     Only '{args.dataset}'")
    if args.dry_run:
        logger.info("Mode:       DRY RUN (no downloads)")
    logger.info("")
    
    try:
        # Load configuration
        print_header("LOADING CONFIGURATION", '=')
        logger.info("")
        
        config = load_config(config_file)
        
        # Show configured directories
        logger.info("Output directories:")
        for name, path in config['directories'].items():
            logger.info(f"  {name:<15} → {path}")
        logger.info("")
        
        # Show data sources
        logger.info("Data sources:")
        for name, source in config['data_sources'].items():
            logger.info(f"  {name}: {source.get('name', 'N/A')}")
        logger.info("")
        
        # Show enabled datasets
        enabled_datasets = [
            name for name, cfg in config['datasets'].items()
            if cfg.get('enabled', True)
        ]
        logger.info(f"Enabled datasets ({len(enabled_datasets)}):")
        for name in enabled_datasets:
            logger.info(f"  - {name}")
        logger.info("")
        
        if args.dry_run:
            logger.info("✓ Dry run complete - no files downloaded")
            return 0
        
        # Download datasets
        print_header("DOWNLOADING DATASETS", '=')
        logger.info("")
        
        orchestrator = DownloadOrchestrator(root_path, config)
        orchestrator.download_all(dataset_filter=args.dataset)
        
        # Print summary
        orchestrator.print_summary()
        
        # Final message
        print_header("COMPLETE", '=')
        logger.info("")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.info(f"End time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed:     {elapsed:.2f} seconds")
        logger.info(f"Log saved:   {log_file}")
        logger.info("")
        
        return 0 if orchestrator.stats['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        # Close log file handler to prevent file lock issues
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.error("\n✗ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        sys.exit(1)
        
        
        
# PyPSA-NZA Data Downloader - User Guide

## Overview

# `nza_download_static_data_from_url.py` is a configuration-driven tool for downloading and validating New Zealand electricity system datasets from multiple sources. The script always downloads fresh data to ensure you have the latest information.

# ## Purpose

# This script automates the download of:
# - **Static network data**: Grid connection points, transmission lines, generation units
# - **Registry data**: Official lists of generation plants and network supply points

# All downloaded files are validated, timestamped, and archived for version control.

# ## Prerequisites

# ### Required Python Packages
# ```
# pyyaml
# requests
# pandas
# ```

# Install with:
# ```bash
# pip install pyyaml requests pandas
# ```

# ### Required Files
# - Configuration file: `config/nza_download_data_from_source.yaml`
# - Root directory module: `nza_root.py` (defines ROOT_DIR)

# ## Configuration

# All settings are controlled through `config/nza_download_data_from_source.yaml`. 

# ### Directory Structure
# ```yaml
# directories:
#   static_data_dir: "data/external/static"
#   archive: "data/external/static/archived"
#   generators: "data/external/generation"
#   load: "data/external/load"
# ```

# Directories are created automatically if they don't exist.

# ### Data Sources
# Two primary sources are pre-configured:

# 1. **Transpower ArcGIS** - Network infrastructure data
# 2. **Electricity Authority (EMI)** - Generation and market data

# ### Adding a New Dataset

# To add a new dataset to download, edit the `datasets` section in the YAML file:
# ```yaml
# datasets:
#   my_new_dataset:
#     enabled: true
#     source: "transpower"  # or "electricity_authority"
#     dataset_id: "abc123_0"  # For Transpower datasets
#     # OR
#     url_path: "path/to/file.csv"  # For EA datasets
#     format: "csv"  # or "geojson"
#     filename: "output_name.csv"
#     directory: "static_data_dir"
#     description: "Brief description"
#     validate_min_rows: 100
# ```

# ### Disabling a Dataset

# Set `enabled: false` in the YAML file:
# ```yaml
# datasets:
#   old_dataset:
#     enabled: false
#     # ... rest of config
# ```

# ## Usage

# ### Basic Usage

# Download all enabled datasets:
# ```bash
# python nza_download_static_data_from_url.py
# ```

# ### Download Specific Dataset

# Download only one dataset:
# ```bash
# python nza_download_static_data_from_url.py --dataset sites
# ```

# ### Dry Run

# Preview what would be downloaded without actually downloading:
# ```bash
# python nza_download_static_data_from_url.py --dry-run
# ```

# ### From Spyder IDE

# In the IPython console:
# ```python
# %run nza_download_static_data_from_url.py --dataset sites
# ```

# ## Output Files

# ### Downloaded Data Files

# Files are saved to directories specified in the YAML configuration:
# - `data/external/static/` - Static network and registry data
# - `data/external/generation/` - Generation data (if configured)
# - `data/external/load/` - Load data (if configured)

# **Note:** Existing files are always replaced with fresh downloads.

# ### Archive Files

# Timestamped copies are saved to the archive directory:
# - Format: `filename_YYYYMMDD_HHMMSS.extension`
# - Example: `sites_20251221_143052.csv`
# - Useful for tracking changes over time

# ### Log Files

# Detailed logs are saved to `logs/`:
# - Format: `nza_download_YYYYMMDD_HHMMSS.log`
# - Contains download URLs, validation results, errors, and timing

# ## Validation

# Each downloaded file is automatically validated:

# ### CSV Files
# - ✓ File exists and is not empty
# - ✓ File can be parsed as CSV
# - ✓ Row count meets minimum threshold
# - ✓ Column count is reported

# ### GeoJSON Files
# - ✓ File exists and is not empty
# - ✓ Valid JSON structure
# - ✓ Valid GeoJSON FeatureCollection
# - ✓ Feature count meets minimum threshold

# Validation failures are logged and the download is marked as failed.

# ## Troubleshooting

# ### Problem: "Configuration file not found"

# **Cause:** Script can't locate the YAML configuration file.

# **Solution:** Ensure you're running from the correct directory, or check that `ROOT_DIR` in `nza_root.py` is set correctly.

# ### Problem: "charmap codec can't decode"

# **Cause:** YAML file contains UTF-8 characters but Python is using wrong encoding.

# **Solution:** This is fixed in v2.1.0 with `encoding='utf-8'` in the file open statement.

# ### Problem: "HTTP 404" errors

# **Cause:** Dataset URL has changed or dataset no longer exists.

# **Solutions:**
# 1. Check if the dataset is still available on the source website
# 2. Update the `dataset_id` or `url_path` in the YAML file
# 3. Try the REST API fallback (automatic for Transpower datasets)

# ### Problem: "Validation failed: Too few rows"

# **Cause:** Downloaded file has fewer rows than expected.

# **Solutions:**
# 1. Check if this is expected (data may have changed)
# 2. Adjust `validate_min_rows` in YAML if the threshold is too high
# 3. Check the log file for download errors

# ### Problem: Downloads work in Spyder but fail in console

# **Cause:** Different working directories or environment variables.

# **Solution:** 
# 1. Ensure you activate the correct conda environment: `conda activate pypsa`
# 2. Check that `nza_root.py` uses absolute paths, not relative paths

# ## Data Sources

# ### Transpower ArcGIS Open Data

# - **Website:** https://data-transpower.opendata.arcgis.com
# - **Datasets:** Substations, transmission lines, network infrastructure
# - **Format:** CSV or GeoJSON
# - **Update Frequency:** Irregular (infrastructure changes)

# ### Electricity Authority (EMI)

# - **Website:** https://www.emi.ea.govt.nz
# - **Storage:** Azure blob storage
# - **Datasets:** Generation fleet, market data, network supply points
# - **Format:** CSV
# - **Update Frequency:** Monthly or as published

# ## Best Practices

# 1. **Run regularly** - Data sources update periodically, download fresh data before analysis
# 2. **Check logs** - Review log files for warnings or errors
# 3. **Monitor archives** - Archive directory shows history of data changes
# 4. **Test with --dry-run** - Preview downloads before committing
# 5. **Version control YAML** - Track configuration changes in git
# 6. **Keep archives** - Don't delete archived files; they're small and useful for debugging

# ## Exit Codes

# - `0` - Success (all downloads completed)
# - `1` - Failure (one or more downloads failed)
# - `130` - Interrupted by user (Ctrl+C)

# ## Support

# For issues or questions:
# 1. Check the log file in `logs/` for detailed error messages
# 2. Verify YAML configuration syntax
# 3. Test with `--dry-run` to check configuration
# 4. Check data source websites for availability

# ## Version History

# - **2.1.0** (2025-12-21)
#   - Removed `--force` option (always downloads fresh data)
#   - Fixed UTF-8 encoding for Windows compatibility
#   - Added workflow description and user guide
  
# - **2.0.1** (2025-12-17)
#   - Fixed indentation bug in DownloadOrchestrator class
  
# - **2.0.0** (2025-12-17)
#   - Initial configuration-driven version
#   - Full YAML configuration support
#   - Automatic validation and archiving        