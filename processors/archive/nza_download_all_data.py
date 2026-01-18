#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nza_download_all_data.py

Simple data downloader for PyPSA-NZA - ALL configuration is in YAML file.

DESCRIPTION
-----------
This script downloads datasets specified in config/nza_data_sources.yaml.
All configuration (URLs, directories, filenames, datasets) is in the YAML file,
making it easy to modify without touching this code.

To change what/where to download:
    1. Edit config/nza_data_sources.yaml
    2. Run this script
    3. That's it!

USAGE
-----
Download all enabled datasets:
    python nza_download_all_data.py

Force re-download (ignore existing files):
    python nza_download_all_data.py --force

Download only specific dataset:
    python nza_download_all_data.py --dataset pocs

See what would be downloaded (dry run):
    python nza_download_all_data.py --dry-run

CONFIGURATION
-------------
All settings are in: config/nza_data_sources.yaml
    - directories: Where to save files
    - data_sources: URLs for Transpower and EA
    - datasets: What to download

AUTHOR
------
    Philippe Bruneau

CREATED
-------
    2025-12-17

VERSION
-------
    2.0.1 - Fixed indentation bug in DownloadOrchestrator class
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

from nza_root import ROOT_DIR


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
    
    return log_file, file_handler  # ← Return BOTH

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
        with open(config_file, 'r') as f:
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
    
    def __init__(self, root_dir: Path, config: Dict, force: bool = False):
        """
        Initialize orchestrator.
        
        Args:
            root_dir: Root directory
            config: Configuration dictionary
            force: Force re-download even if files exist
        """
        self.root = Path(root_dir)
        self.config = config
        self.force = force
        
        # Initialize components
        self.directories = DirectoryManager(root_dir, config)
        self.downloader = DataDownloader(config)
        self.validator = DataValidator()
        
        # Statistics
        self.stats = {
            'total': 0,
            'downloaded': 0,
            'skipped': 0,
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
        
        # Check if already exists
        if output_path.exists() and not self.force:
            logger.info("âœ“ File exists (use --force to re-download)")
            self.stats['skipped'] += 1
            return True
        
        # Download
        logger.info("Downloading...")
        success = self.downloader.download(name, config, output_path)
        
        if not success:
            logger.error("âœ— Download failed")
            self.stats['failed'] += 1
            return False
        
        # Validate
        min_rows = config.get('validate_min_rows', 1)
        file_format = config.get('format', 'csv')
        validation = self.validator.validate(output_path, min_rows, file_format)
        
        if not validation['valid']:
            logger.error(f"âœ— Validation failed: {validation['errors']}")
            self.stats['failed'] += 1
            return False
        
        # Success
        logger.info("âœ“ Downloaded successfully")
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
        logger.info(f"Skipped (existing):     {self.stats['skipped']}")
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
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
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
    log_file, file_handler = setup_logging(log_dir)  # ← Changed
    
    # Print header
    print_header("PYPSA-NZA DATA DOWNLOADER")
    logger.info("")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config:     {config_file}")
    if args.force:
        logger.info("Mode:       Force re-download")
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
        
        orchestrator = DownloadOrchestrator(root_path, config, force=args.force)
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
        # ====================================================================
        # CLOSE LOG FILE HANDLER - This fixes the file lock issue!
        # ====================================================================
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

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.error("\nâœ— Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"âœ— Fatal error: {e}")
        sys.exit(1)
