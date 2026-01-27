#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nza_download_dynamic_data_from_url.py

Time-series data downloader for PyPSA-NZA - Configuration-driven with YAML.

DESCRIPTION
-----------
This script downloads monthly time-series datasets from the Electricity Authority
(EMI) website. It handles multiple data types (generation, grid import/export, 
HVDC flows, reactive power) across specified year and month ranges.

Configuration (URLs, directories, date ranges) is in a YAML file. The download
logic is kept simple and uses direct requests.get() without streaming,
sessions, or heavy validation to enhance speed.

WORKFLOW
--------
The script sequence follows:

1. INITIALIZATION
   - Load configuration from YAML file (nza_download_dynamic_data.yaml)
   - Validate required sections exist (directories, data_sources, datasets)
   - Set up logging to console and timestamped log file
   - Parse command-line arguments (dataset filter, year/month overrides)

2. CONFIGURATION LOADING
   - Parse directory patterns with {year} placeholders
   - Load EA data source base URL
   - Identify enabled datasets with their year/month ranges
   - Parse month specifications (names to numbers, handle 'all')

3. DOWNLOAD PROCESS (for each enabled dataset)
   - For each year in the dataset's year list:
     * Create year-specific output directory (e.g., data/external/2024/gen/)
     * For each month (1-12 or specified subset):
       - Construct monthly filename (YYYYMM_DatasetName.csv)
       - Build complete URL
       - Download file using simple requests.get()
       - Write entire response content to file
       - Log result (success, not available, or error)

4. SIMPLE VALIDATION
   - Check downloaded file size > 100 bytes
   - No CSV parsing (for speed)
   - No row counting (for speed)
   - Files that download successfully are assumed valid

5. ERROR HANDLING
   - 404 errors: Logged as "Not available" (expected for future months)
   - Network errors: Logged with error message, continue to next file
   - Other HTTP errors: Logged with status code
   - All errors are non-fatal - script continues downloading remaining files

6. REPORTING
   - Track statistics (total files, downloaded, not available, failed)
   - Generate summary table by dataset
   - Write detailed log file with all download attempts
   - Report total elapsed time

DESIGN PHILOSOPHY
   - Speed: No streaming, no heavy validation, no archiving
   - Simplicity: Direct file download using requests.get() approach
   - Reliability: Minimal dependencies, simple error handling
   - Configurability: All settings in YAML, easy to modify

DATA TYPES SUPPORTED
   - Generation: Half-hourly generation by station and unit
   - Grid Import: Half-hourly grid imports at connection points
   - Grid Export: Half-hourly grid exports at connection points
   - HVDC Flows: Half-hourly HVDC transfers between islands
   - Reactive Power: Half-hourly reactive power flows

CONFIGURATION
-------------
All settings are in: config/nza_download_dynamic_data.yaml
    - directories: Output paths (supports {year} placeholder)
    - data_sources: Base URL for EA datasets
    - datasets: What to download, year/month ranges, enable/disable

USAGE
-----
Download all enabled datasets for all configured years/months:
    python nza_download_dynamic_data_from_url.py

Download only specific dataset:
    python nza_download_dynamic_data_from_url.py --dataset generation

Download specific years (overrides YAML config):
    python nza_download_dynamic_data_from_url.py --years 2023 2024

Download specific months (overrides YAML config):
    python nza_download_dynamic_data_from_url.py --months jan feb mar

See what would be downloaded (dry run):
    python nza_download_dynamic_data_from_url.py --dry-run

Combine filters:
    python nza_download_dynamic_data_from_url.py --dataset generation --years 2024 --months all

OUTPUTS
-------
Downloaded files:
    - Organized by year and data type
    - Example: data/external/2024/gen/202401_Generation_MD.csv
    - Filename format: YYYYMM_DatasetName.csv
    - Files are always replaced with fresh downloads

Log files:
    - Saved to logs/ directory
    - Format: nza_download_dynamic_YYYYMMDD_HHMMSS.log
    - Contains all download attempts and results
    - Shows which files succeeded, which weren't available, which failed

PERFORMANCE
-----------
Fast by design:
    - No streaming (loads entire file into memory, then writes)
    - No CSV validation (trusts EA data quality)
    - No archiving (no redundant file copies)
    - No sessions (simple request-per-file)
    - Minimal logging overhead

Typical performance:
    - ~10-20 files per minute depending on file size and network
    - Full year (12 months, 5 datasets) = ~60 files in 3-5 minutes

VALIDATION
----------
Minimal validation for speed:
    - File size > 100 bytes (catches obviously empty/failed downloads)
    - No CSV parsing
    - No row counting
    - Assumes EA publishes valid data

Rationale: EA data is reliable, and downstream processing will catch
any data quality issues. The time saved is significant when downloading
hundreds of large files.

ERROR HANDLING
--------------
Non-fatal errors:
    - 404 Not Found: Expected for future/unpublished months, logged as info
    - Network errors: Logged with details, script continues
    - HTTP errors: Logged with status code, script continues
    
Fatal errors:
    - Configuration file missing/invalid
    - YAML syntax errors
    - Invalid month names

The script always attempts to download all requested files even if some fail.

MONTH SPECIFICATIONS
--------------------
Flexible month input:
    - 'all': All 12 months
    - Names: ['january', 'february'] or ['jan', 'feb']
    - Numbers: [1, 2, 3]
    - Mixed: ['jan', 2, 'march']

NOTES
-----
1. EA publishes data ~1 week after month end
2. Future months return 404 (not an error)
3. Script always overwrites existing files (ensures fresh data)
4. Year-specific directories created automatically
5. No internet connection = all downloads fail (expected)

AUTHOR
------
    Philippe Bruneau

CREATED
-------
    2025-03-31

VERSION
-------
    2.0.0 - Refactored for YAML configuration
            Simplified for speed (removed streaming, heavy validation, archiving)
            Uses original fast download approach (direct requests.get())
            Added flexible command-line overrides
            Proper logging and error handling
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys
import logging
import argparse
import calendar

import requests
import yaml

from importlib.resources import files


# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> tuple:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'nza_download_dynamic_{timestamp}.log'
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter('%(message)s')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    
    console = logging.StreamHandler()
    console.setFormatter(console_formatter)
    root_logger.addHandler(console)
    
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
# MONTH PARSING
# ============================================================================

def parse_months_to_numbers(month_list: List[str]) -> List[int]:
    """Convert month names to numbers."""
    normalized = [m.lower().strip() for m in month_list]
    
    if 'all' in normalized:
        return list(range(1, 13))
    
    month_lookup = {}
    for i in range(1, 13):
        full = calendar.month_name[i].lower()
        abbr = calendar.month_abbr[i].lower()
        month_lookup[full] = i
        month_lookup[abbr] = i
    
    result = []
    for item in normalized:
        matched = False
        for key, value in month_lookup.items():
            if key.startswith(item) or item in key:
                result.append(value)
                matched = True
                break
        
        if not matched:
            raise ValueError(f"Unrecognized month name: {item}")
    
    return sorted(set(result))


def parse_months(month_list):
    """
    Backward-compatible wrapper for parse_months_to_numbers().
    Accepts the same inputs as the CLI would supply (list of strings).
    """
    return parse_months_to_numbers(month_list)

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_file: Path) -> Dict:
    """Load configuration from YAML file."""
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['directories', 'data_sources', 'datasets']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in config: {section}")
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}")


def get_enabled_datasets(config: dict) -> list:
    """
    Return a list of dataset names that are enabled in the YAML.
    Expects config['datasets'] to be a dict of dataset_name -> dataset_config.
    """
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        return []

    enabled = []
    for name, ds in datasets.items():
        if isinstance(ds, dict) and ds.get("enabled", False):
            enabled.append(name)
    return enabled
    

# ============================================================================
# SIMPLE DOWNLOADER (USING YOUR ORIGINAL LOGIC)
# ============================================================================

def download_monthly_file(
    base_url: str,
    url_prefix: str,
    filename: str,
    year: int,
    month: int,
    output_dir: Path,
    dry_run: bool = False,
) -> bool:
    """
    Download a single monthly file (YYYYMM_<filename>) from a dataset URL prefix.

    Diagnostics:
      - Logs the fully resolved URL and local output path for every attempt.
      - In dry-run mode, no HTTP request is made; the function returns False and does
        not create any files.
    """

    monthly_filename = f"{year}{month:02d}_{filename}"
    file_url = f"{base_url.rstrip('/')}/{url_prefix.strip('/')}/{monthly_filename}"
    output_path = output_dir / monthly_filename

    # Always show what we are attempting (console + log file)
    logger.info(f"\n    Attempting URL: {file_url}")
    logger.info(f"    Output file:    {output_path}")

    if dry_run:
        logger.info(f"  {year}-{month:02d}: Dry-run (no download)")
        return False

    try:
        response = requests.get(file_url, timeout=60)
        logger.info(f"    HTTP status:    {response.status_code}")
        response.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        # Quick validation - just check file isn't empty
        if output_path.stat().st_size < 100:
            logger.warning(f"  {year}-{month:02d}: Downloaded but file very small")
            return False

        logger.info(
            f"  {year}-{month:02d}: ✓ Downloaded ({output_path.stat().st_size:,} bytes)"
        )
        return True

    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 404:
            logger.info(f"  {year}-{month:02d}: Not available (404)")
        else:
            logger.error(f"  {year}-{month:02d}: HTTP error {status if status is not None else 'unknown'}")
        return False

    except Exception as e:
        logger.error(f"  {year}-{month:02d}: Error - {e}")
        return False



# ============================================================================
# ORCHESTRATOR
# ============================================================================

class DownloadOrchestrator:
    """Orchestrates downloads using simple, fast approach."""
    
    def __init__(self, root_dir: Path, config: Dict):
        self.root = Path(root_dir)
        self.config = config
        
        # Data sources are selected per-dataset (see datasets[*].source)
        # Default source is 'electricity_authority'.
        
        # Statistics
        self.stats = {
            'total_datasets': 0,
            'total_files': 0,
            'downloaded': 0,
            'not_available': 0,
            'failed': 0,
            'results': []
        }
    
    def get_directory(self, pattern: str, year: int) -> Path:
        """Get directory path with year substitution."""
        dir_patterns = self.config.get('directories', {})
        
        if pattern not in dir_patterns:
            raise ValueError(f"Unknown directory pattern: {pattern}")
        
        path_str = dir_patterns[pattern].format(year=year)
        full_path = self.root / path_str
        full_path.mkdir(parents=True, exist_ok=True)
        
        return full_path
    
    def download_dataset(self, name: str, config: Dict,
                        year_filter: Optional[List[int]] = None,
                        month_filter: Optional[List[int]] = None,
                        dry_run: bool = False) -> None:
        """Download all files for a dataset."""
        print_section(f"Dataset: {name}")
        
        logger.info(f"Description: {config.get('description', 'N/A')}")

        # Select data source (default: electricity_authority)
        source_key = config.get('source', 'electricity_authority')
        source_cfg = self.config.get('data_sources', {}).get(source_key, {})
        base_url = source_cfg.get('base_url', '')
        if not base_url:
            logger.error(f"No base_url configured for data source '{source_key}'")
            return
        
        # Get years
        years = year_filter if year_filter else config.get('years', [])
        if not years:
            logger.error("No years specified")
            return
        
        # Get months
        if month_filter:
            months = month_filter
        else:
            month_spec = config.get('months', 'all')
            if isinstance(month_spec, str):
                months = parse_months_to_numbers([month_spec])
            elif isinstance(month_spec, list):
                if all(isinstance(m, int) for m in month_spec):
                    months = month_spec
                else:
                    months = parse_months_to_numbers(month_spec)
            else:
                months = list(range(1, 13))
        
        logger.info(f"Years:  {years}")
        logger.info(f"Months: {len(months)} months")
        logger.info("")
        
        # Download statistics
        ds_downloaded = 0
        ds_not_available = 0
        ds_failed = 0
        
        # Download each year-month combination
        for year in years:
            logger.info(f"Year {year}:")
            
            output_dir = self.get_directory(config['directory_pattern'], year)
            
            for month in months:
                self.stats['total_files'] += 1
                
                success = download_monthly_file(
                    base_url,
                    config['url_prefix'],
                    config['filename'],
                    year,
                    month,
                    output_dir,
                    dry_run=dry_run,
                )
                
                if success:
                    ds_downloaded += 1
                    self.stats['downloaded'] += 1
                else:
                    # Check if file exists to distinguish not_available from failed
                    monthly_filename = f"{year}{month:02d}_{config['filename']}"
                    if not (output_dir / monthly_filename).exists():
                        ds_not_available += 1
                        self.stats['not_available'] += 1
                    else:
                        ds_failed += 1
                        self.stats['failed'] += 1
            
            logger.info("")
        
        # Record results
        self.stats['results'].append({
            'name': name,
            'downloaded': ds_downloaded,
            'not_available': ds_not_available,
            'failed': ds_failed
        })
    
    def download_all(self, dataset_filter: Optional[str] = None,
                    year_filter: Optional[List[int]] = None,
                    month_filter: Optional[List[int]] = None,
                    dry_run: bool = False) -> None:
        """Download all enabled datasets."""
        datasets = self.config.get('datasets', {})
        
        for name, dataset_config in datasets.items():
            if dataset_filter and name != dataset_filter:
                continue
            
            if not dataset_config.get('enabled', True):
                logger.info(f"Skipping disabled dataset: {name}")
                continue
            
            self.stats['total_datasets'] += 1
            self.download_dataset(name, dataset_config, year_filter, month_filter, dry_run=dry_run)
    
    def print_summary(self) -> None:
        """Print download summary."""
        print_header("SUMMARY", '=')
        logger.info("")
        logger.info(f"Total files attempted: {self.stats['total_files']}")
        logger.info(f"Downloaded:            {self.stats['downloaded']}")
        logger.info(f"Not available:         {self.stats['not_available']}")
        logger.info(f"Failed:                {self.stats['failed']}")
        logger.info("")
        
        if self.stats['results']:
            logger.info(f"{'Dataset':<20} {'Downloaded':<12} {'Not Avail':<12} {'Failed':<8}")
            logger.info("-" * 60)
            for result in self.stats['results']:
                logger.info(
                    f"{result['name']:<20} {result['downloaded']:<12} "
                    f"{result['not_available']:<12} {result['failed']:<8}"
                )
            logger.info("")


def get_repo_root() -> Path:
    """
    Resolve repo root robustly for editable installs.
    This file is: pypsa_nza_data/loaders/<script>.py
    parents[2] => pypsa_nza_data/
    parents[3] => repo root
    """
    return Path(__file__).resolve().parents[3]


def get_default_config_path() -> Path:
    """
    Locate the packaged default YAML config.
    Ensure you place the YAML at: pypsa_nza_data/config/nza_download_dynamic_data.yaml
    """
    return files("pypsa_nza_data").joinpath("config/nza_download_dynamic_data.yaml")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description="Download PyPSA-NZA time-series datasets"
    )
    parser.add_argument("--dataset", type=str, help="Download only a specific dataset")
    parser.add_argument("--years", type=int, nargs="+", help="Override years")
    parser.add_argument("--months", type=str, nargs="+", help="Override months")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded (no HTTP requests, no files written)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional override)",        
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Workspace root used to resolve relative paths in YAML. (overrides YAML root settings).",        
    )    

    args = parser.parse_args()
    start_time = datetime.now()

    # Resolve workspace root
    root_path = Path(args.root).expanduser().resolve() if args.root else Path.cwd().resolve()
    
    # Resolve config path
    config_file = Path(args.config).expanduser().resolve() if args.config else Path(get_default_config_path())
    
    # Load config
    config = load_config(config_file)
    
    # Always write logs under the workspace root
    logs_rel = config.get("directories", {}).get("logs", "logs")
    
    log_dir = (root_path / logs_rel).resolve()    
    
    log_file, file_handler = setup_logging(log_dir)

    # Header (console)
    print_header("PYPSA-NZA DYNAMIC DATA DOWNLOADER")
    logger.info("")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config:     {config_file}")
    if args.dataset:
        logger.info(f"Filter:     {args.dataset}")
    if args.years:
        logger.info(f"Years:      {args.years}")
    if args.months:
        logger.info(f"Months:     {args.months}")
    if args.dry_run:
        logger.info("Mode:       DRY-RUN (no downloads)")
    logger.info("")

    try:
        # Validate and show enabled datasets
        print_section("LOADING CONFIGURATION")
        enabled = get_enabled_datasets(config)
        logger.info(f"Enabled datasets: {', '.join(enabled) if enabled else '(none)'}")

        if not enabled:
            logger.error("No datasets are enabled in the configuration.")
            return 2

        # Create orchestrator
        orchestrator = DownloadOrchestrator(root_path, config)

        # Run downloads
        print_section("DOWNLOADING DATASETS")
        orchestrator.download_all(
            dataset_filter=args.dataset,
            year_filter=args.years,
            month_filter=parse_months(args.months) if args.months else None,
            dry_run=args.dry_run,
        )

        # Summary
        orchestrator.print_summary()

        print_section("COMPLETE")
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed:  {elapsed:.2f} seconds")
        logger.info(f"Log:      {log_file}")

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1

    finally:
        try:
            logger.removeHandler(file_handler)
            file_handler.close()
        except Exception:
            pass




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
