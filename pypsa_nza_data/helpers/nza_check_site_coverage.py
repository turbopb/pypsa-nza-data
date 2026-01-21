#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    nza_check_site_coverage.py
    
    Validate site coverage across monthly generation data files against a reference
    sites database for PyPSA network modeling.
    
    DESCRIPTION
    -----------
    Site names in monthly generation data files are compared against a 
    "reference sites database" to identify sites that are NOT in the master 
    reference.
    
    The tool helps ensure data completeness by identifying sites in monthly data 
    that are not found in the reference (potential data quality issues).
    
    INPUT FILES
    -----------
    Reference File:
        - sites.csv: Master list of all valid sites
        - Location: data/processed/static/sites.csv
        - Expected column: 'site' (3-letter site codes)
    
    Monthly Data Files:
        - Format: YYYYMM_g_MW.csv (e.g., 202407_g_MW.csv)
        - Location: data/processed/YYYY/gen/g_MW/
        - First column: DATE (datetime index)
        - Remaining columns: Site codes (3-letter identifiers)
    
    OUTPUT
    ------
    Console and log file containing:
        - Summary statistics for each monthly file
        - List of sites NOT in master reference (data quality issues)
        - Overall validation report across all months
    
    LOGGING
    -------
    - Creates timestamped log files in logs/ directory
    - Dual output: console (simple) + file (detailed)
    - Log filename: nza_check_site_coverage_YYYYMMDD_HHMMSS.log
    
    AUTHOR
    ------
        Philippe Bruneau
    
    CREATED
    -------
        2025-12-17
    
    VERSION
    -------
        1.1.0
"""

from pathlib import Path
from typing import Set, List, Dict, Tuple
from datetime import datetime
import sys
import logging

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Hardcoded paths (will move to YAML later)
ROOT_DIR = Path("C:/Users/Public/Documents/Thesis/analysis/PYPSA-NZA")
REFERENCE_FILE = ROOT_DIR / "data/processed/static/generators.csv"
DATA_DIR = ROOT_DIR / "data/processed/2024/gen/g_MWh"
LOG_DIR = ROOT_DIR / "logs"

# File pattern for monthly generation files
FILE_PATTERN = "*_g_MW.csv"


# ============================================================================
# LOGGING SETUP
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
    log_file = log_dir / f'nza_check_site_coverage_{timestamp}.log'
    
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
# DATA LOADING
# ============================================================================

def load_reference_sites(filepath: Path) -> Set[str]:
    """
    Load the reference list of valid sites from sites.csv.
    
    Args:
        filepath: Path to sites.csv file
        
    Returns:
        Set of site codes (3-letter identifiers)
        
    Raises:
        FileNotFoundError: If reference file doesn't exist
        ValueError: If 'site' column not found in file
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Reference file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        if 'site' not in df.columns:
            raise ValueError(
                f"Column 'site' not found in {filepath.name}. "
                f"Available columns: {', '.join(df.columns)}"
            )
        
        # Get unique site codes, removing any NaN values
        sites = set(df['site'].dropna().astype(str).unique())
        
        logger.info(f"Loaded {len(sites)} reference sites from: {filepath.name}")
        
        return sites
        
    except Exception as e:
        logger.error(f"Error reading reference file: {e}")
        raise


def load_monthly_file_sites(filepath: Path) -> Tuple[Set[str], str]:
    """
    Load site names from a monthly generation data file.
    
    Args:
        filepath: Path to monthly CSV file
        
    Returns:
        Tuple of (set of site codes, first column name)
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        # Read just the header to get column names
        df = pd.read_csv(filepath, nrows=0)
        
        # Get all columns except the first (which should be DATE/datetime)
        columns = df.columns.tolist()
        
        if len(columns) == 0:
            raise ValueError(f"File has no columns: {filepath.name}")
        
        date_column = columns[0]
        site_columns = columns[1:]  # All columns after the first
        
        sites = set(site_columns)
        
        return sites, date_column
        
    except Exception as e:
        logger.error(f"Error reading file {filepath.name}: {e}")
        raise


# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================

def compare_sites(
    reference_sites: Set[str], 
    file_sites: Set[str], 
    filename: str
) -> Dict[str, any]:
    """
    Compare sites between reference and a monthly file.
    
    Args:
        reference_sites: Set of sites from reference file
        file_sites: Set of sites from monthly file
        filename: Name of the monthly file being checked
        
    Returns:
        Dictionary containing comparison results:
            - present_sites: Sites in file that ARE in reference (good)
            - missing_sites: Sites in file that are NOT in reference (issues)
            - num_present: Count of sites present in reference
            - num_missing: Count of sites missing from reference
            - missing_pct: Percentage of file sites NOT in reference
    """
    present_sites = file_sites & reference_sites  # Sites in both
    missing_sites = file_sites - reference_sites  # Sites in file but NOT in reference
    
    # Calculate percentage of sites that are missing from reference
    missing_pct = (len(missing_sites) / len(file_sites) * 100) if file_sites else 0
    
    return {
        'filename': filename,
        'present_sites': sorted(present_sites),
        'missing_sites': sorted(missing_sites),
        'num_file_sites': len(file_sites),
        'num_present': len(present_sites),
        'num_missing': len(missing_sites),
        'missing_pct': missing_pct
    }


def print_comparison_results(results: Dict[str, any]) -> None:
    """
    Print detailed comparison results for a single file.
    
    Args:
        results: Dictionary from compare_sites()
    """
    logger.info(f"File: {results['filename']}")
    logger.info(f"  Total sites in file:     {results['num_file_sites']}")
    logger.info(f"  Sites in master:         {results['num_present']}")
    logger.info(f"  Sites NOT in master:     {results['num_missing']}")
    logger.info(f"  Missing from master:     {results['missing_pct']:.2f}%")
    logger.info("")
    
    if results['num_missing'] > 0:
        logger.info(f"  ⚠ Sites NOT in master reference ({results['num_missing']}):")
        # Print in rows of 10 for readability
        missing = results['missing_sites']
        for i in range(0, len(missing), 10):
            chunk = missing[i:i+10]
            logger.info(f"    {', '.join(chunk)}")
        logger.info("")
    else:
        logger.info(f"  ✓ All sites in file are present in master reference")
        logger.info("")


# ============================================================================
# FILE PROCESSING
# ============================================================================

def find_monthly_files(data_dir: Path, pattern: str = FILE_PATTERN) -> List[Path]:
    """
    Find all monthly generation files in the specified directory.
    
    Args:
        data_dir: Directory containing monthly files
        pattern: Glob pattern for file matching
        
    Returns:
        Sorted list of file paths
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    files = sorted(data_dir.glob(pattern))
    
    if not files:
        logger.warning(f"No files matching pattern '{pattern}' found in {data_dir}")
    
    return files


def process_all_files(
    reference_sites: Set[str],
    data_dir: Path
) -> List[Dict[str, any]]:
    """
    Process all monthly files and compare against reference.
    
    Args:
        reference_sites: Set of valid site codes
        data_dir: Directory containing monthly files
        
    Returns:
        List of comparison results for each file
    """
    # Find all monthly files
    monthly_files = find_monthly_files(data_dir)
    
    if not monthly_files:
        logger.error(f"✗ No monthly files found in: {data_dir}")
        return []
    
    logger.info(f"Found {len(monthly_files)} monthly files to check")
    logger.info("")
    
    all_results = []
    
    # Process each file
    for idx, filepath in enumerate(monthly_files, 1):
        print_section(f"File {idx}/{len(monthly_files)}: {filepath.name}")
        
        try:
            # Load sites from this monthly file
            file_sites, date_column = load_monthly_file_sites(filepath)
            logger.info(f"  Date column: {date_column}")
            logger.info(f"  Site columns: {len(file_sites)}")
            logger.info("")
            
            # Compare against reference
            results = compare_sites(reference_sites, file_sites, filepath.name)
            
            # Print results
            print_comparison_results(results)
            
            # Store for summary
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"✗ Error processing {filepath.name}: {e}")
            continue
    
    return all_results


# ============================================================================
# SUMMARY REPORTING
# ============================================================================

def print_summary(all_results: List[Dict[str, any]]) -> None:
    """
    Print overall summary of site coverage across all files.
    
    Args:
        all_results: List of comparison results from all files
    """
    print_header("SUMMARY REPORT", '=')
    logger.info("")
    
    if not all_results:
        logger.info("No results to summarize.")
        return
    
    # Calculate aggregate statistics
    total_files = len(all_results)
    avg_missing_pct = sum(r['missing_pct'] for r in all_results) / total_files
    
    files_with_issues = sum(1 for r in all_results if r['num_missing'] > 0)
    files_perfect = total_files - files_with_issues
    
    logger.info(f"Total files checked:              {total_files}")
    logger.info(f"Files with all sites in master:   {files_perfect}")
    logger.info(f"Files with sites NOT in master:   {files_with_issues}")
    logger.info(f"Average missing percentage:       {avg_missing_pct:.2f}%")
    logger.info("")
    
    # Find sites consistently NOT in master across all files
    if files_with_issues > 0:
        all_missing = [set(r['missing_sites']) for r in all_results if r['missing_sites']]
        if all_missing:
            consistently_missing = set.intersection(*all_missing)
            if consistently_missing:
                logger.info(f"Sites NOT in master in ALL files ({len(consistently_missing)}):")
                missing_sorted = sorted(consistently_missing)
                for i in range(0, len(missing_sorted), 10):
                    chunk = missing_sorted[i:i+10]
                    logger.info(f"  {', '.join(chunk)}")
                logger.info("")
                logger.info("Recommendation: Add these sites to master reference or investigate data source.")
                logger.info("")
    else:
        logger.info("✓ All sites in generation files are present in master reference!")
        logger.info("")
    
    # Detailed file-by-file summary table
    print_section("File-by-File Summary")
    
    logger.info(f"{'Filename':<25} {'Sites':<8} {'Present':<10} {'Missing':<10} {'% Missing':<12}")
    logger.info("-" * 80)
    
    for result in all_results:
        logger.info(
            f"{result['filename']:<25} "
            f"{result['num_file_sites']:<8} "
            f"{result['num_present']:<10} "
            f"{result['num_missing']:<10} "
            f"{result['missing_pct']:>8.2f}%"
        )
    
    logger.info("")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    
    # Set up logging
    log_file = setup_logging(LOG_DIR)
    
    # Print startup banner
    print_header("NZA SITE COVERAGE CHECKER")
    logger.info("")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Log configuration
    logger.info(f"Root directory:     {ROOT_DIR}")
    logger.info(f"Reference file:     {REFERENCE_FILE}")
    logger.info(f"Data directory:     {DATA_DIR}")
    logger.info(f"Log directory:      {LOG_DIR}")
    logger.info("")
    
    try:
        # Load reference sites
        print_header("LOADING REFERENCE DATA", '=')
        logger.info("")
        reference_sites = load_reference_sites(REFERENCE_FILE)
        logger.info("")
        
        # Sample of reference sites
        sample_sites = sorted(list(reference_sites))[:20]
        logger.info(f"Sample reference sites: {', '.join(sample_sites)}")
        if len(reference_sites) > 20:
            logger.info(f"  ... and {len(reference_sites) - 20} more")
        logger.info("")
        
        # Process all monthly files
        print_header("CHECKING MONTHLY FILES", '=')
        logger.info("")
        all_results = process_all_files(reference_sites, DATA_DIR)
        
        # Print summary
        print_summary(all_results)
        
        # Final message
        print_header("CHECK COMPLETE", '=')
        logger.info("")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info("")
        logger.info(f"Log saved to: {log_file}")
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