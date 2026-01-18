# -*- coding: utf-8 -*-
"""
nza_update_pocs_from_sites.py

Update Missing Coordinates in pocs.csv from sites.csv

This utility identifies POCs (Points of Connection) in pocs.csv that are missing
geographic coordinates (X, Y, lat, long) and automatically fills them in using 
data from sites.csv.

Additionally creates nodes.csv - a comprehensive node list combining sites.csv
with unique POC entries not found in sites.csv, including zone, island, and volts
information.

Handles multiple POC entries for the same site (different voltage levels) - all
entries for the same site get the same coordinates since they share the same
physical location.

WORKFLOW:
---------
1. Create timestamped backups of ORIGINAL pocs.csv and sites.csv
2. Load and clean data (remove Unnamed, poc, pocs columns)
3. Analyze missing coordinates
4. Update missing coordinates from sites.csv
5. Ensure zone, island, volts columns in all files
6. Standardize coordinate precision (X/Y: 0 decimals, lat/long: 6 decimals)
7. Create nodes.csv from sites.csv + unique POCs
8. Save updated files
9. Create missing_coordinates_report.csv only if gaps remain

USAGE:
------
Run this script to automatically update missing coordinates:

    python update_pocs_coordinates.py

The script will:
- Backup ORIGINAL files before any changes
- Update all findable coordinates from sites.csv
- Standardize coordinate precision
- Add zone, island, volts columns where missing
- Create nodes.csv with all unique site entries
- Log all operations to ROOT_DIR/logs/
- Warn about any remaining missing data

INPUT FILES:
-----------
- pocs.csv: Points of Connection with potentially missing coordinates
- sites.csv: Reference file with complete coordinate data

OUTPUT FILES:
------------
- pocs.csv: Updated with coordinates (NEW data)
- sites.csv: Updated with metadata (NEW data)
- nodes.csv: Combined unique sites (NEW data)
- pocs_backup_TIMESTAMP.csv: Original pocs.csv before changes
- sites_backup_TIMESTAMP.csv: Original sites.csv before changes
- missing_coordinates_report.csv: Created only if missing data remains
- update_pocs_coordinates_TIMESTAMP.log: Detailed operation log

COORDINATE PRECISION:
--------------------
- X, Y: 0 decimal places (integers)
- lat, long: 6 decimal places (~0.1m precision)

METADATA COLUMNS:
----------------
- zone: Network zone identifier
- island: Island location (North Island / South Island)
- volts: Voltage level (kV)

Author: Phil
Created: December 2025
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# Import ROOT_DIR from your nza_root module
from nza_root import ROOT_DIR


def setup_logging(log_dir):
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"update_pocs_coordinates_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def clean_unwanted_columns(df):
    """
    Remove unwanted columns from a dataframe.
    Removes: 'Unnamed' columns, 'poc' (singular), and 'pocs' (plural) columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to clean
    
    Returns:
    --------
    pd.DataFrame : Cleaned dataframe
    """
    cols_to_remove = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        # Remove Unnamed columns
        if 'unnamed' in col_lower:
            cols_to_remove.append(col)
        # Remove poc or pocs columns (singular or plural)
        elif col_lower in ['poc', 'pocs']:
            cols_to_remove.append(col)
    
    if cols_to_remove:
        logging.info(f"Removing unwanted columns: {cols_to_remove}")
        df = df.drop(columns=cols_to_remove)
    
    return df


def standardize_coordinate_precision(df):
    """
    Standardize coordinate decimal precision.
    - X, Y: 0 decimal places (integers)
    - lat, long: 6 decimal places
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with coordinate columns
    
    Returns:
    --------
    pd.DataFrame : Dataframe with standardized precision
    """
    for col in df.columns:
        col_lower = col.lower()
        
        # X/Y get 0 decimal places (integers)
        if col_lower == 'x':
            df[col] = df[col].round(0).astype('Int64')  # Int64 allows NaN
        elif col_lower == 'y':
            df[col] = df[col].round(0).astype('Int64')
        # Lat/Long get 6 decimal places
        elif col_lower in ['lat', 'latitude']:
            df[col] = df[col].round(6)
        elif col_lower in ['long', 'lon', 'longitude']:
            df[col] = df[col].round(6)
    
    return df


def ensure_metadata_columns(df, reference_df):
    """
    Ensure zone, island, volts columns exist in dataframe.
    Fill from reference_df where possible.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Target dataframe
    reference_df : pd.DataFrame or None
        Source dataframe with metadata
    
    Returns:
    --------
    pd.DataFrame : Dataframe with metadata columns
    """
    metadata_cols = ['zone', 'island', 'volts']
    
    # Add columns if missing
    for col in metadata_cols:
        if col not in df.columns:
            df[col] = np.nan
            logging.info(f"Added column '{col}' to dataframe")
    
    # Fill from reference if available
    if reference_df is not None and 'site' in df.columns and 'site' in reference_df.columns:
        for col in metadata_cols:
            if col in reference_df.columns:
                # Create lookup from reference
                lookup = reference_df.groupby('site')[col].first().to_dict()
                
                # Fill missing values
                mask = df[col].isna()
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, 'site'].map(lookup)
                    filled_count = mask.sum() - df[col].isna().sum()
                    if filled_count > 0:
                        logging.info(f"Filled {filled_count} missing '{col}' values from reference")
    
    return df


def create_backup(file_path):
    """
    Create timestamped backup of ORIGINAL file before any modifications.
    
    Parameters:
    -----------
    file_path : Path
        Path to file to backup
    
    Returns:
    --------
    Path : Path to backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    
    # Read original and save as backup
    df = pd.read_csv(file_path)
    df.to_csv(backup_path, index=False)
    
    logging.info(f"Created backup of ORIGINAL file: {backup_path.name}")
    return backup_path


def analyze_missing_coordinates(pocs_path, sites_path):
    """
    Analyze which POCs are missing coordinates.
    
    Parameters:
    -----------
    pocs_path : Path
        Path to pocs.csv file
    sites_path : Path
        Path to sites.csv file
    
    Returns:
    --------
    dict : Analysis results
    """
    logging.info(f"Loading pocs.csv from: {pocs_path}")
    pocs_df = pd.read_csv(pocs_path)
    pocs_df = clean_unwanted_columns(pocs_df)
    logging.info(f"Loaded {len(pocs_df)} POC entries")
    
    logging.info(f"Loading sites.csv from: {sites_path}")
    sites_df = pd.read_csv(sites_path)
    sites_df = clean_unwanted_columns(sites_df)
    logging.info(f"Loaded {len(sites_df)} unique sites")
    
    # Check for sites with multiple voltage levels
    if 'site' in pocs_df.columns:
        site_counts = pocs_df['site'].value_counts()
        multi_voltage_sites = site_counts[site_counts > 1]
        
        if len(multi_voltage_sites) > 0:
            logging.info(f"\nFound {len(multi_voltage_sites)} sites with multiple voltage levels:")
            for site, count in multi_voltage_sites.head(10).items():
                site_rows = pocs_df[pocs_df['site'] == site]
                if 'volts' in pocs_df.columns:
                    voltages = sorted(site_rows['volts'].dropna().unique())
                    logging.info(f"  {site}: {count} entries with voltages {voltages}")
                else:
                    logging.info(f"  {site}: {count} entries")
            
            if len(multi_voltage_sites) > 10:
                logging.info(f"  ... and {len(multi_voltage_sites) - 10} more sites")
    
    logging.info("="*80)
    logging.info("ANALYZING MISSING COORDINATES IN pocs.csv")
    logging.info("="*80)
    
    # Identify coordinate columns
    coord_columns = []
    for col in pocs_df.columns:
        col_lower = col.lower()
        if col_lower in ['x', 'y', 'lat', 'long', 'latitude', 'longitude', 'lon']:
            coord_columns.append(col)
    
    logging.info(f"Coordinate columns found: {coord_columns}")
    
    # Check for missing values
    logging.info(f"\nTotal POC entries: {len(pocs_df)}")
    
    missing_analysis = {}
    for col in coord_columns:
        missing_count = pocs_df[col].isna().sum()
        missing_pct = (missing_count / len(pocs_df)) * 100
        missing_analysis[col] = {
            'count': missing_count,
            'percentage': missing_pct
        }
        logging.info(f"  {col}: {missing_count} missing ({missing_pct:.1f}%)")
    
    # Identify POCs with ANY missing coordinate
    mask_missing_any = pd.Series([False] * len(pocs_df))
    for col in coord_columns:
        mask_missing_any |= pocs_df[col].isna()
    
    pocs_missing_any = pocs_df[mask_missing_any]
    logging.info(f"\nPOC entries with ANY missing coordinate: {len(pocs_missing_any)}")
    
    if 'site' in pocs_df.columns:
        missing_sites = pocs_missing_any['site'].unique()
        logging.info(f"Unique site codes with missing coordinates: {len(missing_sites)}")
        sites_list = sorted(missing_sites)[:20]
        if len(missing_sites) > 20:
            sites_list_str = f"{sites_list}... ({len(missing_sites) - 20} more)"
        else:
            sites_list_str = str(sites_list)
        logging.info(f"Sites: {sites_list_str}")
    
    return {
        'pocs_df': pocs_df,
        'sites_df': sites_df,
        'coord_columns': coord_columns,
        'missing_analysis': missing_analysis,
        'has_missing': len(pocs_missing_any) > 0
    }


def update_pocs_from_sites(pocs_df, sites_df):
    """
    Update missing coordinates in pocs using data from sites.
    
    Parameters:
    -----------
    pocs_df : pd.DataFrame
        POCs dataframe
    sites_df : pd.DataFrame
        Sites dataframe
    
    Returns:
    --------
    tuple : (updated_pocs_df, has_remaining_missing)
    """
    logging.info("="*80)
    logging.info("UPDATING MISSING COORDINATES")
    logging.info("="*80)
    
    # Identify coordinate columns
    pocs_coord_cols = {}
    sites_coord_cols = {}
    
    coord_mappings = {
        'x': ['x'],
        'y': ['y'],
        'lat': ['lat', 'latitude'],
        'long': ['long', 'lon', 'longitude']
    }
    
    # Find coordinate columns in pocs
    for col in pocs_df.columns:
        col_lower = col.lower()
        for standard_name, variants in coord_mappings.items():
            if col_lower in variants:
                pocs_coord_cols[standard_name] = col
                break
    
    # Find coordinate columns in sites
    for col in sites_df.columns:
        col_lower = col.lower()
        for standard_name, variants in coord_mappings.items():
            if col_lower in variants:
                sites_coord_cols[standard_name] = col
                break
    
    logging.info(f"\nCoordinate columns in pocs.csv: {pocs_coord_cols}")
    logging.info(f"Coordinate columns in sites.csv: {sites_coord_cols}")
    
    # Create lookup from sites
    sites_lookup = {}
    for idx, row in sites_df.iterrows():
        site_code = row['site']
        sites_lookup[site_code] = {}
        for standard_name, sites_col in sites_coord_cols.items():
            sites_lookup[site_code][standard_name] = row[sites_col]
    
    logging.info(f"\nSites lookup created with {len(sites_lookup)} unique sites")
    
    # Update coordinates
    updates_made = {coord: 0 for coord in pocs_coord_cols.keys()}
    sites_not_found = set()
    
    for idx, row in pocs_df.iterrows():
        site_code = row['site']
        
        if site_code not in sites_lookup:
            sites_not_found.add(site_code)
            continue
        
        for standard_name, pocs_col in pocs_coord_cols.items():
            if pd.isna(row[pocs_col]) and standard_name in sites_lookup[site_code]:
                new_value = sites_lookup[site_code][standard_name]
                if pd.notna(new_value):
                    pocs_df.at[idx, pocs_col] = new_value
                    updates_made[standard_name] += 1
    
    # Report
    logging.info("\nUPDATE RESULTS:")
    total_updates = sum(updates_made.values())
    logging.info(f"Total coordinate values updated: {total_updates}")
    for coord, count in updates_made.items():
        if coord in pocs_coord_cols:
            logging.info(f"  {pocs_coord_cols[coord]}: {count} values updated")
    
    if sites_not_found:
        logging.warning(f"\nSites in pocs NOT found in sites: {len(sites_not_found)}")
        sites_list = sorted(sites_not_found)[:20]
        if len(sites_not_found) > 20:
            logging.warning(f"Sites: {sites_list}... and {len(sites_not_found)-20} more")
        else:
            logging.warning(f"Sites: {sites_list}")
    
    # Check remaining missing
    logging.info("\nREMAINING MISSING COORDINATES:")
    has_remaining_missing = False
    for standard_name, pocs_col in pocs_coord_cols.items():
        still_missing = pocs_df[pocs_col].isna().sum()
        if still_missing > 0:
            has_remaining_missing = True
            logging.warning(f"  {pocs_col}: {still_missing} values STILL MISSING")
        else:
            logging.info(f"  {pocs_col}: All values filled ✓")
    
    return pocs_df, has_remaining_missing


def create_nodes_file(pocs_df, sites_df):
    """
    Create nodes dataframe combining sites with unique POCs not in sites.
    
    Parameters:
    -----------
    pocs_df : pd.DataFrame
        POCs dataframe
    sites_df : pd.DataFrame
        Sites dataframe
    
    Returns:
    --------
    pd.DataFrame : Nodes dataframe
    """
    logging.info("="*80)
    logging.info("CREATING NODES DATAFRAME")
    logging.info("="*80)
    
    # Get unique sites
    pocs_unique_sites = set(pocs_df['site'].unique())
    sites_existing = set(sites_df['site'].unique())
    new_sites = pocs_unique_sites - sites_existing
    
    logging.info(f"\nSites in sites.csv: {len(sites_existing)}")
    logging.info(f"Unique sites in pocs.csv: {len(pocs_unique_sites)}")
    logging.info(f"New sites to add: {len(new_sites)}")
    
    if new_sites:
        logging.info(f"New sites: {sorted(new_sites)}")
    
    # Get new site entries from pocs (first occurrence of each)
    new_sites_df = pocs_df[pocs_df['site'].isin(new_sites)].copy()
    new_sites_df = new_sites_df.drop_duplicates(subset='site', keep='first')
    
    # Get column order from pocs (excluding unwanted columns)
    unwanted = ['poc', 'pocs', 'unnamed']
    pocs_columns = [col for col in pocs_df.columns 
                    if col.lower() not in unwanted 
                    and not any(u in col.lower() for u in unwanted)]
    
    logging.info(f"Using columns: {pocs_columns}")
    
    # Ensure sites has all columns
    for col in pocs_columns:
        if col not in sites_df.columns:
            sites_df[col] = np.nan
    
    # Reorder columns
    sites_df = sites_df[pocs_columns]
    new_sites_df = new_sites_df[pocs_columns]
    
    # Combine
    nodes_df = pd.concat([sites_df, new_sites_df], ignore_index=True)
    
    # Sort by site
    nodes_df = nodes_df.sort_values('site').reset_index(drop=True)
    
    logging.info(f"\nNodes created: {len(nodes_df)} total unique sites")
    
    return nodes_df


def create_missing_report(pocs_df, sites_df, output_dir):
    """
    Create missing coordinates report only if there are missing values.
    
    Parameters:
    -----------
    pocs_df : pd.DataFrame
        POCs dataframe
    sites_df : pd.DataFrame
        Sites dataframe
    output_dir : Path
        Directory to save report
    
    Returns:
    --------
    pd.DataFrame or None : Missing POCs dataframe if any exist
    """
    logging.info("\n" + "="*80)
    logging.info("CHECKING FOR MISSING COORDINATES")
    logging.info("="*80)
    
    # Find coordinate columns
    coord_columns = []
    for col in pocs_df.columns:
        col_lower = col.lower()
        if col_lower in ['x', 'y', 'lat', 'long', 'latitude', 'longitude', 'lon']:
            coord_columns.append(col)
    
    # Find missing
    mask_missing = pd.Series([False] * len(pocs_df))
    for col in coord_columns:
        mask_missing |= pocs_df[col].isna()
    
    missing_pocs = pocs_df[mask_missing].copy()
    
    if len(missing_pocs) == 0:
        logging.info("No missing coordinates - report not needed ✓")
        return None
    
    # Add metadata
    missing_pocs['in_sites_csv'] = missing_pocs['site'].isin(sites_df['site'])
    site_counts = pocs_df['site'].value_counts()
    missing_pocs['total_poc_entries_for_site'] = missing_pocs['site'].map(site_counts)
    
    # Save report
    report_path = output_dir / 'missing_coordinates_report.csv'
    missing_pocs.to_csv(report_path, index=False)
    
    logging.warning(f"\nMissing coordinates report created: {report_path}")
    logging.warning(f"Total POC entries with missing coordinates: {len(missing_pocs)}")
    logging.warning(f"  Found in sites.csv: {missing_pocs['in_sites_csv'].sum()}")
    logging.warning(f"  NOT in sites.csv: {(~missing_pocs['in_sites_csv']).sum()}")
    
    return missing_pocs


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Setup paths
    root_path = Path(ROOT_DIR)
    static_dir = root_path / "data" / "processed" / "static"
    log_dir = root_path / "logs"
    
    # Setup logging
    log_file = setup_logging(log_dir)
    
    # Define file paths
    pocs_path = static_dir / "pocs.csv"
    sites_path = static_dir / "sites.csv"
    nodes_path = static_dir / "nodes.csv"
    
    logging.info(f"ROOT_DIR: {root_path}")
    logging.info(f"POCs file: {pocs_path}")
    logging.info(f"Sites file: {sites_path}")
    logging.info(f"Nodes file: {nodes_path}")
    
    # Check files exist
    if not pocs_path.exists():
        logging.error(f"pocs.csv not found at {pocs_path}")
        print(f"\n❌ ERROR: pocs.csv not found")
        exit(1)
    
    if not sites_path.exists():
        logging.error(f"sites.csv not found at {sites_path}")
        print(f"\n❌ ERROR: sites.csv not found")
        exit(1)
    
    # ========================================================================
    # STEP 0: Create backups of ORIGINAL files BEFORE any modifications
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 0: CREATING BACKUPS OF ORIGINAL FILES")
    logging.info("="*80)
    
    pocs_backup = create_backup(pocs_path)
    sites_backup = create_backup(sites_path)
    
    # ========================================================================
    # STEP 1: Analyze missing coordinates
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 1: ANALYZING MISSING COORDINATES")
    logging.info("="*80)
    
    analysis = analyze_missing_coordinates(pocs_path, sites_path)
    pocs_df = analysis['pocs_df']
    sites_df = analysis['sites_df']
    
    # ========================================================================
    # STEP 2: Update coordinates
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 2: UPDATING COORDINATES")
    logging.info("="*80)
    
    pocs_df, has_missing = update_pocs_from_sites(pocs_df, sites_df)
    
    # ========================================================================
    # STEP 3: Ensure metadata columns
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 3: ENSURING METADATA COLUMNS")
    logging.info("="*80)
    
    pocs_df = ensure_metadata_columns(pocs_df, None)
    sites_df = ensure_metadata_columns(sites_df, pocs_df)
    
    # Check for missing zone/island
    for col in ['zone', 'island']:
        missing_count = pocs_df[col].isna().sum()
        if missing_count > 0:
            logging.warning(f"{missing_count} entries missing '{col}' in pocs.csv")
        
        missing_count_sites = sites_df[col].isna().sum()
        if missing_count_sites > 0:
            logging.warning(f"{missing_count_sites} entries missing '{col}' in sites.csv")
    
    # ========================================================================
    # STEP 4: Standardize precision
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 4: STANDARDIZING COORDINATE PRECISION")
    logging.info("="*80)
    logging.info("X, Y: 0 decimal places (integers)")
    logging.info("lat, long: 6 decimal places")
    
    pocs_df = standardize_coordinate_precision(pocs_df)
    sites_df = standardize_coordinate_precision(sites_df)
    
    # ========================================================================
    # STEP 5: Create nodes
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 5: CREATING NODES.CSV")
    logging.info("="*80)
    
    nodes_df = create_nodes_file(pocs_df, sites_df)
    nodes_df = clean_unwanted_columns(nodes_df)  # Clean again to be sure
    nodes_df = ensure_metadata_columns(nodes_df, pocs_df)
    nodes_df = standardize_coordinate_precision(nodes_df)
    
    # ========================================================================
    # STEP 6: Clean all dataframes one final time before saving
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 6: FINAL CLEANING")
    logging.info("="*80)
    
    pocs_df = clean_unwanted_columns(pocs_df)
    sites_df = clean_unwanted_columns(sites_df)
    nodes_df = clean_unwanted_columns(nodes_df)
    
    logging.info("Final columns check:")
    logging.info(f"  pocs.csv: {list(pocs_df.columns)}")
    logging.info(f"  sites.csv: {list(sites_df.columns)}")
    logging.info(f"  nodes.csv: {list(nodes_df.columns)}")
    
    # ========================================================================
    # STEP 7: Save all files
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 7: SAVING FILES")
    logging.info("="*80)
    
    pocs_df.to_csv(pocs_path, index=False)
    logging.info(f"Saved: {pocs_path}")
    
    sites_df.to_csv(sites_path, index=False)
    logging.info(f"Saved: {sites_path}")
    
    nodes_df.to_csv(nodes_path, index=False)
    logging.info(f"Saved: {nodes_path}")
    
    # ========================================================================
    # STEP 8: Create missing report if needed
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 8: CHECKING FOR REMAINING GAPS")
    logging.info("="*80)
    
    missing_report = create_missing_report(pocs_df, sites_df, static_dir)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("ALL OPERATIONS COMPLETE!")
    logging.info("="*80)
    logging.info("\nFiles updated (NEW data):")
    logging.info(f"  • {pocs_path.name}")
    logging.info(f"  • {sites_path.name}")
    logging.info(f"  • {nodes_path.name} ({len(nodes_df)} unique nodes)")
    logging.info(f"\nBackups (ORIGINAL data):")
    logging.info(f"  • {pocs_backup.name}")
    logging.info(f"  • {sites_backup.name}")
    
    if missing_report is not None:
        logging.info(f"\nMissing data report:")
        logging.info(f"  • missing_coordinates_report.csv ({len(missing_report)} entries)")
    
    logging.info(f"\nLog file:")
    logging.info(f"  • {log_file.name}")
    
    print("\n" + "="*80)
    print("✓ ALL OPERATIONS COMPLETE!")
    print("="*80)
    print(f"\nUpdated files (NEW data):")
    print(f"  • pocs.csv")
    print(f"  • sites.csv")
    print(f"  • nodes.csv ({len(nodes_df)} unique sites)")
    print(f"\nBackups (ORIGINAL data before changes):")
    print(f"  • {pocs_backup.name}")
    print(f"  • {sites_backup.name}")
    print(f"\nCoordinate precision standardized:")
    print(f"  • X, Y: 0 decimals (integers)")
    print(f"  • lat, long: 6 decimals")
    print(f"\nAll files include: zone, island, volts")
    print(f"\nNo 'poc', 'pocs', or 'Unnamed' columns in any file")
    
    if missing_report is not None:
        print(f"\n⚠ Warning: {len(missing_report)} entries still missing coordinates")
        print(f"  Check missing_coordinates_report.csv")
    
    print(f"\nLog: {log_file}")