#!/usr/bin/env python3
"""
nza_gen_site_check.py

Utility program to compare site codes between generator data and 
point of connection (POC) reference data for the PYPSA-NZA project.

This program:
1. Identifies any generator sites that are missing from the POC reference database
2. Enriches generator data with geographic coordinates (X, Y, long, lat) from POC data
3. Outputs an enriched generator data file with added location information
"""

import pandas as pd
import sys
from pathlib import Path


def check_generator_sites(gen_file='gen_data_std.csv', poc_file='pocs.csv'):
    """
    Compare site codes between generator data and POC reference data.
    
    Parameters:
    -----------
    gen_file : str
        Path to generator data CSV file (contains 'site' column)
    poc_file : str
        Path to POC reference CSV file (contains 'site' column)
        
    Returns:
    --------
    dict
        Dictionary containing analysis results:
        - 'missing_sites': list of sites in gen_data but not in pocs
        - 'gen_sites': set of all generator sites
        - 'poc_sites': set of all POC sites
        - 'all_found': boolean indicating if all sites were found
    """
    
    try:
        # Read both CSV files
        print(f"Reading generator data from: {gen_file}")
        gen_df = pd.read_csv(gen_file)
        
        print(f"Reading POC reference data from: {poc_file}")
        poc_df = pd.read_csv(poc_file)
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read files - {e}")
        sys.exit(1)
    
    # Verify 'site' column exists in both files
    if 'site' not in gen_df.columns:
        print(f"ERROR: 'site' column not found in {gen_file}")
        print(f"Available columns: {list(gen_df.columns)}")
        sys.exit(1)
        
    if 'site' not in poc_df.columns:
        print(f"ERROR: 'site' column not found in {poc_file}")
        print(f"Available columns: {list(poc_df.columns)}")
        sys.exit(1)
    
    # Extract site codes as sets
    gen_sites = set(gen_df['site'].dropna().unique())
    poc_sites = set(poc_df['site'].dropna().unique())
    
    print(f"\nFound {len(gen_sites)} unique generator sites")
    print(f"Found {len(poc_sites)} unique POC sites")
    
    # Find sites in generator data that are NOT in POC data
    missing_sites = gen_sites - poc_sites
    
    # Prepare results
    results = {
        'missing_sites': sorted(list(missing_sites)),
        'gen_sites': gen_sites,
        'poc_sites': poc_sites,
        'all_found': len(missing_sites) == 0
    }
    
    return results, gen_df, poc_df


def print_report(results, gen_df):
    """
    Print detailed report of the site code comparison.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from check_generator_sites()
    gen_df : pd.DataFrame
        Generator dataframe for detailed information
    """
    print("\n" + "="*70)
    print("SITE CODE VALIDATION REPORT")
    print("="*70)
    
    if results['all_found']:
        print("\n✓ SUCCESS: All generator sites found in POC reference data")
        print(f"  All {len(results['gen_sites'])} generator sites are valid.")
    else:
        print(f"\n✗ WARNING: {len(results['missing_sites'])} generator site(s) NOT found in POC data")
        print("\nMissing sites:")
        print("-" * 70)
        
        for site in results['missing_sites']:
            # Get generator info for this site
            gen_info = gen_df[gen_df['site'] == site]
            
            if not gen_info.empty:
                row = gen_info.iloc[0]
                print(f"\nSite: {site}")
                print(f"  Generator: {row['name']}")
                print(f"  Technology: {row['tech']}")
                print(f"  Capacity: {row['power']} MW")
                print(f"  Voltage: {row['volts']} kV")
            else:
                print(f"\nSite: {site} (no details available)")
        
        print("\n" + "-" * 70)
        print("\nACTION REQUIRED:")
        print("These generator sites need to be added to the POC reference data")
        print("or the generator data needs to be corrected.")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total generator sites:     {len(results['gen_sites'])}")
    print(f"Total POC reference sites: {len(results['poc_sites'])}")
    print(f"Sites found in POC:        {len(results['gen_sites']) - len(results['missing_sites'])}")
    print(f"Sites missing from POC:    {len(results['missing_sites'])}")
    print("="*70)


def enrich_generator_data(gen_df, poc_df, output_file='gen_data_enriched.csv'):
    """
    Enrich generator data with geographic coordinates from POC data.
    
    Adds X, Y, long, lat columns from POC data to generator data based on 
    matching site codes.
    
    Parameters:
    -----------
    gen_df : pd.DataFrame
        Generator dataframe
    poc_df : pd.DataFrame
        POC reference dataframe
    output_file : str
        Path to output enriched CSV file
        
    Returns:
    --------
    tuple
        (enriched_df, all_coords_complete)
        - enriched_df: Enriched generator dataframe with geographic coordinates
        - all_coords_complete: Boolean indicating if all generators have complete coordinates
    """
    
    print("\n" + "="*70)
    print("ENRICHING GENERATOR DATA WITH GEOGRAPHIC COORDINATES")
    print("="*70)
    
    # Select only the columns we need from POC data
    # Keep only unique site entries (some sites have multiple voltage levels)
    poc_coords = poc_df[['site', 'X', 'Y', 'long', 'lat']].copy()
    
    # Remove duplicates - keep first occurrence of each site
    # (multiple voltage levels at same location have same coordinates)
    poc_coords = poc_coords.drop_duplicates(subset='site', keep='first')
    
    print(f"\nExtracting coordinates for {len(poc_coords)} unique POC sites")
    
    # Merge generator data with POC coordinates
    # Using left join to keep all generators, even if coordinates are missing
    enriched_df = gen_df.merge(
        poc_coords, 
        on='site', 
        how='left',
        indicator=True
    )
    
    # Check for generators without coordinates
    missing_coords = enriched_df[enriched_df['_merge'] == 'left_only']
    
    # Also check for NaN values in coordinate columns (in case POC has the site but no coords)
    has_null_coords = enriched_df[
        enriched_df['X'].isna() | 
        enriched_df['Y'].isna() | 
        enriched_df['long'].isna() | 
        enriched_df['lat'].isna()
    ]
    
    if len(missing_coords) > 0 or len(has_null_coords) > 0:
        print(f"\n⚠ WARNING: Generators with missing geographic data detected!")
        print("="*70)
        
        if len(missing_coords) > 0:
            print(f"\nGenerators not found in POC data ({len(missing_coords)}):")
            print("-"*70)
            for _, row in missing_coords.iterrows():
                print(f"  Site: {row['site']}")
                print(f"    Generator: {row['name']}")
                print(f"    Technology: {row['tech']}")
                print(f"    Power: {row['power']} MW")
                print()
        
        if len(has_null_coords) > 0:
            print(f"\nGenerators with incomplete coordinate data ({len(has_null_coords)}):")
            print("-"*70)
            for _, row in has_null_coords.iterrows():
                print(f"  Site: {row['site']}")
                print(f"    Generator: {row['name']}")
                print(f"    Technology: {row['tech']}")
                print(f"    Power: {row['power']} MW")
                missing_fields = []
                if pd.isna(row['X']): missing_fields.append('X')
                if pd.isna(row['Y']): missing_fields.append('Y')
                if pd.isna(row['long']): missing_fields.append('long')
                if pd.isna(row['lat']): missing_fields.append('lat')
                print(f"    Missing fields: {', '.join(missing_fields)}")
                print()
        
        print("="*70)
        print("ACTION REQUIRED:")
        print("Add coordinate data to POC file for the sites listed above")
        print("="*70)
    else:
        print(f"\n✓ All {len(gen_df)} generators successfully enriched with coordinates")
    
    # Remove the merge indicator column
    enriched_df = enriched_df.drop('_merge', axis=1)
    
    # Reorder columns to put coordinates at the end
    original_cols = list(gen_df.columns)
    coord_cols = ['X', 'Y', 'long', 'lat']
    enriched_df = enriched_df[original_cols + coord_cols]
    
    # Try to write to current directory, if that fails try outputs directory
    try:
        enriched_df.to_csv(output_file, index=False)
        output_path = output_file
    except (OSError, PermissionError):
        # If current directory is read-only, try outputs directory
        output_path = f"/mnt/user-data/outputs/{output_file}"
        enriched_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Enriched data written to: {output_path}")
    print(f"  Original columns: {len(original_cols)}")
    print(f"  Added columns: {len(coord_cols)} (X, Y, long, lat)")
    print(f"  Total columns: {len(enriched_df.columns)}")
    
    # Show sample of enriched data
    print("\nSample of enriched data (first 5 rows):")
    print("-" * 70)
    sample = enriched_df.head(5)[['name', 'site', 'power', 'X', 'Y', 'long', 'lat']]
    print(sample.to_string(index=False))
    
    # Return enriched dataframe and status
    all_coords_complete = (len(missing_coords) == 0 and len(has_null_coords) == 0)
    
    return enriched_df, all_coords_complete


def main():
    """Main execution function."""
    
    print("NZA Generator Site Code Validation & Data Enrichment")
    print("=" * 70)
    
    # Check if files exist in current directory
    gen_file = 'gen_data_std.csv'
    poc_file = 'pocs.csv'
    output_file = 'gen_data_enriched.csv'
    
    if not Path(gen_file).exists():
        print(f"ERROR: {gen_file} not found in current directory")
        sys.exit(1)
    
    if not Path(poc_file).exists():
        print(f"ERROR: {poc_file} not found in current directory")
        sys.exit(1)
    
    # Perform site code comparison
    results, gen_df, poc_df = check_generator_sites(gen_file, poc_file)
    
    # Print detailed validation report
    print_report(results, gen_df)
    
    # Enrich generator data with geographic coordinates from POC data
    enriched_df, all_coords_complete = enrich_generator_data(gen_df, poc_df, output_file)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Input files:")
    print(f"  - {gen_file}")
    print(f"  - {poc_file}")
    print(f"Output file:")
    print(f"  - {output_file}")
    print()
    print("Status:")
    print(f"  Site validation: {'✓ PASS' if results['all_found'] else '✗ FAIL'}")
    print(f"  Coordinate data: {'✓ COMPLETE' if all_coords_complete else '⚠ INCOMPLETE'}")
    print("="*70)
    
    # Return exit code based on validation results
    # Exit with error if sites are missing OR coordinates are incomplete
    if results['all_found'] and all_coords_complete:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
