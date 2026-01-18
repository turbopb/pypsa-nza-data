# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 19:07:21 2025

@author: OEM
"""

#!/usr/bin/env python
# coding: utf-8

"""
nza_site_analysis.py

The annual, 30 minute interval electrical energy data (published by Transpower 
and the Electricity Authonty) effectively defines the network energy flow, its 
generation and consumption.  These data are encapsulated in the "generation", 
"import" and "export" files for each month of a given year. 

In the context of creating a network for PyPSA analysis, these data are critical
because each node in the data files represents an energy transaction that actually 
took place (i.e. measured) and by implication, confirms the existence, participation 
and role of that node.  

Whilst these assertions may seem pedantic or stating the obvious, there are several 
reasons why a clear and accurate representation of the network, its connnectivity
are crucial:
    
1. The numerical solution of a PyPSA network is highly sensitive to the fidelity 
of the topology. Errors such as a disconnected node or the mis-specification of 
node typw such as generator or load can lead to program crashes or non-sensical 
results.

2. Consequently, care and measures must be taken that the data are correct. However, 
since electrical networks are complex even the best efforts to ensure fidelity 
can lead to errors which are not easy to identify.

3. The published data set defining network is complex and information pertaining 
is distributed amongst many documents and digital files. Whilst the data is 
gemerally consistent and reliable, errors do occur (as openly acknowledged by 
Transpower and the Electricity Authonty).

The primary issue is that PyPSA requirws an accurate definition of the network 
nodes their connectivity. This script checks this data by scanning all monthly 
CSV files from the generation, import, and export directories and generates 
informative statistics. 

An important use case is to determine which generator sites appeared during a 
given year.  In essence, new generators will not be present in the data sets for 
January, but will be be added to the generation mix in later months.  In other 
words, the generator fleet at the beginning of the year is not the same at the 
end of the year. A "superset" list of generators is then created which is used 
later to define the network as starting with ALL the generators for that year. 
The ones that only come on line during the year are initially deactivated (zero 
generation, and remain invisible but present) until activated at the correct 
time (positive generation). 


WORKFLOW:
=========
1. Load configuration specifying which years to analyze
2. For each year and data type (gen, import, export):
   - Scan monthly CSV files in the corresponding aggregated directory
   - Extract column headers (site codes) from each file
   - Build a set of unique site codes for that type
3. Create a combined superset of all site codes across all types
4. Generate comprehensive statistics including:
   - Count of sites per type
   - Sites unique to each type
   - Sites common to multiple types
   - Sites present in all types
5. Output results to console and save to report file

OUTPUT:
=======
- Console display of all statistics
- Text report file: site_stats_YYYY.txt

Author: Phillippe Bruneau
Created: December 2025
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Set, List
from datetime import datetime

# Import project root directory
from nza_root import ROOT_DIR


def load_config(config_path: Path) -> dict:
    """
    Load YAML configuration file.
    
    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_monthly_files(directory: Path) -> List[Path]:
    """
    Get list of monthly CSV files in a directory.
    
    Parameters
    ----------
    directory : Path
        Directory containing monthly CSV files.
    
    Returns
    -------
    List[Path]
        Sorted list of CSV file paths.
    """
    if not directory.exists():
        return []
    
    csv_files = sorted(directory.glob("*.csv"))
    return csv_files


def extract_site_codes(csv_file: Path) -> Set[str]:
    """
    Extract site codes from CSV file column headers.
    
    Parameters
    ----------
    csv_file : Path
        Path to CSV file.
    
    Returns
    -------
    Set[str]
        Set of site codes (column names excluding 'DATE').
    """
    try:
        # Read just the header
        df = pd.read_csv(csv_file, nrows=0)
        
        # Get all columns except DATE
        columns = [col for col in df.columns if col.upper() != 'DATE']
        
        return set(columns)
    
    except Exception as e:
        print(f"  Warning: Could not read {csv_file.name}: {e}")
        return set()


def collect_site_codes_for_type(root_dir: Path, year: int, 
                                data_type: str) -> Set[str]:
    """
    Collect all unique site codes for a specific data type and year.
    
    Parameters
    ----------
    root_dir : Path
        Project root directory.
    year : int
        Year to analyze.
    data_type : str
        Type of data ('gen', 'import', or 'export').
    
    Returns
    -------
    Set[str]
        Set of unique site codes across all months for this type.
    """
    # Construct directory path
    agg_dir_name = f"{data_type}_MWh"
    directory = root_dir / "data" / "processed" / str(year) / data_type / agg_dir_name
    
    print(f"\nProcessing {data_type.upper()} data from: {directory}")
    
    if not directory.exists():
        print(f"  Warning: Directory does not exist")
        return set()
    
    # Get all monthly files
    monthly_files = get_monthly_files(directory)
    
    if not monthly_files:
        print(f"  Warning: No CSV files found")
        return set()
    
    print(f"  Found {len(monthly_files)} monthly files")
    
    # Collect site codes from all months
    all_sites = set()
    
    for csv_file in monthly_files:
        sites = extract_site_codes(csv_file)
        all_sites.update(sites)
        print(f"    {csv_file.name}: {len(sites)} sites")
    
    print(f"  Total unique sites for {data_type}: {len(all_sites)}")
    
    return all_sites


def generate_statistics(site_data: Dict[str, Set[str]]) -> str:
    """
    Generate comprehensive statistics about site code distribution.
    
    Parameters
    ----------
    site_data : Dict[str, Set[str]]
        Dictionary mapping data type to set of site codes.
    
    Returns
    -------
    str
        Formatted statistics report.
    """
    gen_sites = site_data.get('gen', set())
    import_sites = site_data.get('import', set())
    export_sites = site_data.get('export', set())
    
    # Combined superset
    all_sites = gen_sites | import_sites | export_sites
    
    # Intersection sets
    gen_import = gen_sites & import_sites
    gen_export = gen_sites & export_sites
    import_export = import_sites & export_sites
    all_three = gen_sites & import_sites & export_sites
    
    # Unique to each type
    gen_only = gen_sites - import_sites - export_sites
    import_only = import_sites - gen_sites - export_sites
    export_only = export_sites - gen_sites - import_sites
    
    # Sites in exactly two types
    gen_import_only = (gen_sites & import_sites) - export_sites
    gen_export_only = (gen_sites & export_sites) - import_sites
    import_export_only = (import_sites & export_sites) - gen_sites
    
    # Build report
    report = []
    report.append("=" * 80)
    report.append("NZA SITE CODE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary counts
    report.append("-" * 80)
    report.append("SUMMARY STATISTICS")
    report.append("-" * 80)
    report.append(f"Total unique sites across all types: {len(all_sites)}")
    report.append(f"  Generation sites:                  {len(gen_sites)}")
    report.append(f"  Import sites:                      {len(import_sites)}")
    report.append(f"  Export sites:                      {len(export_sites)}")
    report.append("")
    
    # Sites in all three types
    report.append("-" * 80)
    report.append("SITES IN ALL THREE TYPES")
    report.append("-" * 80)
    report.append(f"Count: {len(all_three)}")
    if all_three:
        report.append(f"Sites: {', '.join(sorted(all_three))}")
    else:
        report.append("Sites: None")
    report.append("")
    
    # Sites unique to each type
    report.append("-" * 80)
    report.append("SITES UNIQUE TO EACH TYPE")
    report.append("-" * 80)
    report.append(f"\nGeneration only ({len(gen_only)} sites):")
    if gen_only:
        report.append(f"  {', '.join(sorted(gen_only))}")
    else:
        report.append("  None")
    
    report.append(f"\nImport only ({len(import_only)} sites):")
    if import_only:
        report.append(f"  {', '.join(sorted(import_only))}")
    else:
        report.append("  None")
    
    report.append(f"\nExport only ({len(export_only)} sites):")
    if export_only:
        report.append(f"  {', '.join(sorted(export_only))}")
    else:
        report.append("  None")
    report.append("")
    
    # Sites in exactly two types
    report.append("-" * 80)
    report.append("SITES IN EXACTLY TWO TYPES")
    report.append("-" * 80)
    report.append(f"\nGeneration AND Import (but not Export) - {len(gen_import_only)} sites:")
    if gen_import_only:
        report.append(f"  {', '.join(sorted(gen_import_only))}")
    else:
        report.append("  None")
    
    report.append(f"\nGeneration AND Export (but not Import) - {len(gen_export_only)} sites:")
    if gen_export_only:
        report.append(f"  {', '.join(sorted(gen_export_only))}")
    else:
        report.append("  None")
    
    report.append(f"\nImport AND Export (but not Generation) - {len(import_export_only)} sites:")
    if import_export_only:
        report.append(f"  {', '.join(sorted(import_export_only))}")
    else:
        report.append("  None")
    report.append("")
    
    # Pairwise intersections (including all three)
    report.append("-" * 80)
    report.append("PAIRWISE INTERSECTIONS (including sites in all three)")
    report.append("-" * 80)
    report.append(f"Generation AND Import:  {len(gen_import)} sites")
    report.append(f"Generation AND Export:  {len(gen_export)} sites")
    report.append(f"Import AND Export:      {len(import_export)} sites")
    report.append("")
    
    # Complete site lists for each type
    report.append("-" * 80)
    report.append("COMPLETE SITE LISTS BY TYPE")
    report.append("-" * 80)
    report.append(f"\nGeneration ({len(gen_sites)} sites):")
    if gen_sites:
        report.append(f"  {', '.join(sorted(gen_sites))}")
    else:
        report.append("  None")
    
    report.append(f"\nImport ({len(import_sites)} sites):")
    if import_sites:
        report.append(f"  {', '.join(sorted(import_sites))}")
    else:
        report.append("  None")
    
    report.append(f"\nExport ({len(export_sites)} sites):")
    if export_sites:
        report.append(f"  {', '.join(sorted(export_sites))}")
    else:
        report.append("  None")
    report.append("")
    
    # All unique sites (superset)
    report.append("-" * 80)
    report.append("ALL UNIQUE SITES (SUPERSET)")
    report.append("-" * 80)
    report.append(f"Total: {len(all_sites)} sites")
    if all_sites:
        report.append(f"{', '.join(sorted(all_sites))}")
    else:
        report.append("None")
    report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def analyze_year(root_dir: Path, year: int) -> None:
    """
    Analyze site codes for a specific year.
    
    Parameters
    ----------
    root_dir : Path
        Project root directory.
    year : int
        Year to analyze.
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING YEAR: {year}")
    print(f"{'='*80}")
    
    # Collect site codes for each data type
    site_data = {}
    
    for data_type in ['gen', 'import', 'export']:
        sites = collect_site_codes_for_type(root_dir, year, data_type)
        site_data[data_type] = sites
    
    # Generate statistics
    print(f"\n{'='*80}")
    print("GENERATING STATISTICS")
    print(f"{'='*80}")
    
    report = generate_statistics(site_data)
    
    # Print to console
    print(report)
    
    # Save to file
    output_dir = root_dir / "data" / "processed" / str(year)
    output_file = output_dir / f"site_stats_{year}.txt"
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving report: {e}")


def main():
    """Main execution function."""
    
    # Convert ROOT_DIR to Path
    root_path = Path(ROOT_DIR)
    
    # Load configuration
    config_path = root_path / 'config' / 'nza_site_analysis_config.yaml'
    print(f"Loading configuration from: {config_path}\n")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        print("Please create the configuration file.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return
    
    # Get years to analyze
    years = config.get('years', [2024])
    print(config)
    
    if not years:
        print("No years specified in configuration. Defaulting to 2024.")
        years = [2024]
    
    print(f"Years to analyze: {years}")
    
    # Analyze each year
    for year in years:
        try:
            analyze_year(root_path, year)
        except Exception as e:
            print(f"\nError analyzing year {year}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()