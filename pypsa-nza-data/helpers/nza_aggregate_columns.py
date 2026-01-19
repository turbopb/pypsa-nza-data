#!/usr/bin/env python
# coding: utf-8

"""
nza_aggregate_columns.py

A data processing pipeline for aggregating grid energy data columns.
data from initial processing of results in columns of energy data at POC level, i.e.

    BPE0331	  COL0661	CYD0331	CYD2201	GLN0332	GOR0331	HLY2201
    12.792	    0	    0.989	0.166	102.968	36.065	18.215
    12.286	    0	    1.519	0.271	52.564	36.968	18.046
    11.466	    0	    6.23	0.319	51.505	37.462	17.825

For pypsa analyes on the total energy for a node (site) is required as voltage
is converted to a universal 220kV to ease computational burden. This script converts 
the above to the following form:
    
    BPE	        COL	   CYD	    GLN	    GOR    	HLY
    12.792	    0	   0.989	0.166	102.968	36.065
    12.286	    0	   1.519	0.271	52.564	36.968	
    11.466	    0	   6.23	    0.319	51.505	37.462


Author: Phillippe Bruneau
Created: Wed May 12 06:48:01 2025
Updated: December 2025
"""

import pandas as pd
import os
import re
import yaml
from datetime import datetime
from calendar import monthrange
from pathlib import Path
from typing import Dict, List, Optional

# Import project root directory
from nza_root import ROOT_DIR


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """Extract datetime information from a filename."""
    date_patterns = [
        (r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", "%Y-%m-%d"),
        (r"(\d{2})[-_]?(\d{2})[-_]?(\d{4})", "%d-%m-%Y"),
        (r"(\d{4})[-_]?(\d{2})", "%Y-%m"),
        (r"(\d{2})(\d{2})(\d{4})", "%m%d%Y"),
        (r"(\d{2})(\d{2})(\d{2})", "%y%m%d"),
    ]

    for pattern, date_format in date_patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                date_str = "-".join(match.groups())
                extracted_date = datetime.strptime(date_str, date_format)
                if "%y" in date_format and extracted_date.year < 2000:
                    extracted_date = extracted_date.replace(year=extracted_date.year + 2000)
                return extracted_date
            except ValueError:
                continue
    return None


def extract_detailed_date_info(date_str: str) -> Dict:
    """Extract comprehensive date information from a date string."""
    date_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y",
        "%y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    ]

    date_obj = None
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue

    if date_obj is None:
        raise ValueError(f"Error: The provided date '{date_str}' does not match any recognized formats.")

    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days_in_month = monthrange(year, month)[1]

    date_info = {
        "Year": year,
        "Month": month,
        "Day": day,
        "Days in Month": days_in_month,
    }

    return date_info


def aggregate_columns(df: pd.DataFrame, identifier: str) -> pd.DataFrame:
    """
    Aggregate DataFrame columns based on specified identifier.
    
    Parameters
    ----------
    identifier : str
        'poc' = first 3 chars, 'volts' = first 6 chars, 'unit' = no agg, 'all' = sum all
    
    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with datetime column and aggregated data.
    """
    date_col = df.columns[0]
    result_df = df[[date_col]].copy()

    match identifier.lower():
        case 'poc':
            groups = {}
            for col in df.columns[1:]:
                key = col[:3]
                groups.setdefault(key, []).append(col)
            agg_df = pd.concat(
                [df[cols].sum(axis=1).rename(key) for key, cols in groups.items()],
                axis=1
            )

        case 'volts':
            groups = {}
            for col in df.columns[1:]:
                key = col[:6]
                groups.setdefault(key, []).append(col)
            agg_df = pd.concat(
                [df[cols].sum(axis=1).rename(key) for key, cols in groups.items()],
                axis=1
            )

        case 'unit':
            agg_df = df.iloc[:, 1:].copy()

        case 'all':
            agg_df = pd.DataFrame({'SUM': df.iloc[:, 1:].sum(axis=1)})

        case _:
            raise ValueError(
                f"Invalid identifier '{identifier}'. "
                "Must be one of: 'poc', 'volts', 'unit', 'all'"
            )

    result_df = pd.concat([result_df, agg_df], axis=1)
    return result_df


def list_filenames_in_directory(directory_path: str,
                               include_full_path: bool = True) -> List[str]:
    """Return a list of filenames in the specified directory."""
    if not os.path.isdir(directory_path):
        raise ValueError(
            f"The specified path does not exist or is not a directory: {directory_path}"
        )

    files = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files.append(full_path if include_full_path else entry)

    return files


def rename_files_with_prefix(filenames: List[str], new_basename: str,
                            new_extension: Optional[str] = None) -> List[str]:
    """Rename files with YYYYMM_ prefix using a new base name."""
    renamed = []
    for file in filenames:
        parts = file.split('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {file}")

        prefix = parts[0]
        original_ext = os.path.splitext(file)[1][1:]
        extension = new_extension if new_extension else original_ext
        new_filename = f"{prefix}_{new_basename}.{extension}"
        renamed.append(new_filename)

    return renamed


def ensure_directory_exists(directory_path: str, verbose: bool = True) -> bool:
    """Check if a directory exists and create it if it doesn't."""
    if not directory_path:
        print("Error: Directory path cannot be empty or None.")
        return False

    directory_path = os.path.normpath(directory_path)

    try:
        if os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                if verbose:
                    print(f"Directory already exists: {directory_path}")
                return True
            else:
                print(f"Error: Path exists but is not a directory: {directory_path}")
                return False

        os.makedirs(directory_path, exist_ok=True)
        if verbose:
            print(f"Directory created: {directory_path}")
        return True

    except PermissionError:
        print(f"Error: Permission denied. Cannot create directory: {directory_path}")
        return False
    except OSError as e:
        print(f"Error: Could not create directory '{directory_path}'")
        print(f"Reason: {e}")
        return False
    except Exception as e:
        print(f"Error: Unexpected error while creating directory '{directory_path}'")
        print(f"Reason: {type(e).__name__}: {e}")
        return False


def process_monthly_aggregation(input_dir: str, output_dir: str,
                                aggregation_type: str = "all",
                                output_basename: str = "aggregated") -> None:
    """Process all monthly files in a directory and create aggregated outputs."""
    print(f"Processing files from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Aggregation type: {aggregation_type}\n")

    if not ensure_directory_exists(output_dir, verbose=True):
        print("Failed to create output directory. Exiting.")
        return

    try:
        input_file_list = list_filenames_in_directory(input_dir, include_full_path=False)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not input_file_list:
        print("No files found in input directory.")
        return

    output_file_list = rename_files_with_prefix(
        input_file_list,
        output_basename,
        new_extension=None
    )

    print(f"Found {len(input_file_list)} files to process\n")

    for index, filename in enumerate(input_file_list):
        print(f"Processing file {index + 1}/{len(input_file_list)}: {filename}")

        filepath = os.path.join(input_dir, filename)

        date_extracted = extract_datetime_from_filename(filepath)
        if date_extracted:
            date_info = extract_detailed_date_info(str(date_extracted))
            print(f"  Date: {date_info['Year']}-{date_info['Month']:02d}")
            print(f"  Days in month: {date_info['Days in Month']}")

        try:
            df = pd.read_csv(filepath)
            aggregated = aggregate_columns(
                df,
                identifier=aggregation_type
            )

            output_path = os.path.join(output_dir, output_file_list[index])
            aggregated.to_csv(output_path, index=False)
            print(f"  Saved: {output_file_list[index]}\n")

        except Exception as e:
            print(f"  Error processing file: {e}\n")
            continue

    print(f"Processing complete! {len(input_file_list)} files processed.")


def process_profile(profile_name: str, profile_config: dict, root_dir: Path) -> None:
    """Process a single aggregation profile."""
    print(f"\n{'='*80}")
    print(f"Processing profile: {profile_name}")
    print(f"Description: {profile_config['description']}")
    print(f"{'='*80}")
    
    input_dir = root_dir / profile_config['input_dir']
    output_dir = root_dir / profile_config['output_dir']
    
    if not input_dir.exists():
        print(f"Warning: Input directory does not exist: {input_dir}")
        print("Skipping this profile.\n")
        return
    
    # Use output_basename from config if provided, otherwise derive from profile name
    output_basename = profile_config.get('output_basename', profile_name.replace('_', '-'))
    
    try:
        process_monthly_aggregation(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            aggregation_type=profile_config['aggregation_type'],
            output_basename=output_basename
        )
    except Exception as e:
        print(f"Error processing profile '{profile_name}': {e}\n")


def main():
    """Main execution function."""
    
    # Convert ROOT_DIR to Path
    root_path = Path(ROOT_DIR)
    
    # Load configuration
    config_path = root_path / 'config' / 'nza_aggregate_config.yaml'
    print(f"Loading configuration from: {config_path}\n")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return
    
    # Get profiles to run (use defaults from config)
    profiles_to_run = config.get('default_profiles', list(config['profiles'].keys()))
    
    print(f"Running default profiles: {', '.join(profiles_to_run)}\n")
    
    # Process each profile
    for profile_name in profiles_to_run:
        if profile_name not in config['profiles']:
            print(f"Warning: Profile '{profile_name}' not found in config. Skipping.")
            continue
            
        profile_config = config['profiles'][profile_name]
        process_profile(profile_name, profile_config, root_path)
    
    print(f"\n{'='*80}")
    print("All profiles processed successfully!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
