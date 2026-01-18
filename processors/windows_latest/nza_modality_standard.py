# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 17:10:37 2025

@author: OEM
"""

import pandas as pd
from pathlib import Path


def standard_modality(modality: str) -> str:
    """
    Map non-standard energy modality descriptor to a consistent set.

    Parameters
    ----------
    modality : str
        Energy modality descriptor

    Returns
    -------
    str
        Consistent energy modality descriptor
    """
    # Force lower case
    modality = modality.lower()
    MODALITY_MAP = {
        # Wind
        "wind": "wind",
        "wnd": "wind",
        "onshore": "wind",
        "on-shore": "wind",
        "offshore": "offshore_wind",
        "off-shore": "offshore_wind",
        "win": "wind",

        # Solar
        "solar": "solar_pv",
        "solar pv": "solar_pv",
        "pv": "solar_pv",
        "photovoltaic": "solar_pv",

        # Hydro
        "hydro": "hydro",
        "hyd": "hydro",
        "hydroelectric": "hydro",
        "hydro dam": "hydro",
        "run-of-river": "hydro",
        "ror": "hydro",

        # Pumped Hydro
        "pumped storage": "pumped_storage",
        "pumped hydro": "pumped_storage",
        "psh": "pumped_storage",

        # Geothermal
        "geothermal": "geothermal",
        "geo-thermal": "geothermal",
        "geo": "geothermal",

        # Gas
        "ocgt": "ocgt",
        "open cycle gas turbine": "ocgt",
        "ccgt": "ccgt",
        "closed cycle gas turbine": "ccgt",
        "combined cycle gas turbine": "ccgt",
        "gas": "ocgt",

        # Coal
        "coal": "coal",
        "lignite": "coal",

        # Biomass
        "biomass": "biomass",
        "bio": "biomass",

        # Diesel / Oil
        "diesel": "diesel",
        "oil": "diesel",
        "dsl": "diesel",

        # Battery Storage
        "battery": "battery",
        "battery storage": "battery",
        "battery-storage": "battery",
        "battery_storage": "battery",
        "bess": "battery",
    }
    return MODALITY_MAP[modality]


def standardize_carrier_column(input_file: str, output_file: str = None) -> str:
    """
    Standardize the 'carrier' column in a CSV file using the standard_modality mapping.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to the output CSV file. If None, overwrites the input file.

    Returns
    -------
    str
        Path to the output file

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist
    ValueError
        If the 'carrier' column doesn't exist or contains unmapped values
    """
    # Set output file to input file if not specified
    if output_file is None:
        output_file = input_file

    # Check if input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Check if input file is a CSV
    if input_path.suffix.lower() not in ['.csv', '.txt']:
        raise ValueError(f"Input file must be a CSV file, got: {input_path.suffix}")

    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {input_file} is empty")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError(f"The file {input_file} contains no data")

    # Check if 'carrier' column exists
    if 'carrier' not in df.columns:
        raise ValueError(f"Column 'carrier' not found in {input_file}. "
                        f"Available columns: {', '.join(df.columns)}")

    # Check for null values in carrier column
    null_count = df['carrier'].isna().sum()
    if null_count > 0:
        raise ValueError(f"The 'carrier' column contains {null_count} null/NaN values. "
                        f"Please clean the data before standardization.")

    # Identify unmapped values before attempting transformation
    unmapped_values = []
    for value in df['carrier'].unique():
        try:
            standard_modality(str(value))
        except KeyError:
            unmapped_values.append(value)

    if unmapped_values:
        raise ValueError(f"The following carrier values are not in the mapping: "
                        f"{unmapped_values}. Please update the MODALITY_MAP or "
                        f"correct these values in your data.")

    # Apply the standardization
    try:
        df['carrier'] = df['carrier'].apply(lambda x: standard_modality(str(x)))
    except Exception as e:
        raise ValueError(f"Error during standardization: {str(e)}")

    # Write the output file
    try:
        df.to_csv(output_file, index=False)
    except Exception as e:
        raise IOError(f"Error writing output file {output_file}: {str(e)}")

    return output_file


# Example usage
if __name__ == "__main__":
    # Overwrite original file


    path = "/home/pbrun/pypsa-nza/data/processed/static/"
    file_names = ["gen_data.csv", "gd.csv"]

    for file in file_names:
        try:
            result = standardize_carrier_column(path + file)
            print(f"Successfully standardized carrier column in: {result}")
        except Exception as e:
            print(f"Error: {e}")

    # # Save to a different file
    # try:
    #     result = standardize_carrier_column("gen_data.csv", "gen_data_standardized.csv")
    #     print(f"Successfully created standardized file: {result}")
    # except Exception as e:
    #     print(f"Error: {e}")