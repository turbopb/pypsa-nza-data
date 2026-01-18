# -*- coding: utf-8 -*-
"""
nza_cx_modality_standard.py

Standardize the 'carrier' column in a CSV file using the standard_modality mapping.
Created on Tue Nov 18 17:10:37 2025
"""

import pandas as pd
from pathlib import Path

ROOT_DIR = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/"



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
    # Force lower case and strip whitespace
    modality = modality.lower().strip()
    MODALITY_MAP = {
        # Wind
        "wind": "wind",
        "wnd": "wind",
        "onshore": "wind",
        "onshore wind": "wind",
        "onshore-wind": "wind",
        "onshore_wind": "wind",
        "onshorewind": "wind",
        "onw": "wind",
        "on-shore": "wind",
        "offshore": "offshore_wind",
        "off-shore": "offshore_wind",
        "win": "wind",
        
        # Solar
        "solar": "solar_pv",
        "solar pv": "solar_pv",
        "pv": "solar_pv",
        "photovoltaic": "solar_pv",
        "Photovoltaic panels": "solar_pv",
        
        # Hydro
        "hydro": "hydro",
        "hyd": "hydro",
        "hydro electric": "hydro",
        "hydroelectric": "hydro",
        "hydro-electric": "hydro",
        "hydro_electric": "hydro",
        "hydro dam": "hydro",
        "run-of-river": "hydro",
        "ror": "hydro",
        
        # Pumped Hydro
        "pumped storage": "pumped_storage",
        "pumped hydro": "pumped_storage",
        "psh": "pumped_storage",
        "pshe": "pumped_storage",
        
        # Geothermal
        "geothermal": "geothermal",
        "geo thermal": "geothermal",
        "geo-thermal": "geothermal",
        "geo_thermal": "geothermal",
        "geo": "geothermal",
        
        # Gas
        "ocgt": "gas",
        "open cycle gas turbine": "gas",
        "ccgt": "gas",
        "closed cycle gas turbine": "gas",
        "combined cycle gas turbine": "gas",
        "gas": "gas",
        
        # Coal
        "coal": "coal",
        "lignite": "coal",
        
        # Biomass
        "biomass": "biomass",
        "bio": "biomass",
        
        # Diesel / Oil
        "diesel": "diesel",
        "oil": "diesel",
        "fuel oil": "diesel",
        "fuel-oil": "diesel",
        "fuel_oil": "diesel",
        "dsl": "diesel",
        
        # Battery Storage
        "battery": "battery",
        "battery storage": "battery",
        "bat": "battery",
        "chemical battery energy": "battery",
        "bess": "battery",
        "ele": "battery",
        "electrical energy": "battery",

    }
    return MODALITY_MAP[modality]


def standardize_carrier_column(input_file: str, output_file: str = None, 
                               delimiter: str = ',', encoding: str = 'utf-8') -> str:
    """
    Standardize the 'carrier' column in a CSV file using the standard_modality mapping.
    
    Parameters
    ----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to the output CSV file. If None, overwrites the input file.
    delimiter : str, optional
        CSV delimiter (default: ',')
    encoding : str, optional
        File encoding (default: 'utf-8')
        
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
        # Read the CSV file, treating empty strings as actual empty strings, not NaN
        df = pd.read_csv(input_file, delimiter=delimiter, encoding=encoding, 
                        keep_default_na=False, na_values=[''])
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
    
    # Strip whitespace from carrier column
    df['carrier'] = df['carrier'].astype(str).str.strip()
    
    # Check for empty strings or 'nan' strings after stripping
    empty_mask = (df['carrier'] == '') | (df['carrier'].str.lower() == 'nan')
    empty_count = empty_mask.sum()
    
    if empty_count > 0:
        # Show the problematic rows for debugging
        problem_indices = df[empty_mask].index.tolist()
        raise ValueError(f"The 'carrier' column contains {empty_count} empty or 'nan' values "
                        f"at row indices: {problem_indices}. "
                        f"Please clean the data before standardization.")
    
    # Identify unmapped values before attempting transformation
    unmapped_values = []
    for value in df['carrier'].unique():
        try:
            standard_modality(value)
        except KeyError:
            unmapped_values.append(value)
    
    if unmapped_values:
        raise ValueError(f"The following carrier values are not in the mapping: "
                        f"{unmapped_values}. Please update the MODALITY_MAP or "
                        f"correct these values in your data.")
    
    # Apply the standardization
    try:
        df['carrier'] = df['carrier'].apply(standard_modality)
    except Exception as e:
        raise ValueError(f"Error during standardization: {str(e)}")
    
    # Write the output file
    try:
        df.to_csv(output_file, index=False, encoding=encoding)
    except Exception as e:
        raise IOError(f"Error writing output file {output_file}: {str(e)}")
    
    print(f"Successfully standardized {len(df)} rows")
    return output_file


# Example usage
if __name__ == "__main__":
    # # Overwrite original file
    # try:
    #     result = standardize_carrier_column("energy_data.csv")
    #     print(f"Successfully standardized carrier column in: {result}")
    # except Exception as e:
    #     print(f"Error: {e}")
    
    # Save to a different file
    data_file = "data/processed/static/gen_data.csv"
    data_file_std = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/data/processed/static/gen_data_std.csv"
    try:
        # result = standardize_carrier_column("gen_data.csv", "gen_data_standardized.csv")
        result = standardize_carrier_column(data_file, data_file_std)
        print(f"Successfully created standardized file: {result}")
    except Exception as e:
        print(f"Error: {e}")