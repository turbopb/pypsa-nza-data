# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 10:49:54 2025

@author: OEM

run_process_data_pipeline.py

Master script that choreographs the PYPSA-NZA data processing suite, ensuring 
correct swquwncing and completeness of processing.  It produces a full set of
cleaned and reformatted data used in the PyPSA capacity expandion models.

In principle, "run_pipeline.py" should only need to be run once after download 
of the raw data from the Transpower and EA web-sites. However, each of the 
functional scripts can be run independently if reqired.


DESCRIPTION
-----------
nza_process_dynamic_data.py
Processes and standardizes raw electricity generation and load data from 
New Zealand's electricity market for PyPSA network modeling.This raw monthly 
data is cleaned and reformatted into columns as standardized time series. 
The processing pipeline handles data aggregation, missing value imputation, 
unit conversion (kWh - MWh) and temporal formatting.

nza_create_load_profile.py

nza_convert_energy_to_power.py

nza_site_analysis.py
Processes monthly CSV files from generation, import, and export data and 
analyzes site codes to determine inconsitencies and missing data. It generates 
comprehensive statistics about site code usage patterns.

nza_aggregate_columns.py





"""


import subprocess

scripts = ["nza_process_dynamic_data.py",
           "nza_create_load_profile.py",
           "nza_convert_energy_to_power.py"
           #"nza_aggregate_columns.py",
           #"nza_site_analysis.py",
           ]

for script in scripts:
    subprocess.run(["python", script], check=True)
    
# scripts = ["nza_process_static_data.py",
#            "nza_update_pocs_from_sites.py", 
#            "nza_process_generator_data.py", 
#            "nza_modality_standard.py"]

# #           "nza_transmission_line_processo.py"]

# for script in scripts:
#     subprocess.run(["python", script], check=True)    