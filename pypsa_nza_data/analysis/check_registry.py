# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:50:37 2026

@author: OEM
"""

import pandas as pd

reg = pd.read_csv(
    r'C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace\data\raw\static\20250917_DispatchedGenerationPlant.csv')

# Check all discrepant POC sites
sites = ['KOE', 'GLN', 'WRK', 'TWH', 'TGA', 'NMA', 'WIL', 'GOR', 'BPE', 'LTN']
mask = reg['POC'].apply(lambda x: str(x)[:3] in sites)
cols = [c for c in ['POC', 'Plant_Name', 'Fuel_Type', 'Capacity_MW', 
                     'Commission_Date'] if c in reg.columns]
print(reg[mask][cols].to_string())