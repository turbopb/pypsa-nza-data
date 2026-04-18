# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:00:40 2026

@author: OEM
"""

import pandas as pd

reg = pd.read_csv(
    r'C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace\data\raw\static\20250917_DispatchedGenerationPlant.csv')

sites = ['KOE', 'GLN', 'WRK', 'TWH', 'TGA', 'NMA', 'WIL', 'GOR', 'BPE', 'LTN']

mask = reg['PointOfConnectionCode'].apply(lambda x: str(x)[:3] in sites)
embedded = reg[mask].copy()

# Keep only currently active plant (no decommission date or future date)
embedded['DateDecommissioned'] = pd.to_datetime(
    embedded['DateDecommissioned'], errors='coerce')
active = embedded[
    embedded['DateDecommissioned'].isna() |
    (embedded['DateDecommissioned'] > pd.Timestamp('2024-01-01'))
].copy()

# Drop duplicate units at same POC (same plant listed twice)
active = active.drop_duplicates(
    subset=['PlantName', 'PointOfConnectionCode', 'NameplateMegawatts'])

# Summary by POC
print("=== Active embedded generators by POC ===")
print(f"{'POC':<12} {'Plant':<30} {'Fuel':<15} {'MW':>8} {'Peaker':>8}")
print("-" * 78)
total_mw = 0.0
for _, row in active.sort_values('PointOfConnectionCode').iterrows():
    print(f"{row['PointOfConnectionCode']:<12} "
          f"{row['PlantName']:<30} "
          f"{row['Fuel']:<15} "
          f"{row['NameplateMegawatts']:>8.1f} "
          f"{row['PeakingPlantFlag']:>8}")
    total_mw += row['NameplateMegawatts']

print("-" * 78)
print(f"{'Total installed capacity':>57} {total_mw:>8.1f} MW")

# Summary by fuel type
print("\n=== Summary by fuel type ===")
fuel_summary = active.groupby('Fuel')['NameplateMegawatts'].agg(
    ['sum', 'count']).rename(columns={'sum': 'Total MW', 'count': 'Units'})
fuel_summary['Total MW'] = fuel_summary['Total MW'].round(1)
print(fuel_summary.to_string())

# Cross-check against 2024 dispatch totals
print("\n=== Cross-check: 2024 annual gen dispatch at discrepant POCs ===")
poc_comp = pd.read_csv(
    r'C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace\data\processed\annual\2024_poc_level_comparison.csv',
    index_col=0)
discrepant = poc_comp[poc_comp['diff_GWh'] > 1.0].sort_values(
    'diff_GWh', ascending=False)
print(f"{'POC':<12} {'Gen (GWh)':>12} {'Import (GWh)':>14} {'Diff (GWh)':>12}")
print("-" * 54)
for poc, row in discrepant.iterrows():
    print(f"{poc:<12} {row['gen_GWh']:>12.1f} "
          f"{row['import_GWh']:>14.1f} "
          f"{row['diff_GWh']:>12.1f}")
print("-" * 54)
print(f"{'Total':>12} {discrepant['gen_GWh'].sum():>12.1f} "
      f"{discrepant['import_GWh'].sum():>14.1f} "
      f"{discrepant['diff_GWh'].sum():>12.1f}")