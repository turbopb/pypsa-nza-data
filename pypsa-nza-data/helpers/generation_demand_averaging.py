# -*- coding: utf-8 -*-
"""
generation_demand_averaging.py

Calculate monthly average generation and demand from 30-minute interval data.

This script reads monthly CSV files for generation and demand data,
calculates the mean for each month, and produces a summary dataframe.

Created on Tue Jan  6 17:40:23 2026

"""

import pandas as pd
import os

# Define base paths
gen_dir = r"C:\Users\Public\Documents\Thesis\analysis\PYPSA-NZA\data\processed\2024\gen\g_MWh_sum"
demand_dir = r"C:\Users\Public\Documents\Thesis\analysis\PYPSA-NZA\data\processed\2024\demand\d_MWh_sum"

# Initialize list to store results
monthly_sums = []

# Process each month
for month in range(1, 13):
    # Construct file paths
    month_str = f"{month:02d}"  # Format as 01, 02, etc.
    gen_file = os.path.join(gen_dir, f"2024{month_str}_g_MWh_sum.csv")
    demand_file = os.path.join(demand_dir, f"2024{month_str}_d_MWh_sum.csv")
    #print(demand_file)
    
    # Check if files exist
    if not os.path.exists(gen_file):
        print(f"Warning: Generation file not found for month {month}")
        continue
    if not os.path.exists(demand_file):
        print(f"Warning: Demand file not found for month {month}")
        continue
    
    # Read CSV files
    gen_data = pd.read_csv(gen_file)
    demand_data = pd.read_csv(demand_file)
    
    # Calculate averages (mean of SUM column)
    g_sum = gen_data['SUM'].sum()
    d_sum = demand_data['SUM'].sum()
    
    # Store results
    monthly_sums.append({
        'month': month,
        'g_sum': g_sum,
        'd_sum': d_sum
    })
    
    print(f"Month {month}: g_avg = {g_sum:.2f} MWh, d_sum = {d_sum:.2f} MWh")

# Create final dataframe
df_monthly = pd.DataFrame(monthly_sums)

# Display results
print("\nMonthly Averages Summary:")
print(df_monthly.to_string(index=False))

# Optional: Save to CSV
output_file = r"C:\Users\Public\Documents\Thesis\analysis\PYPSA-NZA\data\processed\2024\monthly_sums_2024.csv"
df_monthly.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")


# This script will:
# 1. Loop through all 12 months
# 2. Read each generation and demand CSV file
# 3. Calculate the mean of the 'SUM' column for each
# 4. Create a dataframe with columns: `month`, `g_avg`, `d_avg`
# 5. Print progress and results
# 6. Save the summary to a CSV file

# The output will look like:
# ```
#  month     g_avg     d_avg
#      1  1234.567  1456.789
#      2  1345.678  1567.890
#      3  1456.789  1678.901
#    ...       ...       ...