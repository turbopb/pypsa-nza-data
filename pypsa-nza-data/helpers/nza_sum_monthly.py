# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 08:16:45 2026

@author: OEM
"""

import sys
import pandas as pd
import numpy as np


from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths
ROOT_PROJECT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_PROJECT))
from nza_root import ROOT_DIR


# Load your data file (replace 'your_data_file.csv' with your file path)
# If your first column is not being used as the index, you can proceed directly.
# If the first column is meant to be an index, make sure to load it correctly.
# Example: df = pd.read_csv('your_data_file.csv')
#root = "C://Users/Public/Documents/Thesis/analysis/PYPSA-NZA/"
sub_dir = "data/processed/2024/gen/g_MWh/"
file = "202401_gen_md.csv"

file_path = ROOT_DIR + sub_dir + file
df = pd.read_csv(file_path)

# Select all columns except the first one using iloc[:, 1:]
# This selects all rows (:) and columns starting from index 1 (1:)
df_subset = df.iloc[:, 1:]

# Sum all values in the subset DataFrame
# The first .sum() sums up each column individually, resulting in a Series.
# The second .sum() sums the values in that Series to give the grand total.
total_sum = df_subset.sum().sum() * 1e-3

print(f"Original DataFrame:\n{df}")
print(f"\nDataFrame subset (excluding first column):\n{df_subset}")
print(f"\nTotal sum of all values in the subset: {total_sum}")