# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:42:07 2026

@author: OEM
"""

import pandas as pd

nodes = pd.read_csv(r'C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace\data\manual\nodes.csv',
                    sep=None, engine='python')
nodes = nodes.set_index('site')

pocs_to_check = ['TGA', 'KOE', 'GLN', 'WRK', 'TWH', 'NMA', 'WIL', 'GOR', 'BPE', 'LTN']
cols = ['name', 'zone', 'island', 'volts']
available = [c for c in cols if c in nodes.columns]
print(nodes.loc[[s for s in pocs_to_check if s in nodes.index], available])