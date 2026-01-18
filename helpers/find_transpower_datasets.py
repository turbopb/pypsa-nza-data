# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:10:07 2025

@author: OEM
"""

# find_transpower_datasets.py
"""Helper to find available Transpower datasets."""

import requests

def search_transpower_datasets(keyword: str):
    """Search for datasets on Transpower portal."""
    url = "https://data-transpower.opendata.arcgis.com/api/v3/datasets"
    params = {'q': keyword}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    print(f"\nFound {len(data.get('data', []))} datasets matching '{keyword}':\n")
    
    for dataset in data.get('data', []):
        print(f"Title: {dataset['attributes']['name']}")
        print(f"ID: {dataset['id']}")
        print(f"URL: https://data-transpower.opendata.arcgis.com/datasets/{dataset['id']}")
        print("-" * 80)

if __name__ == '__main__':
    # search_transpower_datasets("transmission")
    # search_transpower_datasets("line")
    search_transpower_datasets("Sites")
    #search_transpower_datasets("line")