# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 10:49:54 2025

@author: OEM
"""

# run_loader_pipeline.py
import subprocess
scripts = ["nza_load_dynamic_data_from_url.py", "nza_load_static_data_from_url.py"]
for script in scripts:
    subprocess.run(["python", script], check=True)
    
    
    
    