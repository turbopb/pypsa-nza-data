# NUST BE IN EVERY SCRIPT!!!! 
from pypsa_nza_data.config.project_paths import load_paths_config

PATHS = load_paths_config()

raw_dir = PATHS["raw_data_dir"]
processed_dir = PATHS["processed_data_dir"]
