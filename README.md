# pypsa-nza-data

`pypsa-nza-data` is a Python-based data preparation and processing toolkit for
constructing PyPSA-compatible power system datasets for New Zealand (Aotearoa).

The package provides a transparent, reproducible pipeline for downloading,
cleaning, and transforming authoritative public electricity datasets into
consistent formats suitable for PyPSA-based modelling workflows.

---

## Overview

Electricity system modelling requires substantial data preparation before
analysis can begin. In the case of New Zealand, relevant datasets are publicly available
but heterogeneous, distributed across multiple sources, and not directly
compatible with modelling frameworks such as PyPSA.

`pypsa-nza-data` addresses this gap by providing a structured, end-to-end data
pipeline that:

- downloads raw static and time-series electricity data,
- cleans, validates, and aligns the data temporally and spatially,
- produces PyPSA-compatible inputs, and
- demonstrates successful PyPSA network construction.

The package is intended for academic and policy research where transparency,
reproducibility, and traceability of data preparation are essential.

---

## What this package does

This package:

- Downloads raw static and dynamic electricity data from authoritative New
  Zealand sources (e.g. Electricity Authority, Transpower)
- Cleans, aligns, and aggregates datasets into consistent formats
- Performs robust temporal alignment and unit conversion (energy → power)
- Saves all outputs to a user-defined external workspace (external to the repo)
- Provides reproducible demonstration scripts that:
  - create inventories of data outputs
  - construct PyPSA networks and export NetCDF files

---

## What this package does NOT do

This package does **not**:

- Perform power system optimisation
- Solve dispatch or capacity expansion problems
- Provide policy scenarios or forecasts
- Ship raw data inside the repository

It is a data-preparation and demonstration package, not a modelling study.

---

## Data sources

The pipeline draws on publicly available datasets from New Zealand electricity
sector institutions, including:

- Electricity Authority (generation and demand data)
- Transpower New Zealand (transmission network information)

Source datasets are accessed programmatically and are not redistributed in this
repository.

---

## Repository structure

The package is organised by function:

```text
pypsa_nza_data/
├── loaders/        # Download raw static and time-series data
├── processors/     # Clean, transform, and harmonise datasets
├── demo/           # Reproducible demonstration scripts
├── utils/          # Workspace initialisation and helpers
├── config/         # YAML configuration files
└── resources/      # Manual mapping templates
````

All scripts are executed via `python -m pypsa_nza_data.<module>`.

---

## Configuration and portability

All file paths are defined relative to a user-specified `--root` workspace
directory. No data or logs are written inside the repository.

The software does not rely on `PYTHONPATH`, IDE state, or hard-coded absolute
paths. All file system interactions use standard Python path handling.

The package has been verified to run identically on both Windows and Linux
systems.

---

## Manual transmission geometry and generator metadata

Certain transmission line geometries and generator metadata from public sources 
are either not available, not defined consistently or in some way not appropraite 
for direct inclusion. These are are therefore modified externally and imported 
manually.

These manual CSV files are:

* initialised automatically into the workspace during setup,
* treated as explicit, version-controlled inputs, and
* documented as assumptions rather than hidden transformations.

This approach prioritises transparency and reproducibility over automation.

---

## Installation and platform compatibility

The package requires Python ≥ 3.9 and is developed and tested using Conda-based
environments.

```
git clone https://github.com/turbopb/pypsa-nza-data.git
cd pypsa-nza-data

conda create -n pypsa-nza-joss python=3.10 -y
conda activate pypsa-nza-joss

python -m pip install -e .
```

The software has been verified on both Windows and Linux platforms.

---

## Quickstart (end-to-end)

Choose or create an empty directory to act as a workspace:

```
mkdir ~/pypsa_nza_workspace
```

Run the full data preparation pipeline:

```
python -m pypsa_nza_data.loaders.nza_run_loader_pipeline \
  --root ~/pypsa_nza_workspace

python -m pypsa_nza_data.utils.nza_init_workspace \
  --root ~/pypsa_nza_workspace

python -m pypsa_nza_data.processors.nza_run_processing_pipeline \
  --all \
  --root ~/pypsa_nza_workspace
```

All raw data, processed outputs, and logs are written under the workspace
directory.

---

## Demonstrations

Two demonstration scripts are provided to verify correct installation and
integration with PyPSA.

### Demo 1: Data inventory and provenance

This demo inventories all raw, processed, and derived datasets produced by the
pipeline.

```
python -m pypsa_nza_data.demo.nza_demo_data_inventory \
  --root ~/pypsa_nza_workspace \
  --config pypsa_nza_data/config/paths.yaml \
  --config pypsa_nza_data/config/nza_load_static_data.yaml \
  --config pypsa_nza_data/config/nza_load_dynamic_data.yaml \
  --config pypsa_nza_data/config/nza_process_static_data.yaml \
  --config pypsa_nza_data/config/nza_process_dynamic_data.yaml
```

Outputs are written to:

```
<workspace>/demo/inventory/
```

---

### Demo 2: PyPSA network construction

This demo constructs a PyPSA network for a full year of data and exports monthly
NetCDF files along with a network topology plot.

```bash
python -m pypsa_nza_data.demo.nza_demo_build_network \
  --root ~/pypsa_nza_workspace \
  --config pypsa_nza_data/config/paths.yaml \
  --config pypsa_nza_data/config/nza_process_static_data.yaml \
  --config pypsa_nza_data/config/nza_process_dynamic_data.yaml \
  --config pypsa_nza_data/config/nza_create_load_profile.yaml \
  --year 2024 \
  --months all
```

Outputs include:

* Monthly PyPSA NetCDF network files
* A single network topology map
* Annual summary statistics

---

## Reproducibility and design principles

* All scripts are executed via `python -m`
* All outputs are written exclusively to a user-defined workspace
* No generated data are committed to the repository
* All configuration is explicit and file-based
* The full pipeline is deterministic and cross-platform

---

## License

This project is released under an open-source licence. See `LICENSE` for details.

```

