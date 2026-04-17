# pypsa-nza-data

`pypsa-nza-data` is a Python-based data preparation and processing toolkit for
constructing PyPSA-compatible power system datasets for New Zealand (Aotearoa).

The package provides a transparent, reproducible pipeline for downloading,
cleaning, and transforming authoritative public electricity datasets into
consistent formats suitable for PyPSA-based modelling workflows.

---

## Overview

Electricity system modelling requires substantial data preparation before
analysis can begin. In New Zealand, relevant datasets are publicly available
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

- Downloads raw static and dynamic electricity data from authoritative NZ sources  
- Cleans, aligns, and harmonises datasets  
- Converts energy → power where required  
- Writes all outputs to a **user workspace** (never inside the repo)  
- Provides reproducible demonstrations that:
  - inventory datasets
  - build PyPSA networks
  - export NetCDF and summaries

---

## What this package does NOT do

This package does **not**:

- Perform optimisation  
- Solve dispatch problems  
- Provide scenarios or forecasts  
- Ship raw data  

It prepares data and proves integration.

---

## Data sources

The pipeline uses public datasets including:

- Electricity Authority  
- Transpower New Zealand  

Data are accessed programmatically and not redistributed.

---

## Repository structure

```text
pypsa_nza_data/
├── config/             # Example/default configuration templates
├── demo/               # Demonstration scripts
├── geospatial_utils/   # Coordinate utilities
├── loaders/            # Raw data download from EA and Transpower
├── processors/         # Cleaning and harmonisation into PyPSA components
├── analysis/           # EA time series aggregation, validation and plotting
├── utils/              # Workspace helpers
└── resources/          # Manual mapping templates
```

Package functionality is embodied in modules wihin the loaders, processors, demo, 
analysis and utils direcories. A specific functionality is invoked on the command line by   

```
python -m pypsa_nza_data.<module dir><module name> --<option flags> 
```

For example, raw data is loaded from url by running the loader pipeline : 

```
python -m pypsa_nza_data.loaders.nza_run_loader_pipeline --root dispatch_data
```
where ```nza_run_loader_pipeline``` is the python module inside the ```loaders``` directory 
and ```--root dispatch_data``` is an options flag indicating the name of the workspace 
("```dispatch_data```" in this example).



## Workspace philosophy

The repository contains **code only** and all operations require a workspace (working directory) 
created by the user and specified on the command line via an option flag:

```
--root <workspace>
```

Example:

```text
workspace/
├── config/
├── data/
├── demo/
└── logs/
```

Users modify configs in the workspace, not in the package.

This ensures:

- reproducibility  
- portability  
- archive capability  
- independence from installation location  

---

## System architecture

```
python -m pypsa_nza_data.<module>
                │
                ▼
┌────────────────────────────────────┐
│             Workspace              │
│                                    │
│  config/   user YAML               │
│  data/     raw + processed         │
│  outputs/  networks, figures       │
│  logs/     diagnostics             │
└────────────────────────────────────┘
```

No script writes to the repository.

---

## Installation

Python ≥ 3.9.

---

### Windows (PowerShell)

```powershell
git clone https://github.com/turbopb/pypsa-nza-data.git
cd pypsa-nza-data

conda create -n pypsa-nza python=3.10 -y
conda activate pypsa-nza

python -m pip install -e .
```

---

### Linux / macOS

```bash
git clone https://github.com/turbopb/pypsa-nza-data.git
cd pypsa-nza-data

conda create -n pypsa-nza python=3.10 -y
conda activate pypsa-nza

python -m pip install -e .
```

---

## Creating a workspace

```
mkdir dispatch_data
```

---

# Pipeline overview

```
Download raw data
        ↓
Initialise manual inputs
        ↓
Process & harmonise
        ↓
Create demand/generation
        ↓
Build PyPSA network
        ↓
Export NetCDF + summaries
```

---

## Loaders

STEP 1 : Download raw data from the Electricity Authority and Transpower websites
```

Assuming a working directory called "pypsa_nza_workspace" with full path  
"C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace"" has been created 
and has been populated with a "\config" directory containing appropriate .yaml 
config files, define local environment variables $ROOT, $CFG_STAT, $CFG_DYN:   

    $ROOT = "C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace"
    $CFG_STAT = "$ROOT\config\nza_load_static_data.yaml"
    $CFG_DYN = "$ROOT\config\nza_load_dynamic_data.yaml"

Then execute:
python -m pypsa_nza_data.loaders.nza_run_loader_pipeline --root $ROOT --dynamic-config $CFG_DYN  --static-config $CFG_STAT
  
  
  
  
python -m pypsa_nza_data.loaders.nza_run_loader_pipeline --root <workspace>
```

### Initialise manual inputs
```
python -m pypsa_nza_data.utils.nza_init_workspace --root <workspace>
```

## Processors
```
python -m pypsa_nza_data.processors.nza_run_processing_pipeline --all --root <workspace>
```

---
## Analysis submodule

The `analysis/` submodule provides post-processing and analysis tools for
working with the EA half-hourly time series datasets. It operates on
processed monthly CSV files produced by the `processors/` pipeline and
produces annual aggregated datasets, a site registry, validation outputs,
and plots.

### Modules

| Module | Purpose |
|---|---|
| `nza_poc_aggregator` | Aggregates half-hourly POC-level data to site level with energy conservation checking. Processes a single monthly CSV file. |
| `nza_process_annual` | Orchestrates full-year processing of monthly export, import, or generation dispatch files. Produces annual site-aggregated time series CSVs. |
| `nza_build_site_registry` | Builds the definitive PyPSA bus registry from the union of all sites appearing in the annual export and import datasets. Merges with `nodes.csv` to add coordinates and metadata. |
| `nza_validate_gen_import` | Systematic five-check validation of the discrepancy between annual generation dispatch and grid import totals. Documents the investigation for reproducibility. |
| `nza_plot_annual_overview` | Plots annual overview of generation dispatch, grid import, and grid export. Produces three groups of plots: all-three dataset overview, import/export model inputs, and validation plots. |

### Running the analysis pipeline

**Process a full year of export, import, or generation dispatch data:**

```
python -m pypsa_nza_data.analysis.nza_process_annual \
  --year 2024 \
  --basedir <workspace>/data/processed \
  --outdir  <workspace>/data/processed/annual \
  --flow export

python -m pypsa_nza_data.analysis.nza_process_annual \
  --year 2024 \
  --basedir <workspace>/data/processed \
  --outdir  <workspace>/data/processed/annual \
  --flow import

python -m pypsa_nza_data.analysis.nza_process_annual \
  --year 2024 \
  --basedir <workspace>/data/processed \
  --outdir  <workspace>/data/processed/annual \
  --flow gen
```

**Build the site registry:**

```
python -m pypsa_nza_data.analysis.nza_build_site_registry \
  --year   2024 \
  --anndir <workspace>/data/processed/annual \
  --nodes  <workspace>/data/manual/nodes.csv \
  --outdir <workspace>/data/processed/annual
```

**Produce overview plots:**

```
# All plots (default)
python -m pypsa_nza_data.analysis.nza_plot_annual_overview \
  --year   2024 \
  --anndir <workspace>/data/processed/annual \
  --outdir <workspace>/data/processed/plots

# Import/export model inputs only
python -m pypsa_nza_data.analysis.nza_plot_annual_overview \
  --year 2024 --no-all-three --no-validation

# Validation plots only
python -m pypsa_nza_data.analysis.nza_plot_annual_overview \
  --year 2024 --no-all-three --no-import-export
```

**Run the generation dispatch vs grid import validation:**

```
python -m pypsa_nza_data.analysis.nza_validate_gen_import \
  --anndir <workspace>/data/processed/annual \
  --rawdir <workspace>/data/processed
```

### Annual outputs

Running the full analysis pipeline for a given year produces:

```
<workspace>/data/processed/annual/
├── 2024_export_sites_all.csv       # annual demand time series, site level
├── 2024_import_sites_all.csv       # annual generation time series, site level
├── 2024_gen_sites_all.csv          # annual dispatch time series, site level
├── 2024_site_registry.csv          # definitive bus list with coordinates
├── 2024_export_monthly_summary.csv
├── 2024_import_monthly_summary.csv
└── 2024_gen_monthly_summary.csv

<workspace>/data/processed/plots/
├── 2024_annual_timeseries.png
├── 2024_monthly_bar.png
├── 2024_weekly_detail_2024-07-07.png
├── 2024_annual_timeseries_imp_exp.png
├── 2024_monthly_bar_imp_exp.png
├── 2024_weekly_detail_imp_exp_2024-07-07.png
├── 2024_validation_scatter.png
├── 2024_validation_accounting.png
└── 2024_validation_site_diff.png
```

### Data notes

The analysis module processes three distinct EA datasets which differ in
scope and metering convention:

- **Grid export** (`export`) — energy leaving the National Grid at GXPs.
  Used as demand (load) inputs in the PyPSA model.
- **Grid import** (`import`) — energy entering the National Grid at GIPs.
  Used as generation inputs in the PyPSA model.
- **Generation dispatch** (`gen`) — metered generation at registered
  generator sites, mapped from GIP injections by the EA. Used for
  carrier classification and capacity factor derivation.

A systematic discrepancy of approximately 4.8% exists between the annual
generation dispatch total and the grid import total. Investigation
confirmed this is not attributable to data processing artefacts. Full
details are documented in `nza_validate_gen_import.py` and in the
accompanying thesis appendix.

---

## Demonstrations

---

### Demo 1 — Data inventory

```
python -m pypsa_nza_data.demo.nza_demo_data_inventory --root <workspace>
```

Outputs:
```
<workspace>/demo/inventory/
```

---

### Demo 2 — Build PyPSA networks

Canonical config:

```
<workspace>/config/nza_do_demo.yaml
```

---

### Windows example

```powershell
$ROOT = "C:\path\to\dispatch_data"
$CFG  = "$ROOT\config\nza_do_demo.yaml"

python -m pypsa_nza_data.demo.nza_demo_build_network `
  --root $ROOT `
  --config $CFG `
  --year 2024 --months jan
```

---

### Linux example

```bash
ROOT=/path/to/dispatch_data
CFG=$ROOT/config/nza_do_demo.yaml

python -m pypsa_nza_data.demo.nza_demo_build_network \
  --root $ROOT \
  --config $CFG \
  --year 2024 --months jan
```

---

## PowerShell run script (recommended)

Create:

```
<workspace>/run_demo.ps1
```

```powershell
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$CFG  = Join-Path $ROOT "config\nza_do_demo.yaml"

python -m pypsa_nza_data.demo.nza_demo_build_network `
  --root $ROOT `
  --config $CFG `
  --year 2024 --months jan
```

Run:

```
.\run_demo.ps1
```

---

## Command line interface (CLI)

| Parameter | Meaning |
|----------|---------|
| `--root` | Workspace location |
| `--config` | YAML config (repeatable) |
| `--year` | Year |
| `--months` | jan, feb, … or all |

Examples:

```
--year 2024 --months jan
--year 2024 --months jan,feb,mar OR --year 2024 --months 1,2,3
--year 2024 --months all
```

---

## Outputs

```
<workspace>/demo/
├── 202401.nc
├── summaries/
└── figures/
```
## Methodological transparency

`pypsa-nza-data` is designed to make every transformation from raw public data
to modelling-ready inputs explicit and auditable.

The software follows three principles:

1. **No hidden state**  
   All inputs, intermediate artefacts, and outputs live inside a user-defined
   workspace. The repository itself is never modified during execution.

2. **Configuration-driven behaviour**  
   Data locations, mappings, and processing choices are defined in YAML files.
   Re-running the pipeline with the same configuration and environment will
   reproduce identical results.

3. **Separation of acquisition and interpretation**  
   Raw datasets are downloaded without alteration. Cleaning, alignment,
   aggregation, and assumption-based modifications are performed in distinct,
   documented processing stages.

Manual adjustments (for example, where public information is incomplete or
ambiguous) are externalised as workspace files so they can be inspected,
versioned, and cited.

This structure allows researchers, reviewers, and future users to trace how
each modelling input was derived and to repeat or modify the procedure
without reverse-engineering the code.

## Reproducibility checklist

To reproduce a study, archive:

- workspace directory  
- config files  
- raw data versions or URLs  
- Python environment  
- commands used  

No hidden state is required.

---

## Long-term rerun guarantee

Because paths are workspace-relative and configuration is explicit,
a preserved workspace can be executed years later to regenerate results.

---

## Suggested citation

Please cite the repository and record the version or commit hash used.

---

## License

See `LICENSE`.