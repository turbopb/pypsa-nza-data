# Node Data Workflow — pypsa-nza-data

This document traces the complete sequence of file creation from primary
downloaded sources through to the inputs used by `pypsa-nza-net` to build
monthly PyPSA networks.

---

## Terminology

The terms **site** and **node** are used interchangeably throughout. A site is
a named location on the NZ transmission grid where metered energy flows are
recorded. In the EA data these are called **Points of Connection (POCs)**.
Multiple POCs can exist at one site (e.g. a site with 220 kV and 110 kV
connections has separate POCs for each voltage level). The processing pipeline
aggregates POC-level data to site level.

---

## Primary Sources (Downloaded)

### 1. Transpower — `Sites.csv`
**What it is:** The complete register of all named sites on the NZ National
Grid, as maintained by Transpower. Contains site codes, names, island
classification (NI/SI), and geographic coordinates.

**How obtained:** Downloaded directly from Transpower's grid data portal.

**How used:** Manually curated into `nodes.csv` (see below). Not used directly
by any script.

---

### 2. Electricity Authority — Monthly POC files
Downloaded from the EA's EMI data portal. Three file types, one file per month:

| File pattern | Content | Direction |
|---|---|---|
| `{YYYYMM}_export_md.csv` | Metered energy leaving the National Grid at all registered POCs | Grid → customers (demand/load) |
| `{YYYYMM}_import_md.csv` | Metered energy entering the National Grid at all registered POCs | Generators → grid |
| `{YYYYMM}_gen_md.csv` | Metered generation at registered named generator sites only | Subset of import |

**EA naming convention (important):**
- **Grid export** = power flowing *out* of the grid to loads (i.e. demand). This
  is what the grid *exports* to customers.
- **Grid import** = power flowing *into* the grid from generators. This is what
  generators *import* into the grid.

This is the opposite of the intuitive convention. In this codebase, EA
"export" = load timeseries input; EA "import" = generation timeseries input.

**Resolution:** 30-minute (i.e., half-hourly intervals). 1,488 timesteps per 31-day
month (48 half-hours × 31 days).

**Storage location:**
```
pypsa_nza_workspace/data/raw/
├── 2024/
│   ├── export/cons_MWh/   {YYYYMM}_export_md.csv  (12 files)
│   ├── import/cons_MWh/   {YYYYMM}_import_md.csv  (12 files)
│   └── gen/               {YYYYMM}_gen_md.csv     (12 files)
└── 2025/
    └── ...
```

---

## Manual Reference Files

These files are created and maintained manually. They are the authoritative
references for network topology and site metadata.

### `nodes.csv`
**Location:** `pypsa_nza_workspace/data/manual/nodes.csv`

**What it is:** A curated register of all NZ grid sites derived from
Transpower's `Site.csv`. Contains:
- `site` — site code (3–4 character EA/Transpower code, e.g. BEN, MAN, ISL)
- `name` — full site name
- `island` — SI or NI
- `lat`, `lon` — WGS84 coordinates
- Additional metadata columns as required

**How used:**
- By `build_site_registry.py` — to add coordinates and island classification
  to the site registry
- By `nza_base_net.py` — to filter load/generation timeseries to SI sites
  only (island column). **Not** used as a topology authority.

**Key rule:** `nodes.csv` is a master reference register only. It is not
authoritative for network topology. The topology authority for the SI network
is `lines_data_si.csv`.

### `lines_data_si.csv`
**Location:** `pypsa_nza_workspace/data/manual/lines_data_si.csv`

**What it is:** Manually compiled transmission line parameters for the SI
network. Each row is one circuit. Contains line endpoints (bus0, bus1),
voltage level, reactance, resistance, and thermal rating.

**How used:** By `nza_base_net.py` as the sole authority for SI network
topology. Only sites appearing as line endpoints in this file become buses
in the PyPSA network.

---

## Processing Pipeline

### Step 1 — POC aggregation (`nza_poc_aggregator.py`)
**Script:** `pypsa-nza-data/pypsa_nza_data/analysis/nza_poc_aggregator.py`

**Input:** One monthly EA POC file (export or import), e.g.
`202407_import_md.csv`

**What it does:**
- Loads the raw POC-level timeseries (131 POCs for July 2024 import)
- Parses site code and voltage level from each POC identifier
- Aggregates all POCs at each site to a single site-level timeseries
  (energy conservation checked; difference must be 0.000%)
- Reports voltage level breakdown and sub-threshold fraction

**Key finding from validation:**
- Grid export (demand): 80.5% of energy exits below 110 kV (sub-threshold)
- Grid import (generation): only 4.7% of energy enters below 110 kV
- This confirms that generation is predominantly at transmission voltage
  (110 kV and 220 kV), while demand is predominantly distributed

**Output:** Site-aggregated timeseries CSV for a single month.

---

### Step 2 — Annual aggregation (`nza_process_annual.py`)
**Script:** `pypsa-nza-data/pypsa_nza_data/analysis/nza_process_annual.py`

**Input:** All 12 monthly POC files for a given year and file type (export
or import)

**What it does:**
- Calls `nza_poc_aggregator.py` for each monthly file
- Concatenates monthly outputs into a single annual timeseries
- Writes annual site-aggregated CSV

**Outputs (written to `pypsa_nza_workspace/data/processed/annual/`):**

| File | Sites | Description |
|---|---|---|
| `{year}_export_sites_all.csv` | 162 | Full-year demand timeseries, all NZ sites |
| `{year}_import_sites_all.csv` | 83–107 | Full-year generation timeseries, all NZ sites |

**Note on site counts:** The import file site count varies slightly between
months (some generators are only active in certain months). The annual file
contains the union of all sites appearing in any month.

---

### Step 3 — Site registry (`nza_build_site_registry.py`)
**Script:** `pypsa-nza-data/pypsa_nza_data/analysis/nza_build_site_registry.py`

**Inputs:**
- `{year}_export_sites_all.csv` — load sites (162)
- `{year}_import_sites_all.csv` — generation sites (83)
- `nodes.csv` — coordinates and island classification

**What it does:**
- Takes the union of all sites appearing in either export or import
- Classifies each site as load-only, generation-only, or both
- Merges with `nodes.csv` to add coordinates and metadata
- Reports annual energy totals per site

**Output:** `{year}_site_registry.csv`

**2024 results:**

| Category | Count |
|---|---|
| Total sites | 165 |
| Load only | 82 |
| Generation only | 3 |
| Both load and generation | 80 |
| Total load | 37,849 GWh |
| Total generation | 39,242 GWh |
| Implied transmission losses | ~3.5% |

---

## Network Build Inputs

The following files are consumed directly by `pypsa-nza-net/nza_base_net.py`
to build the monthly SI PyPSA networks:

| File | Source | Used for |
|---|---|---|
| `{year}_export_sites_all.csv` | Step 2 | Load timeseries at each bus |
| `{year}_import_sites_all.csv` | Step 2 | Generation timeseries at each bus |
| `nodes.csv` | Manual | Island filter (SI vs NI) — not topology |
| `lines_data_si.csv` | Manual | Network topology and line parameters |

**Important:** `{year}_site_registry.csv` from Step 3 is a reference and
validation file. It is not directly consumed by `nza_base_net.py`. The network
topology comes from `lines_data_si.csv`, not from the site registry.

---

## SI Network Participating Sites

After filtering `{year}_import_sites_all.csv` to SI sites (using the `island`
column of `nodes.csv`), the SI network has:

- **29 GIP sites** (generation injection points) in the measured timeseries
- **57 GXP sites** (grid exit points, i.e. load) in the measured timeseries

The 29 SI GIP sites are:
ABY, ARG, ASY, AVI, BEN, BWK, COL, CYD, DOB, GOR, HKK, HWB, ISL, KUM,
MAN, MCH, NMA, NSY, OHA, OHB, OHC, ORO, ROX, SBK, STK, TKA, TKB, WPR, WTK

Note: An earlier count of 19 SI GIP sites was incomplete. The correct count
is 29, confirmed by running `nza_net_pf.py` with site code logging enabled.
The mean generation total (986 MW, Jan 2024) is consistent across both counts,
confirming the additional 10 sites are genuine SI generators.

---

## Summary Diagram

```
DOWNLOADED SOURCES
    Transpower Site.csv          EA monthly POC files
         |                    (export, import, gen)
         | manual curation          |
         v                          v
    nodes.csv              nza_poc_aggregator.py
    lines_data_si.csv         (per month, per type)
         |                          |
         |                          v
         |               nza_process_annual.py
         |                 (concatenate 12 months)
         |                          |
         |              +-----------+-----------+
         |              |                       |
         |   {year}_export_sites_all.csv    {year}_import_sites_all.csv
         |   (162 sites, all NZ, demand)    (83 sites, all NZ, generation)
         |              |                       |
         |              +-------+-------+       |
         |                      |               |
         |              nza_build_site_registry.py
         |                      |
         |              {year}_site_registry.csv
         |              (165 sites, reference only)
         |
         +----------+----------------------------+
                    |
              nza_base_net.py
              (SI filter applied to export/import)
                    |
         Monthly PyPSA SI networks
         models/networks/reference/{year}/si_{month}_{year}/
```
