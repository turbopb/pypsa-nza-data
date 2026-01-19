# pypsa-nza-data

## Overview
`pypsa-nza-data` is an open-source Python software package for the preparation, 
processing, and validation of New Zealand electricity system data for use with 
the PyPSA (Python for Power System Analysis) modelling framework.

The software ingests heterogeneous public datasets from sources such as the 
Electricity Authority of New Zealand, Transpower, and international technology 
cost databases, and transforms them into a consistent, reproducible, and PyPSA
compatible set of input files. It is designed to support transparent power flow 
and capacity expansion studies by separating data engineering from modelling and 
scenario design.

The package focuses exclusively on **data preparation**. It does not define 
optimisation scenarios or policy assumptions, and it can be used independently of 
any larger modelling framework.

---

## Scope and design principles

The design of `pypsa-nza-data` is guided by the following principles:

* **Reproducibility**: all data transformations are scripted and documented
* **Transparency**: assumptions and limitations are made explicit
* **Modularity**: data ingestion, processing, and validation are separated
* **PyPSA compatibility**: outputs conform to PyPSA’s component CSV schema
* **Pragmatism**: where full automation is not feasible, controlled manual steps 
are documented

---

## Repository structure

The repository is organised into three main functional layers:

```
pypsa_nza_data/
+-- loaders/        # Downloading and ingesting raw static and dynamic datasets
+-- processors/     # Cleaning, transforming, and standardising data
+-- helpers/        # Reusable utilities (aggregation, resampling, validation)
```

Pipeline entry points are provided via script-level drivers (e.g. `run_loader_pipeline.py`, 
`run_process_data_pipeline.py`) that execute the full data preparation workflow.

---

## Data sources

The software is designed to work with publicly available data, including:

* Electricity Authority of New Zealand market and demand data
* Transpower transmission network and asset information
* Technology cost and performance data from sources such as MBIE, IRENA, and CSIRO

Specific data sources, versions, and access URLs are documented in the 
accompanying documentation.

---

## Manual inputs and known limitations

### Mapping of network buses to transmission line geometries

The source datasets provide geographic coordinates for network buses (nodes) and 
separate high-resolution polyline geometries for named transmission lines. However, 
no canonical or machine-readable mapping is provided that associates individual 
bus locations with specific points along the corresponding transmission line 
geometries. The bus and line datasets are therefore spatially related but topologically 
independent. In many cases, the association between a pair of buses and a named 
transmission line is visually apparent when plotted on a map. Nevertheless, constructing 
a fully automated and robust mapping between bus points and polyline geometries 
would require substantial additional GIS tooling, including spatial snapping, 
topological inference, and manual validation in areas where multiple transmission 
corridors overlap. Existing GIS plugins were evaluated but did not provide a 
reliable or sufficiently general solution.

For this reason, a small, static mapping table is created manually to associate 
buses with transmission line identifiers. This step is performed once, based on 
visual inspection of the geospatial data, and the resulting CSV file is included 
as part of the reproducible data pipeline. The schema, units, and expected contents 
of this file are documented explicitly, and the mapping can be inspected or modified 
by users as required.

This design choice prioritises correctness and transparency over partial automation 
and reflects current limitations in the available source data rather than a 
limitation of the modelling framework itself.

---

## Outputs

The primary outputs of `pypsa-nza-data` are PyPSA-compatible CSV files, including:

* buses
* generators
* loads
* transmission lines
* links and other network components as required

These files can be loaded directly into a PyPSA `Network` object for power flow 
analysis or capacity expansion modelling.

---

## Validation

To ensure data integrity and network fidelity, the workflow includes validation 
steps such as:

* consistency checks on units and naming conventions
* verification of temporal alignment and resampling
* power flow analysis on the constructed network

These checks provide confidence that the prepared dataset is internally consistent 
and suitable for subsequent modelling studies.

---

## Intended use

This software is intended for:

* researchers performing energy system modelling of the New Zealand electricity 
sector, 
* users of PyPSA requiring transparent and reproducible input data,
* comparative or scenario-based studies where data provenance and validation are 
important.

The package may also serve as the data preparation layer for larger, ongoing 
modelling efforts, but it is fully usable as a standalone tool.

---

## License

This project is released under an open-source license. See the `LICENSE` file for 
details.

---

## Citation

If you use this software, please cite the accompanying JOSS paper (once published). 
A Zenodo DOI will be provided for archived releases.

---

## Acknowledgements

During software development, generative AI tools (including ChatGPT and Claude) 
were used to assist with code refactoring, debugging, and the generation of 
standard data-processing routines. All conceptual design, methodological decisions, 
validation, and final implementation were performed by the author.





























# pypsa-nza-data
A structured data preparation pipeline for constructing PyPSA-compatible 
representations of the New Zealand electricity system from  publicly available 
source data.

Manual mapping of buses to transmission lines

The source datasets provide geographic coordinates for network buses and separate 
high-resolution polyline geometries for named transmission lines. However, no 
explicit mapping is provided that associates individual bus locations with specific 
points along the corresponding transmission line geometries.

While such associations are often visually apparent when plotted, constructing a 
fully automated and robust mapping would require substantial additional GIS tooling 
and validation, particularly in areas where multiple transmission corridors overlap.

For this reason, a small, static mapping table linking buses to transmission line 
identifiers is created manually based on visual inspection. This step is performed 
once, documented explicitly, and the resulting CSV file is included as part of the 
reproducible data pipeline.

Manual inputs and known limitations
1. Mapping of network buses to transmission line geometries

The source datasets used by this project provide geographic coordinates for network 
buses (nodes) and separate high-resolution polyline geometries for named transmission 
lines. However, no canonical or machine-readable mapping is provided that associates 
individual bus locations with specific points along the corresponding transmission 
line geometries. The bus and line datasets are therefore spatially related but 
topologically independent.

In many cases, the association between a pair of buses and a named transmission 
line is visually apparent when plotted on a map. Nevertheless, constructing a fully 
automated and robust mapping between bus points and polyline geometries would 
require substantial additional GIS tooling, including spatial snapping, topological 
inference, and manual validation in areas where multiple transmission corridors 
overlap. Existing GIS plugins were evaluated but did not provide a reliable or 
sufficiently general solution.

For this reason, a small, static mapping table is created manually to associate 
buses with transmission line identifiers. This step is performed once, based on 
visual inspection of the geospatial data, and the resulting CSV file is included 
as part of the reproducible data pipeline. The schema, units, and expected contents 
of this file are documented explicitly, and the mapping can be inspected or modified 
by users as required.

This design choice prioritises correctness and transparency over partial automation 
and reflects current limitations in the available source data rather than a limitation 
of the modelling framework itself.


2. Transmission Line Length 
Straight line assumption for transmission lines although calculations include 
curvature of the earth. several manual checks (non-exhaustive) indicate a length ratio
between 1 and 3. for the code presented here and the sake of consistency the line 
lengths are the straight line lengths. 