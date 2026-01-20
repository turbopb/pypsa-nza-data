# pypsa-nza-data

**pypsa-nza-data** is a Python-based data preparation and processing toolkit for 
* constructing PyPSA-compatible power system datasets for New Zealand (Aotearoa).
It focuses on the **acquisition, cleaning, transformation, and harmonisation** of 
publicly available electricity system data into a form suitable for power flow and 
capacity expansion analysis using [PyPSA](https://pypsa.org).

The software has been developed to support reproducible energy systems modelling 
and research workflows, particularly in academic and policy-oriented studies of 
the New Zealand electricity system.


## Scope and purpose

This repository provides tools to:

* acquire static and time-varying electricity system data from public sources,
* clean and standardise heterogeneous datasets with inconsistent formats and naming,
* construct load, generation, and network input tables compatible with PyPSA,
* generate derived time series (e.g. demand profiles, renewable availability),
* perform consistency checks on generator locations, modalities, and site coverage.

The package is **not** a complete power system model.
Instead, it provides the **data engineering layer** required before PyPSA-based 
modelling can be undertaken.


## Data sources

The software primarily uses publicly available New Zealand electricity system data, 
including (but not limited to):

* Electricity Authority of New Zealand datasets,
* Transpower New Zealand grid and transmission data,
* MBIE energy statistics and demand data,
* publicly available renewable generation site information.

All source data remain subject to their original licences and attribution requirements.
This repository does **not** redistribute raw proprietary data.


## Repository structure

The package follows a modular structure organised by function:


=======
<pre>'''pypsa_nza_data/|

+-- loaders/|
    +-- nza_load_static_data_from_url.py|
    +-- nza_load_dynamic_data_from_url.py|
    +-- run_loader_pipeline.py|
 
+-- processors/|
    +-- nza_process_static_data.py|
    +-- nza_process_dynamic_data.py|
    +-- nza_create_load_profile.py|
    +-- nza_convert_energy_to_power.py|
    +-- nza_modality_standard.py|
    +-- nza_transmission_line_processor.py|
    +-- run_process_data_pipeline.py|
    +-- helpers/|
        +-- nza_resample_timeseries.py|
        +-- nza_process_generator_data.py|
        +-- ...|
 
+-- config/|
   +-- paths_and_settings.yaml|

+-- README.md|'''<pre>


=======



The two main execution entry points are:

* `run_loader_pipeline.py` Â– acquires and stores raw source data
* `run_process_data_pipeline.py` Â– transforms raw data into PyPSA-ready inputs


## Configuration and portability

All file paths, directory locations, and processing options are defined in a **YAML configuration file**.
This design choice ensures:

* portability across Windows and Linux platforms,
* reproducibility when the repository is cloned,
* separation of code logic from environment-specific paths.

Users are expected to edit the YAML configuration file to define:

* root data directories,
* intermediate and output locations,
* optional processing switches.

No hard-coded absolute paths are required.


## Transmission line geometry mapping (manual step)

One processing step requires **manual intervention**, which is documented explicitly here.

### Background

Source datasets provide:

* geographic point locations for network nodes (buses),
* separate, high-resolution polyline geometries for named transmission lines.

However, there is **no provided mapping** between node locations and exact points along the transmission line geometries.
Although node-to-node connectivity is visually apparent when plotted on a map, it cannot be reliably inferred programmatically from the available data alone.

### Current approach

To resolve this, a small CSV file is created manually that associates:

* node identifiers,
* corresponding transmission line segments.

This mapping is informed by visual inspection using GIS tools (e.g. QGIS).

### Rationale and limitations

* A fully automated solution would require either:

  * interactive GUI-based selection tools, or
  * complex spatial inference algorithms.
* Developing and validating such tooling was beyond the scope of the current project.
* The chosen approach is pragmatic, transparent, and reproducible once the mapping is defined.

This limitation is documented clearly and is not hidden from users or reviewers.


## Installation and platform compatibility

The codebase is written in pure Python and is designed to run transparently on both **Windows** and **Linux** platforms.

Development and testing have primarily been carried out on Windows systems; however, the software does not rely on operating-system-specific features, shell commands, or path conventions. All file and directory handling is performed using standard Python libraries, and environment-specific configuration is isolated via YAML configuration files.

Linux compatibility is explicitly tested as part of the project workflow to ensure portability and reproducibility across common research environments.

Formal installation instructions (including package installation and execution from the command line) will be finalised prior to publication.


## Intended use

The software is intended for:

* researchers building PyPSA-based models of the New Zealand electricity system,
* reproducible academic workflows,
* scenario and sensitivity analysis of generation and demand inputs.

It is not intended to be a turn-key modelling framework.


## Relationship to PyPSA

This repository produces **input data only**.
Simulation, optimisation, and analysis are performed downstream using PyPSA.


## Limitations

* The software reflects the structure and completeness of publicly available data.
* Some assumptions and simplifications are unavoidable due to data gaps.
* Transmission line geometry mapping currently requires a documented manual step.
* The software does not perform optimisation or solve power system models.

All limitations are documented explicitly to support transparent interpretation of results.


## Citation and attribution

If you use this software in academic work, please cite the associated JOSS paper (once published).


## License

An open-source licence will be added prior to public release.


## Acknowledgements

The development of this software benefited from modern AI-assisted tools for refactoring, debugging, and implementation of standard data-processing patterns.
All conceptual design, system architecture, and validation were performed by the author.
