```markdown
---
title: 'pypsa-nza-data': A Python package for processing New Zealand electricity system
data for use in PyPSA
tags:
  - Python
  - energy systems modeling
  - pypsa
  - data extraction and pre-processing
authors:
  - name: Phillippe R. Bruneau.
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, United States
   index: 1
   ror: 00hx57361
 - name: University of Canterbury, New Zealand
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

Dirk Pons (0000-0001-7141-0291)
---
## Summary

Power system modeling studies are heavily dependent on the quality, consistency, 
and transparency of their input data. In practice, preparing real-world electricity 
system datasets for modeling frameworks such as PyPSA often requires substantial 
bespoke data engineering that is rarely documented or reusable.

**pypsa-nza-data** is a Python-based data preparation and processing toolkit 
developed to construct PyPSA-compatible input datasets for the New Zealand (Aotearoa) 
electricity system. The software focuses on the acquisition, cleaning, transformation 
and harmonisation of heterogeneous publicly availabsle data sources, including demand, 
generation, and transmission network information. It provides a reproducible and 
modular workflow that converts raw source data into structured tables suitable for 
power flow and capacity expansion analysis in PyPSA.

The package is designed as a stand-alone data engineering layer rather than a 
complete power system model. It explicitly documents assumptions, limitations, 
and manual intervention steps where data gaps prevent full automation. By separating 
data preparation from model formulation and optimisation, **pypsa-nza-data** supports 
transparent and reproducible energy systems modelling workflows for New Zealand-focused 
studies.

## Statement of Need

Open-source power system modelling frameworks such as PyPSA have enabled increasingly 
sophisticated analyses of electricity networks and generation expansion. However, 
applying these frameworks to real-world systems remains challenging due to the 
fragmented, inconsistent, and incomplete nature of publicly available input data. 
As a result, substantial effort is often expended on one-off data processing pipelines 
that are not shared, documented, or reusable, limiting reproducibility and comparability 
across studies.

New Zealand presents a particularly illustrative case. Public datasets describing 
demand, generation assets, and the transmission network are available from multiple 
agencies, but they differ in structure, temporal resolution, naming conventions, 
and geographic representation. In particular, network node locations and transmission 
line geometries are provided as separate datasets without an explicit mapping, 
requiring additional interpretation and manual intervention. These challenges 
are common in applied energy systems research but are rarely addressed explicitly 
in published software.

**pypsa-nza-data** addresses this gap by providing a documented, modular, and 
reusable data preparation pipeline tailored to the New Zealand electricity system. 
The software standardises disparate source datasets, constructs PyPSA-compatible 
inputs, and makes modelling assumptions and limitations explicit. It is designed 
to support reproducible academic workflows rather than to serve as a general-purpose 
modelling framework.

While related projects such as PyPSA-Earth provide large-scale, globally applicable 
modelling infrastructures, **pypsa-nza-data** focuses on the practical challenges 
of preparing high-fidelity national-scale input data from primary sources. The 
package is intended to lower the barrier to transparent and repeatable PyPSA-based 
studies of the New Zealand power system and to provide a reference implementation 
for similar data preparation efforts in other regional contexts.

## Functionality

The functionality of **pypsa-nza-data** is organised around a modular data preparation 
pipeline that transforms heterogeneous publicly available electricity system 
datasets into PyPSA-compatible input tables. The soaftware is structured to separate 
data acquisition, processing, and validation steps, enabling reproducible and 
inspectable workflows.

### Data acquisition

The package provides loader scripts that retrieve static and time-varying electricity 
system data from publicly accessible online sources. These include datasets describing 
demand, generation assets, and transmission infrastructure. The acquisition step 
focuses on preserving source data fidelity while storing local copies for reproducibility 
and subsequent processing.

### Data processing and standardisation

Processing scripts clean and standardise acquired datasets by:

* harmonising naming conventions across sources,
* converting units and temporal resolutions,
* resolving inconsistencies and missing values,
* aggregating or disaggregating data where required for modelling.

Time series data are resampled and aligned to consistent temporal grids, supporting 
subsequent power flow and optimisation analyses.

### Load and generation profile construction

The software includes utilities for constructing electricity demand profiles and 
generation availability time series suitable for PyPSA. These workflows convert 
energy-based source data into power-based representations, apply modality-specific 
assumptions, and generate consistent profiles across all network nodes.

Renewable generation modalities such as wind and solar are treated explicitly through 
availabilsity time series, while dispatchable technologies are represented through 
static availability parameters.

### Network and transmission data handling

Transmission network inputs are assembled from publicly available node location 
and transmission line geometry datasets. Where explicit mappings between network 
nodes and transmission line geometries are not provided in source data, the 
software incorporates a documented manual mapping step based on GIS-assisted 
interpretation. This approach ensures transparency while acknowledging current 
data limitations.

### Validation and consistency checks

Helper utilities perform basic consistency checks, including:

* verification of generator site coverage,
* validation of modality assignments,
* checks on temporal completeness and alignment.

These checks are intended to identify common data issues prior to model execution 
rather than to enforce strict data validation rules.

### Configuration-driven execution

All major processing steps are controlled via a YAML configuration file that 
specifies file paths, processing options, and directory structures. This design 
enables the same codebase to be executed across different environments and platforms 
without modification and supports reproducible re-runs of the data preparation 
pipeline.

**pypsa-nza-data** does not perform power flow calculations or optimisation. 
Instead, it produces structured input datasets that are intended to be used directly 
by PyPSA for network simulation and capacity expansion studies.

## Acknowledgements

The author used AI-assisted tools, including large language models, to support 
software development tasks such as code refactoring, debugging, and the implementation 
of standard data-processing patterns (e.g. routine data manipulation and boilerplate 
functionality). All conceptual design, system architecture, modelling assumptions, 
and validation of results were performed by the author. The author takes full 
responsibility for the correctness, originality, and integrity of the software 
and associated documentation.


## References

Brown, T., H rsch, J., & Schlachtberger, D. (2018). PyPSA: Python for Power System 
Analysis. *Journal of Open Research Software*, 6(1), 4. [https://doi.org/10.5334/jors.188](https://doi.org/10.5334/jors.188)

Electricity Authority of New Zealand. (n.d.). Electricity market and demand data. 
Retrieved from [https://www.ea.govt.nz/](https://www.ea.govt.nz/)

Transpower New Zealand. (n.d.). Transmission network and asset data. Retrieved 
from [https://www.transpower.co.nz/](https://www.transpower.co.nz/)

Ministry of Business, Innovation and Employment (MBIE). (n.d.). Energy statistics 
and projections. Retrieved from [https://www.mbie.govt.nz/](https://www.mbie.govt.nz/)

International Renewable Energy Agency (IRENA). (2023). Renewable power generation 
costs. [https://www.irena.org/](https://www.irena.org/)


https://elements.canterbury.ac.nz/viewobject.html?cid=1&id=501998


https://elements.canterbury.ac.nz/viewobject.html?cid=1&id=501986
