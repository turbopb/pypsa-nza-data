---
title: "pypsa-nza-data: Reproducible preparation of New Zealand electricity system data for PyPSA"
tags:
  - python
  - energy systems modelling
  - pypsa
  - data preparation
authors:
  - name: Phillippe Bruneau
    orcid: 0009-0002-7136-3477
    corresponding: true
    affiliation: 1

  - name: Dirk Pons
    orcid: 0000-0001-7141-0291
    affiliation: 1

  - name: Alan Wood
    affiliation: 1

  - name: Patrick Shepard
    affiliation: 1

affiliations:
 - name: University of Canterbury, New Zealand
   index: 1
   ror: 00hx57361

bibliography: paper.bib
---

## Summary

pypsa_nza_data is an open-source Python package that provides a reproducible
data preparation pipeline for constructing electricity system datasets for
New Zealand suitable for use with PyPSA. The package downloads authoritative
public electricity data, performs cleaning, validation, temporal alignment,
and unit conversion, and produces consistent, PyPSA-compatible outputs. All
data products are written to a user-defined external workspace, ensuring
transparent provenance and reproducibility across platforms. Demonstration
scripts are included to inventory available datasets and to construct PyPSA
networks and export NetCDF files, illustrating correct integration without
performing optimisation.

## Statement of Need

Electricity system modelling frameworks such as PyPSA [@brown2018pypsa] require substantial
pre-processing of input data before analysis can begin. For New Zealand,
relevant datasets are publicly available but are distributed across multiple
institutions [@ea_nz; @transpower_nz], provided in heterogeneous formats, and lack consistent 
temporal and spatial alignment. As a result, researchers frequently implement bespoke,
non-reproducible data pipelines, limiting transparency and comparability
between studies.

Earlier versions of the data preparation workflow and resulting datasets were
used in conference studies [@bruneau_conf1; @bruneau_conf2]. The present package
formalises and generalises the workflow into a reusable, documented and open-source 
software tool.

pypsa_nza_data addresses this gap by providing a transparent and reproducible
data preparation workflow tailored to New Zealand’s electricity system. By
formalising data acquisition, cleaning, and transformation into a single,
documented pipeline, the package reduces duplication of effort and enables
consistent use of PyPSA-based models in academic and policy research contexts.

## Functionality

The pypsa_nza_data package is organised into loader, processor, utility, and
demonstration components. Loader modules download raw static and time-series
datasets from authoritative public sources, including generation, demand, and
transmission information. Processor modules clean and transform these datasets
into consistent formats, performing temporal alignment, aggregation, and
conversion from energy to power units where required.

All data are written to a user-defined external workspace, ensuring
that the repository remains free of generated data and that results are
reproducible across platforms. Configuration files define all paths relative
to this workspace, avoiding reliance on environment variables or hard-coded
locations.

Two demonstration scripts are provided. The first creates inventories of all raw,
processed, and derived datasets to provide transparent provenance. The second
constructs a PyPSA network for a specified year and exports monthly NetCDF
network files along with a network topology plot, demonstrating successful
integration with PyPSA without performing optimisation.

## Acknowledgements

AI-assisted tools, including large language models, were used to support routine
software development tasks such as refactoring and debugging. All conceptual
design, modelling assumptions, and validation of results were performed by the
first author, who takes full responsibility for the software and documentation.

## References
