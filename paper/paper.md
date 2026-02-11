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

## State of the field

Electricity system modelling frameworks such as PyPSA [@brown2018pypsa] require
substantial pre-processing of input data before analysis can begin. In many
jurisdictions, including New Zealand, relevant datasets are publicly available
but distributed across multiple institutions [@ea_nz; @transpower_nz], provided
in heterogeneous formats, and lack consistent temporal and spatial alignment.
As a result, researchers typically implement bespoke, study-specific data
pipelines that are difficult to reproduce and compare across studies.

Several PyPSA-related data workflows exist internationally, often tailored to
European or global datasets. However, no open-source, end-to-end data
preparation pipeline has previously been published for New Zealand that
formalises data acquisition, validation, transformation, and export into
PyPSA-compatible formats.

## Statement of need

pypsa_nza_data addresses this gap by providing a transparent and reproducible
data preparation workflow tailored specifically to New Zealand’s electricity
system. By formalising data acquisition, cleaning, and transformation into a
single, documented pipeline, the package reduces duplication of effort and
enables consistent use of PyPSA-based models in academic and policy research
contexts.

Earlier versions of the data preparation workflow and resulting datasets were
used in conference studies [@bruneau_conf1; @bruneau_conf2]. The present package
generalises and documents this workflow as a reusable open-source software
tool, allowing other researchers to reproduce, extend, and audit the data
inputs used in PyPSA-based analyses.

## Software design

The pypsa_nza_data package is organised into loader, processor, utility, and
demonstration components. Loader modules download raw static and time-series
datasets from authoritative public sources, including generation, demand, and
transmission information. Processor modules clean and transform these datasets
into consistent formats, performing temporal alignment, aggregation, and unit
conversion where required.

All data products are written to a user-defined external workspace, ensuring
that the repository remains free of generated data and that results are
reproducible across platforms. Configuration files define all paths relative to
this workspace, avoiding reliance on environment variables or hard-coded
locations.

Two demonstration scripts are provided. The first inventories all raw,
processed, and derived datasets to provide transparent provenance. The second
constructs a PyPSA network for a specified year and exports monthly NetCDF
network files along with a network topology plot, demonstrating successful
integration with PyPSA without performing optimisation.

## Research impact

By providing a reproducible and auditable data preparation pipeline, 
pypsa_nza_data lowers the barrier to PyPSA-based electricity system modelling
for New Zealand. The package supports consistent comparison of scenarios,
methods, and assumptions across studies, improving transparency and
reproducibility in academic research.

The software is intended for use by researchers, students, and analysts
conducting capacity expansion, dispatch, and scenario studies of New
Zealand’s electricity system, and provides a foundation for future extensions
such as additional datasets, alternative temporal resolutions, or integration
with other modelling frameworks.

## AI usage disclosure

AI-assisted tools, including large language models, were used to support
routine software development tasks such as refactoring and debugging. All
conceptual design, modelling assumptions, and validation of results were
performed by the first author, who takes full responsibility for the software and
documentation.

## Acknowledgements

The authors acknowledge the providers of public electricity system data used
by this package, including the Electricity Authority of New Zealand and
Transpower New Zealand.

## References
