# 4Dflow Python Codes

Python scripts for preprocessing, analysis, and visualization of 4D flow inlet velocity profiles, with a focus on patient-specific and statistical shape model workflows.

## Overview

This repository contains standalone scripts used to:

- preprocess probed 4D flow velocity data,
- align and shift temporal flow profiles,
- build statistical shape model inputs,
- run PCA-based analyses,
- visualize flow-related quantities and extracted volumes.

Most scripts are configured as research workflows rather than packaged command-line tools. They currently use hard-coded local paths and expect project-specific input data on disk.

## Main Scripts

- `preProcess_PatientSpecific.py`: preprocess patient-specific inlet profiles and generate fixed-plane representations.
- `preProcess_circular.py`: preprocess data on an idealized circular plane.
- `ProfileShifting.py`: temporally align profiles by shifting peak flow to an expected frame.
- `mainSSM_SV.py`: perform PCA-based statistical analysis on preprocessed velocity profile matrices.
- `PCA plot.py`: generate PCA plots and related variance visualizations.
- `syntheticGeneration_kaihong.py`: create synthetic samples from the statistical model.
- `FlowPlot.py`, `IVP_visualization.py`, `VelocityMean.py`, `Volume_extraction.py`: plotting and post-processing utilities.
- `utils.py`, `descriptors_utils.py`: shared helper functions used across the workflow.

## Typical Workflow

1. Prepare probed 4D flow profile data and target plane geometry.
2. Run preprocessing for either patient-specific or circular representations.
3. Apply temporal profile shifting if alignment is required.
4. Export or load the resulting matrix representation.
5. Run PCA and downstream analysis or synthetic profile generation.

## Dependencies

The scripts import the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `pyvista`
- `scikit-learn`
- `scipy`
- `tqdm`

Depending on the workflow, a Qt-compatible Matplotlib backend may also be required because several scripts explicitly use `Qt5Agg`.

## Notes

- Many scripts assume Windows-style absolute paths such as `D:/...`.
- Inputs are expected to be VTK-based files such as `.vtp`, `.vtk`, or `.stl`.
- The repository is currently script-oriented and does not yet provide a single entry point or environment file.

## Repository Status

This README reflects the current code layout and usage patterns visible in the repository. If you want, the next useful step would be to turn this into a fuller project README with setup instructions, input/output examples, and a reproducible run sequence.
