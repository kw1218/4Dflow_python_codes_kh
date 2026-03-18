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

## Main steps
1. dicom_to_vtk .py: after obtaining .vtk files, open it in Paraview, then exported as .e file————open .e file in Ensight, and crop a plane at the inlet

1. plane_selection_meshinlet.py: the plane cropped from step 1 will be used to generate .vtp files
2. Preprocessing.py: this code is to calculate the mean velocity profile for all subjects.
3. mainSSM.py
4. syntheticDatasetGeneration.py: set the size of synthetic data, and acceptance criteria
