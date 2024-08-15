

import os
import os.path as osp
from glob import glob
import numpy as np
import pandas as pd
import pyvista as pv

import utils as ut

#-----------------------------------------------------------------------------------------------------------------------
## Options
outputDir = r'D:/InletProfileStudy/VTPfiles/TBAD16_circle/' # path for saving probed .vtp files
saveName = 'TBAD_16_inlet'   # filename of resamples .vtp files
source_flow_dir = r'D:/InletProfileStudy/VTKfiles/TBAD16/flow' # directory containing .vtk files of a 4D flow acquisition (processed by dicoms_to_vtk.py)
source_mask_fn = r'D:/InletProfileStudy/STLgeometry/TBAD16_circle.stl' # path of the .vtk file containing the binary segmentation mask: it must be aligned with .vtk 4D flow files

#-----------------------------------------------------------------------------------------------------------------------
## Read data
flowData = [pv.read(fn) for fn in sorted(glob(osp.join(source_flow_dir, '*.vtk')))]
mask = pv.read(source_mask_fn)

#-----------------------------------------------------------------------------------------------------------------------
## Probe 4D flow data and save
os.makedirs(outputDir, exist_ok=True)
probedDir = osp.join(outputDir, 'probed_planes')
for k in range(len(flowData)):
    probed_plane = flowData[k].probe(mask)
    probed_plane.save(osp.join(outputDir, saveName + '_probed_{:02d}.vtp'.format(k)))

