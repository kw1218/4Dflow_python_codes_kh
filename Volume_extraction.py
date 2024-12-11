
import os
import os.path as osp
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyvista as pv

#-----------------------------------------------------------------------------------------------------------------------
## Options
outputDir = r'D:/InletProfileStudy/Volume_Extraction/TBAD02_part' # path for saving probed .vtp files
saveName = 'TBAD_02_part'   # filename of resamples .vtp files
source_flow_dir = r'D:/InletProfileStudy/VTKfiles/TBAD02/flow' # directory containing .vtk files of a 4D flow acquisition (processed by dicoms_to_vtk.py)
source_mask_fn = r'D:/InletProfileStudy/STLgeometry/Volume/TBAD02_part.stl' # path of the .vtk file containing the binary segmentation mask: it must be aligned with .vtk 4D flow files

#-----------------------------------------------------------------------------------------------------------------------
## Read data
flowData = [pv.read(fn) for fn in sorted(glob(osp.join(source_flow_dir, '*.vtk')))]
mask = pv.read(source_mask_fn)
mask = mask.extract_largest()
mask = mask.clean()

#-----------------------------------------------------------------------------------------------------------------------
## Probe 4D flow data and save
os.makedirs(outputDir, exist_ok=True)
probedDir = osp.join(outputDir, 'volume_extracted')
for i in tqdm(range(len(flowData)), desc='Processing and saving segmented vtks'):
    vol_enclosed=flowData[i].select_enclosed_points(mask)
    inside = vol_enclosed.threshold(0.01)
    inside.save(osp.join(outputDir, saveName + '_{:02d}'.format(i) + '.vtk'), binary=True)
    # inside.plot()
print('Segmented vtks saved.')

