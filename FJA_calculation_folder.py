import sys
import os
import os.path as osp
import numpy as np
from glob import glob
import pyvista as pv
import descriptors_utils as dut
from matplotlib import pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------------------------------------------------
rootDir=r'D:\InletProfileStudy\SSM\Output_2024\Circular\synthetic_cohort_first8modes'  # path to synthetic files or sampled profiles

# IVP_number = '002'
# IVP_path = osp.join(rootDir,IVP_number)
# vtp_path = osp.join(IVP_path, '*.vtp')
# os.makedirs(Out_Dir, exist_ok=True)



#-----------------------------------------------------------------------------------------------------------------------

## Read IVPs
folders = [f.path for f in os.scandir(rootDir) if f.is_dir()]
fja_all = 0
fdi_all = 0
for folder in folders:
    vtp_path=osp.join(folder, '*.vtp')
    input_vtps = pv.read(sorted(glob((vtp_path))))
    ## PPV calculation
    PPV_all = dut.compute_positive_peak_velocity(input_vtps)

    ## FJA calculation
    FJA_all=dut.compute_flow_jet_angle(input_vtps)

    ## FDI calculation
    FDI_all = dut.compute_flow_dispersion(input_vtps)

    ## SFD calculation
    SFD_all = dut.compute_secondary_flow_degree(input_vtps)

    ## HFI calculation
    HFI_all = dut.compute_helical_flow_index(input_vtps)

    fja_all += FJA_all['fja_mean']
    fdi_all += FDI_all['fdi_mean']

avg_fja = fja_all/len(folders)
avg_fdi = fdi_all/len(folders)
print('Average FJA:', avg_fja)
print('Average FDI:', avg_fdi)


# ## Plot
# input_vtps[5].plot(scalars='Velocity',clim=[0,0.5],cmap='jet')
# input_vtps[5].warp_by_vector(factor=0.5).plot(scalars='Velocity',clim=[0,0.5],cmap='jet')


