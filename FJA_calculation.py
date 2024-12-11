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
# rootDir=r'D:\InletProfileStudy\SSM\Output_2024\Circular\synthetic_cohort_first8modes'  # path to synthetic files or sampled profiles
Out_Dir = r'D:/InletProfileStudy/SSM/Input/flow_csv'
# IVP_number = '002'
# IVP_path = osp.join(rootDir,IVP_number)
# vtp_path = osp.join(IVP_path, '*.vtp')
os.makedirs(Out_Dir, exist_ok=True)
Output_csv = False


rootDir=r'D:\InletProfileStudy\SSM\Output_2024\Circular\clinical_cohort\TBAD05'
vtp_path = osp.join(rootDir, 'prof*.vtp')


#-----------------------------------------------------------------------------------------------------------------------

## Read IVPs

input_vtps = pv.read(sorted(glob((vtp_path))))


# ## Plot
# input_vtps[5].plot(scalars='Velocity',clim=[0,0.5],cmap='jet')
# input_vtps[5].warp_by_vector(factor=0.5).plot(scalars='Velocity',clim=[0,0.5],cmap='jet')


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

# print('PPV at peak systole:', PPV_all['ppv_systole'])
# print('PPV mean:', PPV_all['ppv_mean'])

print('FJA at peak systole:', FJA_all['fja_systole'])
print('FJA mean:', FJA_all['fja_mean'] -180)

print('FDI at peak systole:', FDI_all['fdi_systole'])
print('FDI mean:', FDI_all['fdi_mean'])

# print('SFD at peak systole:', SFD_all['sfd_systole'])
# print('SFD mean:', SFD_all['sfd_mean'])

# print('HFI at peak systole:', HFI_all['hfi_systole'])
# print('HFI mean:', HFI_all['hfi_mean'])

## Plot
flow=dut.compute_flowrate(input_vtps)['Q(t)']
peak = np.argmax(flow)
peak = int(peak)



input_vtps[peak].plot(scalars='Velocity',clim=[0,0.5],cmap='jet')
input_vtps[peak].warp_by_vector(factor=0.5).plot(scalars='Velocity',clim=[0,0.5],cmap='jet')