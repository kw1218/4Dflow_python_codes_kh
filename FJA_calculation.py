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
# Out_Dir = r'D:/InletProfileStudy/SSM/Input/flow_csv'
# IVP_number = '010'
# IVP_path = osp.join(rootDir,IVP_number)
# vtp_path = osp.join(IVP_path, '*.vtp')
# os.makedirs(Out_Dir, exist_ok=True)
Output_csv = False


rootDir=r'D:\InletProfileStudy\SSM\Output_2024\Circular\mean_profile'
vtp_path = osp.join(rootDir, '*.vtp')


#-----------------------------------------------------------------------------------------------------------------------

## Read IVPs

input_vtps = pv.read(sorted(glob((vtp_path))))

## FJA calculation
FJA_all=dut.compute_flow_jet_angle(input_vtps)

print('FJA at peak systole:', FJA_all['fja_systole'])
print('FJA mean:', FJA_all['fja_mean'])
