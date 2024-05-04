import sys
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pyvista as pv
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.interpolate import interp1d
import utils as ut
import descriptors_utils as dut
from scipy import interpolate


profilesDir = r'D:/InletProfileStudy/ReadandScale/scaling/'
entries = os.listdir(profilesDir)
sub_folders = [entry for entry in entries if os.path.isdir(os.path.join(profilesDir, entry))]
num_folders = len(sub_folders)

vel = []
vel_mean={}
vel_0 ={}
vel_1 ={}
vel_2 ={}


for i in range(num_folders):
    folder_path = os.path.join(profilesDir,sub_folders[i])
    interp_planes= [pv.read(fn) for fn in sorted(glob(osp.join(folder_path, '*.vtp')))]
    vel = np.array([interp_planes[k]['Velocity'] for k in range(len(interp_planes))])
    vel_mean[sub_folders[i]] =vel[3].mean(0)
    vel_0.setdefault(sub_folders[i],[])
    vel_1.setdefault(sub_folders[i],[])
    vel_2.setdefault(sub_folders[i],[])
    for k in range(len(interp_planes)):
        vel_0[sub_folders[i]].append(vel[k].mean(0)[0])
        vel_1[sub_folders[i]].append(vel[k].mean(0)[1])
        vel_2[sub_folders[i]].append(vel[k].mean(0)[2])

fig, axs = plt.subplots(1, 4, figsize=(10, 5))  # figsize can be adjusted as needed

# Plotting on the first subplot
axs[0].plot(vel_0['P5_case02'], label='rl direction')
axs[0].plot(vel_1['P5_case02'], label='ap direction')
axs[0].plot(vel_2['P5_case02'], label='fh direction')
axs[0].set_title('case02')
axs[0].legend()

# Plotting on the second subplot
axs[1].plot(vel_0['P5_case03'], label='rl direction')
axs[1].plot(vel_1['P5_case03'], label='ap direction')
axs[1].plot(vel_2['P5_case03'], label='fh direction')
axs[1].set_title('case03')
axs[1].legend()

# 3rd
axs[2].plot(vel_0['P5_case04'], label='rl direction')
axs[2].plot(vel_1['P5_case04'], label='ap direction')
axs[2].plot(vel_2['P5_case04'], label='fh direction')
axs[2].set_title('case04')
axs[2].legend()

# 4th
axs[3].plot(vel_0['P5_case05'], label='rl direction')
axs[3].plot(vel_1['P5_case05'], label='ap direction')
axs[3].plot(vel_2['P5_case05'], label='fh direction')
axs[3].set_title('case05')
axs[3].legend()
# Display the plot
plt.show()



