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
time_intp_options = {
    'T4df': 0.858,   # adjust to the real period.  for patients with 4D-flow, it should be the value from 4D-flow.
                                                # For patients without 4D-flow, it should be the same value as expected period
    'Tfxd': 0.858,   # fixed period, same as T4df in patient-specific study ?

    'num_frames_fxd': 20}


profilesDir = r'D:/InletProfileStudy/ReadandScale/scaling/P2_SINE/'



saveName = 'healthy_02'
cfd_delta_t = 0.001  # simulation time steps
cardiac_cycle_period = 0.858
time_interpolation = 'cubic'
interp_planes = [pv.read(fn) for fn in sorted(glob(osp.join(profilesDir, '*.vtp')))]
num_frames = len(interp_planes)


# tcfd = np.arange(0, cardiac_cycle_period, cfd_delta_t)
tcfd = np.arange(0, cardiac_cycle_period , cfd_delta_t)
timepoints = len(tcfd)

time_intp_options['num_frames_fxd'] = timepoints

tinterp_planes = ut.time_interpolation(interp_planes,time_intp_options)

flow =ut.compute_flowrate(tinterp_planes)['Q(t)']
plt.plot(flow)
plt.show()

