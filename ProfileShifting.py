import sys
import os
import os.path as osp
import numpy as np
from glob import glob
import math
from tqdm import tqdm
import pandas as pd
import pyvista as pv
pv.set_plot_theme("document")
pv.global_theme.outline_color = 'white'
pv.global_theme.silhouette.color = pv.Color('white')
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import deque

import utils as ut
import descriptors_utils as dut


intp_options = {
    'zero_boundary_dist': 0.25,
    'zero_backflow': False,
    'kernel': 'linear',
    'smoothing': 1.5,
    'epsilon': 1,
    'degree': 0,
    'hard_noslip':False}
time_intp_options = {
    'T4df': 1,
    'Tfxd': 1,
    'num_frames_fxd': 25}

## ---> SETTINGS
rootDataDir = r'D:/InletProfileStudy/Shifting/Inputs/'
probedDataDir = osp.join(rootDataDir, 'probed_data')
rootOutDir = r'D:/InletProfileStudy/Shifting/Outputs/'
expect_peak = 7
## ---> PREPROCESSING STEPS FOR SSM
probedDataDirs = sorted(glob(osp.join(probedDataDir, '*')))
num_subjects = len(probedDataDirs)


## read vtps
vtp_path = osp.join(probedDataDirs[0],'*.vtp')
input_vtps = pv.read(sorted(glob((vtp_path))))
num_frames = len(input_vtps)


# landmark for in-plane rotation: point with max z coordinate (most left in patient's direction) !TODO: doublecheck for all patients
lm_ids = [np.argmax(input_vtps[k].points[:, 2]) for k in range(num_frames)]

# create fixed plane points
fxdpts, fxd_lm = ut.set_fixed_points(r_spac=0.05, circ_spac=5)

# -------------------------- alignment
# 0. check normals
normals = [input_vtps[k].compute_normals()['Normals'] for k in range(num_frames)]
signs = np.dot(input_vtps[3]['Velocity'], normals[3].mean(0))
# if np.sum(np.array(signs) < 0, axis=0) > np.sum(np.array(signs) > 0, axis=0):
normals = [normals[k] * -1 for k in range(num_frames)]

# 1.center at origin
coms = [np.mean(np.array(input_vtps[k].points), 0) for k in range(num_frames)]
xyz = [input_vtps[k].points - coms[k] for k in range(num_frames)]

# 2. rotate s.t. normal = [0, 0, 1]
new_normal = np.asarray([0, 0, 1])
Rots = [ut.rotation_matrix_from_vectors(normals[k].mean(0), new_normal) for k in range(num_frames)]
# Rots = [ut.rotation_matrix_from_axis_and_angle(np.cross(normals[k].mean(0), new_normal),
#        -math.acos(np.dot(normals[k].mean(0), new_normal))) for k in range(num_frames)]
pts = [Rots[k].dot(xyz[k].T).T for k in range(num_frames)]
for k in range(num_frames): pts[k][:, -1] = 0.
vel = [Rots[k].dot(input_vtps[k]['Velocity'].T).T for k in range(num_frames)]
# pts = [np.matmul(Rots[k], xyz[k].T).T for k in range(num_frames)]
# vel = [np.matmul(Rots[k], input_vtps[k]['Velocity'].T).T for k in range(num_frames)]

# 3. normalize w.r.t. max coordinate norm
Xmax = [np.max(np.sqrt(np.sum(xyz[k] ** 2, axis=1))) for k in range(num_frames)]
pts = [pts[k] / Xmax[k] for k in range(num_frames)]

# 4. second rotation to ensure consistent in-plane alignment
Rots_final = [ut.rotation_matrix_from_vectors(pts[k][lm_ids[k], :], fxd_lm) for k in range(num_frames)]
# Rots_final = [ut.rotation_matrix_from_axis_and_angle(new_normal,
#                                 math.acos(np.dot(pts[k][lm_ids[k], :], fxd_lm))) for k in range(num_frames)]
# for k in range(num_frames):
#    Rots_final[k][-1, :-1] = 0.
#    Rots_final[k][:-1, -1] = 0.
pts = [Rots_final[k].dot(pts[k].T).T for k in range(num_frames)]
vel = [Rots_final[k].dot(vel[k].T).T for k in range(num_frames)]

# create new polydatas
aligned_planes = [input_vtps[k].copy() for k in range(num_frames)]
for k in range(num_frames):
    aligned_planes[k].points = pts[k]
    aligned_planes[k]['Velocity'] = vel[k]


# -------------------------- spatial interpolation
interp_planes = ut.interpolate_profiles(aligned_planes, fxdpts, intp_options)

# -------------------------- temporal interpolation
tinterp_planes = ut.time_interpolation(interp_planes,time_intp_options)  # interpolate with 20 frames on the predefined plane
flowrate = dut.compute_flowrate(tinterp_planes)['Q(t)']
peak = np.argmax(np.abs(flowrate))
q = deque(np.arange(len(tinterp_planes)))
circshift = expect_peak-peak
q.rotate(circshift)
tinterp_planes = [tinterp_planes[k] for k in q]   # shifting
#peak_new = np.argmax(dut.compute_flowrate(new_vtps)['Q(t)'])

#output

pid = osp.basename(probedDataDirs[0])
os.makedirs(osp.join(rootOutDir, 'after_shifting', pid), exist_ok=True)

for k in range(len(tinterp_planes)):
    tinterp_planes[k].save(osp.join(rootOutDir, 'after_shifting', pid, 'prof_{:02d}.vtp'.format(k)))
    # Pts = pv.PolyData(pts[k]).delaunay_2d(alpha=10)
    # Pts['Velocity'] = vel[k]
    # Pts.save(osp.join(rootOutDir, 'after_shifting', pid, 'Pts_{:02d}.vtp'.format(k)))


