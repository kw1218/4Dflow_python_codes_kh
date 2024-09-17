

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


## ---> SETTINGS
rootDataDir = r'D:/InletProfileStudy/SSM/Input/'
probedDataDir = osp.join(rootDataDir, 'Probed_root')
rootOutDir = r'D:/InletProfileStudy/SSM/Output_2024/Circular/'
Fig_Dir = osp.join(rootOutDir, 'Figure')
os.makedirs(Fig_Dir, exist_ok=True)

#target_profile_fn = r'D:/InletProfileStudy/STLgeometry/P5mesh_1100.stl'  # can be a .stl, .vtk or .vtp file

intp_options = {
    'zero_boundary_dist': 0.02,
    'zero_backflow': False,
    'kernel': 'linear',
    'smoothing': 1.5,
    'epsilon': 1,
    'degree': 0,
    'hard_noslip':False}
time_intp_options = {
    'T4df': 1,   # adjust to the real period.  for patients with 4D-flow, it should be the value from 4D-flow.
                                                # For patients without 4D-flow, it should be the same value as expected period
    'Tfxd': 1,   # fixed period, same as T4df in patient-specific study ?
    'num_frames_fxd': 20}
#  read flowrate from ECG analysis

## ---> PREPROCESSING STEPS FOR SSM
probedDataDirs = sorted(glob(osp.join(probedDataDir, '*')))
num_subjects = len(probedDataDirs)
i = 12
VEL = []
sub_dict_list = []
print('Preprocessing dataset...')

for i in tqdm(range(num_subjects)):

    pid = osp.basename(probedDataDirs[i])
    os.makedirs(osp.join(rootOutDir, 'clinical_cohort', pid), exist_ok=True)

    # read input data
    input_vtps = [pv.read(i) for i in sorted(glob(osp.join(probedDataDirs[i], '*.vtp')))]
    num_frames = len(input_vtps)


    # adjust units to m and m/s
    input_vtps = ut.adjust_units(input_vtps, array_name='Velocity')

    # landmark for in-plane rotation: point with max z coordinate (most left in patient's direction) !TODO: doublecheck for all patients
    lm_ids = [np.argmax(input_vtps[k].points[:, 0]) for k in range(num_frames)]

    # create fixed plane points, idealized circle plane
    fxdpts, fxd_lm = ut.set_fixed_points(r_spac=0.05, circ_spac=5)

    # patient-specific plane
    #target_plane = pv.read(target_profile_fn)

    # fxdpts = target_plane.points
    # fxd_lm_id = np.argmax(fxdpts[:,0])
    # fxd_lm = fxdpts[fxd_lm_id]

    # -------------------------- alignment
    # 0. check normals
    normals = [input_vtps[k].compute_normals()['Normals'] for k in range(num_frames)]
    signs = np.dot(input_vtps[5]['Velocity'], normals[5].mean(0))
    below = np.sum(np.array(signs) < 0, axis=0)
    above = np.sum(np.array(signs) > 0, axis=0)
    #if np.sum(np.array(signs) < 0, axis=0) > np.sum(np.array(signs) > 0, axis=0):
    #normals = [normals[k] * -1 for k in range(num_frames)]
    # target_com = fxdpts.mean(0)  # centre of mass of points on the reference plane
    # dis2 = target_plane.points.mean(0)
    # target_normal = target_plane.compute_normals()['Normals'].mean(0)

    # 1.center at origin
    coms = [np.mean(np.array(input_vtps[k].points), 0) for k in range(num_frames)]
    xyz = [input_vtps[k].points - coms[k] for k in range(num_frames)]
    # fxdpts -= target_com

    # 2. rotate s.t. target normal
    new_normal = np.asarray([0, 0, 1])
    Rots = [ut.rotation_matrix_from_vectors(normals[k].mean(0), new_normal) for k in range(num_frames)]
    # Rots = [ut.rotation_matrix_from_axis_and_angle(np.cross(normals[k].mean(0), new_normal),
    #        -math.acos(np.dot(normals[k].mean(0), new_normal))) for k in range(num_frames)]
    pts = [Rots[k].dot(xyz[k].T).T for k in range(num_frames)]
    for k in range(num_frames): pts[k][:, -1] = 0.   ## ?
    vel = [Rots[k].dot(input_vtps[k]['Velocity'].T).T for k in range(num_frames)]
    # pts = [np.matmul(Rots[k], xyz[k].T).T for k in range(num_frames)]
    # vel = [np.matmul(Rots[k], input_vtps[k]['Velocity'].T).T for k in range(num_frames)]

    # 3. normalize w.r.t. max coordinate norm
    #target_max = np.max(np.sqrt(np))
    Xmax = [np.max(np.sqrt(np.sum(xyz[k] ** 2, axis=1))) for k in range(num_frames)]
    targetmax = np.max(np.sqrt(np.sum(fxdpts ** 2, axis=1)))
    ratio = targetmax / Xmax[1]
    pts = [pts[k] * ratio for k in range(num_frames)]

    # 4. second rotation to ensure consistent in-plane alignment
    Rots_final = [ut.rotation_matrix_from_vectors(pts[k][lm_ids[k], :], fxd_lm) for k in range(num_frames)]
    # Rots_final = [ut.rotation_matrix_from_axis_and_angle(new_normal,
    #                                 math.acos(np.dot(pts[k][lm_ids[k], :], fxd_lm))) for k in range(num_frames)]
    # for k in range(num_frames):
    #    Rots_final[k][-1, :-1] = 0.
    #    Rots_final[k][:-1, -1] = 0.
    pts = [Rots_final[k].dot(pts[k].T).T for k in range(num_frames)]
    vel = [Rots_final[k].dot(vel[k].T).T for k in range(num_frames)]
    # pts = [np.matmul(Rots_final[k], pts[k].T).T for k in range(num_frames)]
    # vel = [np.matmul(Rots_final[k], vel[k].T).T for k in range(num_frames)]



    # create new polydatas
    aligned_planes = [input_vtps[k].copy() for k in range(num_frames)]


    for k in range(num_frames):
        aligned_planes[k].points = pts[k]
        aligned_planes[k]['Velocity'] = vel[k]

    # -------------------------- spatial interpolation

    interp_planes = ut.interpolate_profiles(aligned_planes, fxdpts, intp_options)
    #interp_planes[3].warp_by_vector(factor=0.5).plot(scalars='Velocity')

    #displacement = interp_planes[0].points.mean(0) - dis2

    #recenter
    # for k in range(num_frames):
    #     #interp_planes[0].points.mean(0) - dis2
    #     interp_planes[k].points -= interp_planes[0].points.mean(0) - dis2



    # --------------------------- shift before scale
    nnn = interp_planes[1].compute_normals()
    flowrate = dut.compute_flowrate(interp_planes)['Q(t)']
    peak = np.argmax(np.abs(flowrate))
    q = deque(np.arange(len(interp_planes)))
    circshift = 3 - peak  # define the index of peak flowrate
    q.rotate(circshift)
    interp_planes = [interp_planes[int(k)] for k in q]  # shifting



    # -------------------------- temporal interpolation
    tinterp_planes = ut.time_interpolation(interp_planes, time_intp_options) # interpolate with 20 frames on the predefined plane

    for k in range(len(tinterp_planes)):
        tinterp_planes[k].save(osp.join(rootOutDir, 'clinical_cohort', pid, 'prof_{:02d}.vtp'.format(k)))
        Pts = pv.PolyData(pts[k]).delaunay_2d(alpha=10)
        Pts['Velocity'] = vel[k]
        Pts.save(osp.join(rootOutDir, 'clinical_cohort', pid, 'Pts_{:02d}.vtp'.format(k)))


    descrDir = osp.join(rootOutDir, 'real_descriptors')
    os.makedirs(descrDir, exist_ok=True)
    flowDescriptors = dut.compute_flow_descriptors(tinterp_planes)
    np.save(osp.join(descrDir, pid + '_descriptors.npy'), flowDescriptors, allow_pickle=True)

    VEL.append([tinterp_planes[k]['Velocity'] for k in range(len(tinterp_planes))])

VEL = np.array(VEL)


# assemble and save column matrix V
n_pat, n_frames, n_nodes = np.shape(VEL)[:-1]     # .shape()[:-1] get the shape without the last column

V = np.empty((n_pat, n_frames*n_nodes*3)) #This represents the velocities of all nodes and frames for a specific patient.
for i in range(n_pat):
    uvw = VEL[i].flatten()
    V[i, :] = uvw        # PCA requires both variables and observations to be assembled into a single column vector

feat_cols = ['vel'+str(i) for i in range(V.shape[1])]
df = pd.DataFrame(V, columns=feat_cols)
df.to_csv(osp.join(rootOutDir, 'matrixV.csv'), index=False)


# Compute and save mean time-dependent profile
meanProfDir = osp.join(rootOutDir, 'mean_profile')
os.makedirs(meanProfDir, exist_ok=True)
mean_prof = np.mean(V, 0).reshape((n_frames, n_nodes, 3))
mean_planes = [interp_planes[0].copy() for _ in range(n_frames)]

for k in range(len(mean_planes)):
    mean_planes[k]['Velocity'] = mean_prof[k]

    mean_planes[k].save(osp.join(meanProfDir, 'meanProf_{:02d}.vtp'.format(k)))
mean_planes[5].warp_by_vector(factor=0.05).plot(scalars='Velocity')   # first figure


## for a figure

pv.global_theme.transparent_background = True
pv.global_theme.hidden_line_removal = True

pl = pv.Plotter()
#pl.add_points(aligned_planes[3].points, color='red')
pl.add_points(interp_planes[5].points, color='green')  # here 3 is related to the peak flow point, show points
pl.camera_position = 'xy'
pl.camera.zoom(1.3)
pl.show(screenshot=osp.join(Fig_Dir, 'points_rbf_kh.png'))


pl = pv.Plotter()
pl.add_mesh(mean_planes[3], clim=[0,1.4], scalars='Velocity', cmap='jet')
#pl.add_mesh(interp_planes[3], clim=[0,1.4], scalars='Velocity', cmap='jet')
pl.camera_position = 'xy'
pl.camera.zoom(1.3)
pl.remove_scalar_bar()
pl.show(screenshot=osp.join(Fig_Dir, 'mean_profile_raw_kh.png'))


