import sys
import os
import os.path as osp
import numpy as np
from glob import glob
import pyvista as pv
pv.set_plot_theme("document")
pv.global_theme.outline_color = 'white'
pv.global_theme.silhouette.color = pv.Color('white')
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from collections import deque
import descriptors_utils as dut
# from Filter import edge_filter

from scipy.spatial import distance


intp_options = {
    'zero_boundary_dist': 0.05,
    'zero_backflow': False,
    'kernel': 'linear',
    'smoothing': 1.5,
    'epsilon': 1,
    'degree': 0,
    'hard_noslip':False}
time_intp_options = {
    'T4df': 1,
    'Tfxd': 1,
    'num_frames_fxd': 20}
expected_peak =3

## ---> SETTINGS
rootDataDir = r'D:/InletProfileStudy/Shifting/Inputs/'
probedDataDir = osp.join(rootDataDir, 'probed_data')
rootOutDir = r'D:/InletProfileStudy/Shifting/Outputs/'

## ---> PREPROCESSING STEPS FOR SSM
probedDataDirs = sorted(glob(osp.join(probedDataDir, '*')))
num_subjects = len(probedDataDirs)


## read vtps
vtp_path = osp.join(probedDataDirs[0],'*.vtp')
input_vtps = pv.read(sorted(glob((vtp_path))))
num_frames = len(input_vtps)



# -------------------------- spatial interpolation
#interp_planes = ut.interpolate_profiles(aligned_planes, fxdpts, intp_options)

# -------------------------- temporal interpolation
#tinterp_planes = ut.time_interpolation(interp_planes,time_intp_options)  # interpolate with 20 frames on the predefined plane
flowrate = dut.compute_flowrate(input_vtps)['Q(t)']
peak = np.argmax(np.abs(flowrate))
q = deque(np.arange(len(input_vtps)))
circshift = expected_peak-peak
q.rotate(circshift)
input_vtps = [input_vtps[int(k)] for k in q]   # shifting
peak_new = np.argmax(dut.compute_flowrate(input_vtps)['Q(t)'])

# Filtering plane edges
#interp_planes = edge_filter(input_vtps, intp_options)  #Be careful whether it is needed






## Plot only
flow = dut.compute_flowrate(input_vtps)['Q(t)']  # If no filtering, input_vtps, otherwise interp_planes
x = range(num_frames)
plt.plot(x,flowrate,label = 'Original')
plt.plot(x,flow,label='After shifting')
plt.legend()
plt.xlabel('Timeframe ')
plt.ylabel('Flowrate ')
plt.show()




#output

pid = osp.basename(probedDataDirs[0])
os.makedirs(osp.join(rootOutDir, 'after_shifting', pid), exist_ok=True)

for k in range(len(input_vtps)):
    input_vtps[k].save(osp.join(rootOutDir, 'after_shifting', pid, 'prof_{:02d}.vtp'.format(k)))
    #Pts = pv.PolyData(pts[k]).delaunay_2d(alpha=10)
    #Pts['Velocity'] = vel[k]
    #Pts.save(osp.join(rootOutDir, 'after_shifting', pid, 'Pts_{:02d}.vtp'.format(k)))