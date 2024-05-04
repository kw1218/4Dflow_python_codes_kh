import sys
import os
import os.path as osp
import numpy as np
from glob import glob
from tqdm import tqdm
import pandas as pd
import pyvista as pv
pv.set_plot_theme("Document")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import descriptors_utils as dut
from scipy import interpolate
from collections import deque
import utils as ut


time_intp_options = {
    'T4df': 1,   # adjust to the real period.  for patients with 4D-flow, it should be the value from 4D-flow.
                                                # For patients without 4D-flow, it should be the same value as expected period
    'Tfxd': 1,   #

    'num_frames_fxd': 20}


preprocDir = r'D:/InletProfileStudy/ReadandScale/mapped'
outputDir = r'D:/InletProfileStudy/ReadandScale/scaling/P1_SINE'
csv_path = r'D:/InletProfileStudy/SSM/Output_Kaihong/Patient_5_new700/Flowrate.csv'  # path to well-tuned flowwaveform .csv file
os.makedirs(outputDir, exist_ok=True)
synthetic_planes = [pv.read(fn) for fn in sorted(glob(osp.join(preprocDir, 'P1', '*.vtp')))]  ## select the mapped file folder

SV111= dut.compute_flowrate(synthetic_planes)['Q(t)']
total_SV11 = np.sum(SV111)* 1000*1000

## read flow waveform from Matlab output
tuned_flowrate_csv = pd.read_csv(csv_path, header=None)
tuned_flowrate_csv.rename(columns={0:'Velocity'},inplace=True)
tuned_velocity = tuned_flowrate_csv['Velocity']
t_tuned = np.linspace(0,1,len(tuned_velocity))
t_new = np.linspace(0,1,20)
tinterp_tuned_velocity = interpolate.interp1d(t_tuned,tuned_velocity,kind = 'cubic', axis=0)(t_new)
flowrate_tuned = tinterp_tuned_velocity
SV_tuned =[]
for k in range(len(t_new)-1):
    SV_tuned.append((flowrate_tuned[k]+flowrate_tuned[k+1])/2 * 1/20)

total_SV_tuned = np.sum(SV_tuned) * 1000 * 1000 # unit: ml/s
tuned_peak = np.argmax(abs(flowrate_tuned)) # used for following shifting
tuned_min = np.argmin(flowrate_tuned) # used for following shifting


## interpolation on systole and diastole

synthetic_planes_ratioScaled = ut.time_interpolation(synthetic_planes[:16],time_intp_options)
SV_synthetic_planes_ratioScaled = dut.compute_flowrate(synthetic_planes_ratioScaled)['Q(t)']
synthetic_peak = np.argmax(SV_synthetic_planes_ratioScaled)
q= deque(np.arange(len(synthetic_planes_ratioScaled)))
circshift = tuned_peak - synthetic_peak
q.rotate(circshift)
synthetic_planes_ratioScaled = [synthetic_planes_ratioScaled[int(k)] for k in q]

SV_syn = dut.compute_flowrate(synthetic_planes_ratioScaled)['Q(t)']
SV_chloe = []
for k in range(len(t_new)-1):
    SV_chloe.append((SV_syn[k] + SV_syn[k+1])/2 * 1/20)
total_SV_syn_chloe = np.sum(SV_chloe)*1000*1000 # unit ml/s
#ratio = total_SV_tuned / total_SV_syn_chloe
ratio = np.max(SV_tuned) / np.max(SV_chloe)


plt.plot(SV_syn * 60000,label='synthetic')
plt.plot(flowrate_tuned * 60000,label='tuned')
plt.ylabel('flowrate L/min')
plt.legend()
plt.show()

## scale
n=[]
for k in range(len(synthetic_planes_ratioScaled)):
    if k !=0 and k!= 19:
        if  (SV_syn[k]>0 and flowrate_tuned[k] > 0) or (SV_syn[k]<0 and flowrate_tuned[k] < 0):
            synthetic_planes_ratioScaled[k]['Velocity'] *= ratio
            n.append(k)
    #synthetic_planes_ratioScaled[k].save(osp.join(outputDir, 'syntheticProf_{:02d}.vtp'.format(k)))

flow = dut.compute_flowrate(synthetic_planes_ratioScaled )['Q(t)']

#plt.plot(SV_syn,label='synthetic')
plt.plot(flowrate_tuned,label='tuned')
plt.plot(flow,label='synthetic scaled')
plt.ylabel('flowrate L/min')
plt.legend()
plt.show()


