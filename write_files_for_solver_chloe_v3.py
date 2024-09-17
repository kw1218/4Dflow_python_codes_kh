import sys
import os
import os.path as osp
import numpy as np
from numpy.random import default_rng
import pyvista as pv
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.interpolate import interp1d


#-----------------------------------------------------------------------------------------------------------------------
## Options
profilesDir = r'D:\InletProfileStudy\SSM\Output_2024\SH_P5\synthetic_cohort_first13modes\089'
outputDir = r'D:\InletProfileStudy\SolverOutput\Shared_csv\Syn_root_89'
saveName = 'syn_root_89'
cfd_delta_t = 0.001  # simulation time steps
cardiac_cycle_period = 0.858
time_interpolation = 'cubic' # can be linear, nearest, quadratic, ...
solver = 'cfx_xyz' # can be star, cfx, cfx_xyz or fluent
# samplerate = 0.2 # 1 = all points used, set to < 1 to lower the number of points sampled when writing cfx files - there is a file size limit of ~40 MB per .csv file

#-----------------------------------------------------------------------------------------------------------------------
## Prepare variables
interp_planes = [pv.read(fn) for fn in sorted(glob(osp.join(profilesDir, '*.vtp')))]
num_frames = len(interp_planes)

os.makedirs(outputDir, exist_ok=True)

# tcfd = np.arange(0, cardiac_cycle_period, cfd_delta_t)
tcfd = np.arange(cfd_delta_t, (cardiac_cycle_period + cfd_delta_t), cfd_delta_t)
timepoints = len(tcfd)
t4df = np.linspace(0, cardiac_cycle_period, num_frames)
pos = interp_planes[0].points
npts = pos.shape[0]
vel4df = np.array([interp_planes[k]['Velocity'] for k in range(len(interp_planes))]) #change to lower case v in 'velocity' if using vtk files previously created in matlab
velcfd = interp1d(t4df, vel4df, axis=0, kind=time_interpolation)(tcfd)

#-----------------------------------------------------------------------------------------------------------------------
## the next few lines creates the same info but at a lower sample rate to reduce file to a size that can be opened in excel/cfx
maxpts = 1048566 #maximum no points in excel
samplerate = min((maxpts/(cardiac_cycle_period/cfd_delta_t))/npts,1) #take min of calcuation and 1, if total number of points would not cause memory issue then sample rate can = 1
rng = default_rng()
npts_sampled = int(npts*samplerate)
sample = rng.choice(npts, size=(npts-npts_sampled), replace=False)
pos_sampled = np.delete(pos, sample, axis=0)
for i in range(vel4df.shape[0]):
    vel4df_sampled = np.delete(vel4df, sample, axis=1)
velcfd_sampled = interp1d(t4df, vel4df_sampled, axis=0, kind=time_interpolation)(tcfd)

#----------------------------------------------------------------------------------------------------------------------
## Write files for solver

if solver == 'star':
    # write .csv for star-ccm+
    with open(osp.join(outputDir, saveName + '.csv'), 'w') as fn:
        riga = 'X,Y,Z'
        for j in range(len(tcfd)):
            riga += ',u(m/s)[t={}s],v(m/s)[t={}s],w(m/s)[t={}s]'.format(tcfd[j], tcfd[j], tcfd[j])
        riga += '\n'
        fn.write(riga)
        for i in tqdm(range(len(pos))):
            riga = '{},{},{}'.format(pos[i, 0], pos[i, 1], pos[i, 2])
            for j in range(len(tcfd)):
                riga += ',{},{},{}'.format(velcfd[j, i, 0], velcfd[j, i, 1], velcfd[j, i, 2])
            riga += '\n'
            fn.write(riga)

if solver == 'cfx':
    cfx_arr = pos_sampled[:,:2] #taking x and y from position table
    cfx_arr = np.tile(cfx_arr,reps=(timepoints,1)) #repeat whole array of numbers x times (not repeating individual elements)

    time_stem = np.repeat(tcfd,npts_sampled,axis=0) # repeat individual elements of tcfd
    time_stem = time_stem.reshape(-1,1) # reshape to one dimension ahead of concatenation

    cfx_arr = np.concatenate((cfx_arr, time_stem),axis=1) # combine position and time arrays

    df_list = []
    assert velcfd_sampled.shape[2] == 3
    #for i in range(velcfd_sampled.shape[2]):
    for i, direction in enumerate(['rl', 'ap', 'fh']): #CHECK ORDER OF DIRECTIONS ARE CORRECT FOR INDIVIDUAL PATIENT #og was rl, ap, fh
        #creating dataframe with required 7 lines of text for cfx to read the file
        df_header = pd.DataFrame(
            {
                '1': ['[Name]','InletV' + direction, None, '[Spatial Fields]', 'x', None, '[Data]', 'x[m]'],
                '2': [None, None, None, None, 'y', None, None, 'y[m]'],
                '3': [None, None, None, None, 't', None, None, 't[s]'],
                '4': [None, None, None, None, None, None, None, 'Velocity[m s^-1]']
            }
        )
        vel_arr = velcfd_sampled[:,:,i]
        vel_arr = vel_arr.reshape(-1,1)
        output_arr = np.concatenate((cfx_arr, vel_arr), axis=1)
        output_df = pd.DataFrame(output_arr,columns=['1','2','3','4'])
        output_df = pd.concat((df_header,output_df), axis=0)
        output_df.to_csv(outputDir + '/vel' + direction.upper() + '.csv',index=False,header=False)

if solver == 'fluent':
    # write .prof for ansys fluent
    xx, yy, zz = pos[:, 0].tolist(), pos[:, 1].tolist(), pos[:, 2].tolist()
    fu = np.swapaxes(velcfd[:, :, 0], 0, 1)
    fv = np.swapaxes(velcfd[:, :, 1], 0, 1)
    fw = np.swapaxes(velcfd[:, :, 2], 0, 1)
    for i in tqdm(range(len(tcfd))):
        with open(osp.join(outputDir, saveName + '_{:05d}.prof'.format(i)), 'w') as fn:
            fn.write('((velocity point {})\n'.format(npts))
            fn.write('(x\n')
            for xi in xx:
                fn.write(str(xi) + '\n')
            fn.write(')\n')
            fn.write('(y\n')
            for yi in yy:
                fn.write(str(yi) + '\n')
            fn.write(')\n')
            fn.write('(z\n')
            for zi in zz:
                fn.write(str(zi) + '\n')
            fn.write(')\n')
            fn.write('(u\n')
            for ui in fu[:, i]:
                fn.write(str(ui) + '\n')
            fn.write(')\n')
            fn.write('(v\n')
            for vi in fv[:, i]:
                fn.write(str(vi) + '\n')
            fn.write(')\n')
            fn.write('(w\n')
            for wi in fw[:, i]:
                fn.write(str(wi) + '\n')
            fn.write(')\n')
            fn.write(')')

if solver == 'cfx_xyz':
    cfx_arr = pos_sampled[:,:3] #taking x and y and z from position table
    cfx_arr = np.tile(cfx_arr,reps=(timepoints,1)) #repeat whole array of numbers x times (not repeating individual elements)

    time_stem = np.repeat(tcfd,npts_sampled,axis=0) # repeat individual elements of tcfd
    time_stem = time_stem.reshape(-1,1) # reshape to one dimension ahead of concatenation

    cfx_arr = np.concatenate((cfx_arr, time_stem),axis=1) # combine position and time arrays

    df_list = []
    assert velcfd_sampled.shape[2] == 3
    #for i in range(velcfd_sampled.shape[2]):
    for i, direction in enumerate(['rl', 'ap', 'fh']): #CHECK ORDER OF DIRECTIONS ARE CORRECT FOR INDIVIDUAL PATIENT #og was rl, ap, fh
        #creating dataframe with required 7 lines of text for cfx to read the file
        df_header = pd.DataFrame(
            {
                '1': ['[Name]','InletV' + direction, None, '[Spatial Fields]', 'x', None, '[Data]', 'x[m]'],
                '2': [None, None, None, None, 'y', None, None, 'y[m]'],
                '3': [None, None, None, None, 'z', None, None, 'z[m]'],
                '4': [None, None, None, None, 't', None, None, 't[s]'],
                '5': [None, None, None, None, None, None, None, 'Velocity[m s^-1]']
            }
        )
        vel_arr = velcfd_sampled[:,:,i]
        vel_arr = vel_arr.reshape(-1,1)
        output_arr = np.concatenate((cfx_arr, vel_arr), axis=1)
        output_df = pd.DataFrame(output_arr,columns=['1','2','3','4','5'])
        output_df = pd.concat((df_header,output_df), axis=0)
        output_df.to_csv(outputDir + '/vel' + direction.upper() + '.csv',index=False,header=False)

print("files written.")