


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
from matplotlib import colormaps



## ---> SETTINGS
preprocDir = r'D:/InletProfileStudy/SSM/Output_2024/SH_P5/'
synthOutDir = osp.join(preprocDir, 'sampled_profiles')
Fig_Dir = osp.join(preprocDir, 'Figure')
os.makedirs(synthOutDir, exist_ok=True)
os.makedirs(Fig_Dir, exist_ok=True)


V = pd.read_csv(osp.join(preprocDir, 'matrixV.csv')).to_numpy()
n_pat, n_frames, n_nodes = np.shape(V)[0], 20, np.shape(V)[1]//20//3

mean_prof = V.mean(0).reshape(n_frames, n_nodes, 3)
mean_planes = [pv.read(fn) for fn in sorted(glob(osp.join(preprocDir, 'mean_profile', '*.vtp')))]

#-----------------------------------------------------------------------------------------------------------------------
## ---> PRINCIPAL COMPONENT ANALYSIS
print('Performing PCA...')


# 2. Compute individual and cumulative variance
pca = PCA(n_components=33)                                 # total n. of components has to be equal to min(n_samples, n_variables)
pca.fit(V)
var_components = pca.explained_variance_ratio_                # individual variance assciated to each component (mode)
var = np.sum(pca.explained_variance_ratio_[:26])              # total variance

cum_explained_var = []                                        # cumulative variance: # sum of invidual variances for subsequent components up to 10th mode.

for i in range(0, len(pca.explained_variance_ratio_)):
    if i == 0:
        cum_explained_var.append(pca.explained_variance_ratio_[i])
    else:
        cum_explained_var.append(pca.explained_variance_ratio_[i] + cum_explained_var[i-1])
cum_explained_var = np.asarray(cum_explained_var)


# Create figure and axes
fig, ax1 = plt.subplots()

# Plot individual values on the left y-axis
ax1.bar(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color='b', alpha=0.5, label='Individual Values')
ax1.set_ylabel('Individual Variance (%)', color='b')
ax1.set_xlabel('Principal componants', color='k')
ax1.tick_params('y', colors='b')

# Create second y-axis on the right
ax2 = ax1.twinx()

# Plot cumulative values as bar chart on the right y-axis
ax2.plot(np.arange(len(pca.explained_variance_ratio_)), cum_explained_var, color='r', alpha=0.5, label='Cumulative Values')
ax2.set_ylabel('Cumulative Variance (%)', color='r')
ax2.tick_params('y', colors='r')


plt.xlabel('Principal componants')


# Add legend and title
ax1.legend(loc='upper right')

#save the figure
plt.savefig(os.path.join(Fig_Dir, 'cumulative_variance.png'))
# plt.savefig('D:/InletProfileStudy/SSM/figures/cumuVariance_kaihong.png')


# Display the plot
plt.show()



# 3. Deform mean shape towards max/min of specific mode (es. mean shape +_ 3*var[mode1])


pc = pca.components_.T
variance = pca.explained_variance_
b_range = np.arange(-1.5,2,0.5)
total_flowrates = np.empty((5,20,b_range.shape[0]))
total_flowrate = np.zeros((n_frames, b_range.shape[0]))
meanPlane_flowrate = dut.compute_flowrate(mean_planes)['Q(t)'] * 60000# calculate flowrates on the mean plane, and unit change to L/min
SV_meanPlane = np.sum(meanPlane_flowrate)
#mean_planes_SV = [mean_planes[0].copy() for _ in range(n_frames)]
#ratio_SV = 113 / SV_meanPlane
ratio_SV = 1


# Define colormap
cmap = colormaps.get_cmap('Greys')
norm = plt.Normalize(vmin=0, vmax=len(b_range) - 1)

### plot flow waveform of each mode #######
for which_mode in range(0, 5): #first 5 modes
    fig,ax = plt.subplots()
    #ratio_SV = 113 / SV_meanPlane
    for count, how_much_std in enumerate(b_range, start=0):
        pc_i = pc[:, which_mode].reshape((n_frames, n_nodes, 3))
        std = np.sqrt(variance[which_mode])

        outDirSynth = osp.join(synthOutDir, 'prof_mode{}_coeff_{}_'.format(which_mode, how_much_std))
        os.makedirs(outDirSynth, exist_ok=True)

        gen_planes = [mean_planes[0].copy() for _ in range(n_frames)]

        for k in range(n_frames):
            gen_planes[k]['Velocity'] = (mean_prof[k] + (how_much_std * std * pc_i[k])) * ratio_SV  # all modes are scaled to match the SV
            gen_planes[k].save(osp.join(outDirSynth,
                    'prof_mode{}_coeff_{}_frame{:02d}.vtp'.format(which_mode, how_much_std, k)))    # {:02d} padded with 0 and has 2 digitals.
        flowRate = dut.compute_flowrate(gen_planes)['Q(t)'] * 60000  # unit convert to L/min

        total_flowrate[:,count] = flowRate
    x_value = np.linspace(0,0.7,20)   # time
    for variation in range(total_flowrate.shape[1]):
        color = cmap(norm(variation))
        ax.plot(x_value, total_flowrate[:, variation] , label=f"b={variation * 0.5 - 3}", color=color)
        ax.set_xlabel('Normalized Time (s)')
        ax.set_ylabel('Flowrate (L/min)')
        ax.set_title(f"Mode {which_mode}")
        ax.legend()
        
    ### Save mean profiles scaled to match the SV
    ax.plot(x_value, meanPlane_flowrate * ratio_SV, label='mean plane', color='red')
    plt.savefig(osp.join(Fig_Dir, 'meanProf_mode{}.png'.format(which_mode)))
plt.show()










