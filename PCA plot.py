
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



## ---> SETTINGS
preprocDir = r'D:/InletProfileStudy/SSM/Output_kaihong/Patient_5_new700/'
figureDir = osp.join(preprocDir, 'PCA figures')

os.makedirs(figureDir, exist_ok=True)

#synthOutDir = osp.join(preprocDir, 'sampled_profiles')
#meanProfDir = osp.join(preprocDir, 'mean_profile_SV')
#os.makedirs(synthOutDir, exist_ok=True)
#os.makedirs(meanProfDir, exist_ok=True)


V = pd.read_csv(osp.join(preprocDir, 'matrixV.csv')).to_numpy()
n_pat, n_frames, n_nodes = np.shape(V)[0], 20, np.shape(V)[1]//20//3



#-----------------------------------------------------------------------------------------------------------------------
## ---> PRINCIPAL COMPONENT ANALYSIS
print('Performing PCA...')


# 2. Compute individual and cumulative variance
pca = PCA(n_components=32)                                 # total n. of components has to be equal to min(n_samples, n_variables)
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
plt.savefig(osp.join(figureDir,'culmulative variance.png'))


# Display the plot
plt.show()
