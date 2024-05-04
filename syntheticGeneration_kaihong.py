

import sys
import os
import os.path as osp
import numpy as np
import pandas as pd
import pyvista as pv
from glob import glob
from tqdm import tqdm
import shutil
from sklearn.decomposition import PCA
import random
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import descriptors_utils as dut


dataDir = r'D:/InletProfileStudy/SSM/Output_Kaihong/Patient_5_new700/'
outDir = osp.join(dataDir, 'synthetic_cohort_first5modes')

# read computed descriptors for real dataset
real_descriptors = [np.load(fn, allow_pickle=True).tolist() for fn in sorted(glob(osp.join(dataDir, 'real_descriptors', '*.npy')))]
real_ds = []
ks1 = ['ppv_mean', 'fdi_mean', 'fja_mean', 'sfd_mean', 'hfi_mean', 'rfi']
for i in range(len(real_descriptors)):
    nd = {k: real_descriptors[i][k] for k in ks1}
    nd['type'] = 'Real'
    #nd['sfd_mean'] *= 100
    real_ds.append(nd)

# read mean profile
mean_profs = [pv.read(fn) for fn in sorted(glob(osp.join(dataDir, 'mean_profile', '*.vtp')))]

# read matrix V
V = pd.read_csv(osp.join(dataDir, 'matrixV.csv')).to_numpy()
V_mean = np.mean(V, 0)

n_frames = len(mean_profs)
n_nodes = mean_profs[0].points.shape[0]


# ----- Principal Component Analysis
pca = PCA(n_components=18)
pca.fit(V)
var_components = pca.explained_variance_ratio_      # Individual variance associated to single mode
var = np.sum(pca.explained_variance_ratio_[:18])    # Cumulative variance

pc = pca.components_.T
variance = pca.explained_variance_
a = pc
lam = variance


# -------- Shape Sampling: synthetic dataset generation
dict_list_mean = lambda x, k: sum(d[k] for d in x) / len(x)
def dict_list_std(x, k):
    v = [d[k] for d in x]
    return np.std(v)
'''
ks = ['ppv_mean', 'fdm_mean', 'fdi_mean', 'fja_mean', 'sfd_mean', 'rfi']
crits = []
for k in ks:
    mu_d = dict_list_mean(real_descriptors, k)
    std_d = dict_list_std(real_descriptors, k)
    print('{} -- {:.2f}+-{:.2f}'.format(k, mu_d, 2*std_d))
'''

M = 5  # number of nodes selected
valid_count = 0
synth_ds = []
for i in tqdm(range(200)):
    variation = 0
    for m in range(M):  # change for number of modes
        c = random.uniform(-1.5, 1.5)   # a random floating number
        variation += c * np.sqrt(lam[m]) * a[:, m]    # lam is the variance; a is the componant
    U = (V_mean + variation).reshape((n_frames, n_nodes, 3))
    new_profs = [mean_profs[0].copy() for _ in range(n_frames)]
    for k in range(len(new_profs)):
        new_profs[k]['Velocity'] = U[k]

    # acceptance criteria
    synth_descriptors = dut.compute_flow_descriptors(new_profs)
    ks = ['ppv_mean', 'fdi_mean', 'fja_mean', 'sfd_mean', 'hfi_mean', 'rfi']
    #ks = ['ppv_systole', 'fdm_systole', 'fdi_systole', 'fja_systole', 'sfd_systole', 'rfi']
    crits = []
    for k in ks:
        I_p = [dict_list_mean(real_descriptors, k) - 2*dict_list_std(real_descriptors, k),
                 dict_list_mean(real_descriptors, k) + 2*dict_list_std(real_descriptors, k)]
        crits.append(synth_descriptors[k] > I_p[0] and synth_descriptors[k] < I_p[1])
    if all(crits):
        valid_count += 1
        #print('valid count =', valid_count)
        nd = {k: synth_descriptors[k] for k in ks1}
        nd['type'] = 'Synthetic'
        synth_ds.append(nd)

        synthOutDir = osp.join(outDir, '{:03d}'.format(valid_count))
        os.makedirs(synthOutDir, exist_ok=True)
        for k in range(len(new_profs)):
            new_profs[k].save(osp.join(synthOutDir, '{:03d}_{:02d}.vtp'.format(valid_count, k)))


print('valid_count', valid_count)
all_ds = real_ds + synth_ds
df = pd.DataFrame(all_ds)


## STATISTICS
from scipy import stats

alphaLevel = 0.05

fn = osp.join('D:/InletProfileStudy/SSM/Output_Kaihong/', 'statistics.json')
if osp.exists(fn):
     os.remove(fn)

metric  = 'sfd_mean'
for metric in ks1:

    metricReal = [real_ds[i][metric] for i in range(len(real_ds))]
    metricSynth = [synth_ds[i][metric] for i in range(len(synth_ds))]

    s1, p1 = stats.shapiro(metricReal)
    s2, p2 = stats.shapiro(metricSynth)
    if p1 < alphaLevel or p2 < alphaLevel: #non-normal distribution
        print('non normal')
        normal = False
        st, pval = stats.mannwhitneyu(metricReal, metricSynth)
    else: #normal distribution
        print('normal')
        normal = True
        st, pval = stats.ttest_ind(metricReal, metricSynth)


    if normal:
        print('real: {:.2f} ± {:.2f}'.format(np.mean(metricReal), np.std(metricReal)))
        print('synthetic: {:.2f} ± {:.2f}'.format(np.mean(metricSynth), np.std(metricSynth)))
    else:
        print('real: {:.2f} [{:.2f}, {:.2f}]'.format(np.median(metricReal), np.min(metricReal), np.max(metricReal)))
        print('synthetic: {:.2f} [{:.2f}, {:.2f}]'.format(np.median(metricSynth), np.min(metricSynth), np.max(metricSynth)))
    print('p-value: {:.3f}'.format(pval))

    with open(fn, "a") as fo:
        fo.write('Descriptor: {}\n'.format(metric))
        fo.write('\tNormal: {}\n'.format(normal))
        if normal:
            fo.write('\treal: {:.2f} ± {:.2f}\n'.format(np.mean(metricReal), np.std(metricReal)))
            fo.write('\tsynthetic: {:.2f} ± {:.2f}\n'.format(np.mean(metricSynth), np.std(metricSynth)))
        else:
            fo.write('\treal: {:.2f} [{:.2f}, {:.2f}]\n'.format(np.median(metricReal), np.min(metricReal), np.max(metricReal)))
            fo.write('\tsynthetic: {:.2f} [{:.2f}, {:.2f}]\n'.format(np.median(metricSynth), np.min(metricSynth),
                                                              np.max(metricSynth)))
        fo.write('\tp-value: {:.3f}\n\n'.format(pval))


##
sns.set_palette("gray", 2)
fs = 11
fig, ax = plt.subplots(1, 5)

g1 = sns.violinplot(x="type", y="ppv_mean", data=df, ax=ax[0])
g1.set_title('$PPV_{mean} [m/s]$', fontsize=fs)
g1.set(xlabel=None)
g1.set(ylabel=None)
"""
g2 = sns.violinplot(x="type", y="fdm_mean", data=df, ax=ax[0, 1])
g2.set_title('$FDM_{mean} [\%]$', fontsize=fs)
g2.set(xlabel=None)
g2.set(ylabel=None)
"""
g3 = sns.violinplot(x="type", y="fdi_mean", data=df, ax=ax[1])
g3.set_title('$FDI_{mean} [\%]$', fontsize=fs)
g3.set(xlabel=None)
g3.set(ylabel=None)

g4 = sns.violinplot(x="type", y="fja_mean", data=df, ax=ax[2])
g4.set_title('$FJA_{mean} [\%]$', fontsize=fs)
g4.set(xlabel=None)
g4.set(ylabel=None)

g5 = sns.violinplot(x="type", y="sfd_mean", data=df, ax=ax[3])
g5.set_title('$SDF_{mean} [\%]$', fontsize=fs)
g5.set(xlabel=None)
g5.set(ylabel=None)

g5 = sns.violinplot(x="type", y="rfi", data=df, ax=ax[4])
g5.set_title('$RFI [\%]$', fontsize=fs)
g5.set(xlabel=None)
g5.set(ylabel=None)

#fig.set_size_inches(10.5, 6)
fig.tight_layout()
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.65)
#plt.show()
plt.savefig(osp.join('D:/InletProfileStudy/SSM/figures', 'violinplot2_Kaihong.png'), dpi=600)




##
#sns.set(font_scale = 2)
sns.set_palette("gray", 2)
fs = 6
fig, ax = plt.subplots(1, 5)


g1 = sns.boxplot(x="type", y="ppv_mean", data=df, ax=ax[0])
g1.set_title('PPV', fontsize=fs, fontweight='bold')
g1.set(xlabel=None)
#g1.set(ylabel=None)
#g1.xlabel('Normalized time', fontsize=18, fontweight='bold')
#g1.set_yticks(np.linspace(0.3, 0.9, 4), fontsize=22, fontweight='bold')
g1.set_ylabel('[m/s]', fontsize=fs)
ax[0].yaxis.set_tick_params(labelsize = fs)
g1.set_xticklabels(['Clinical', 'Synthetic'], fontsize=fs, fontweight='bold')
#g1.set_yticks(g1.get_yticks(), fontsize = fs)

'''
g2 = sns.boxplot(x="type", y="fdm_mean", data=df, ax=ax[1])
g2.set_title('$FDM_{mean} [\%]$', fontsize=fs)
g2.set(xlabel=None)
g2.set(ylabel=None)
'''
g3 = sns.boxplot(x="type", y="fdi_mean", data=df, ax=ax[1])
g3.set_title('FDI', fontsize=fs, fontweight='bold')
g3.set(xlabel=None)
#g3.set(ylabel=None)
ax[1].yaxis.set_tick_params(labelsize = fs)
g3.set_ylabel('[%]', fontsize=fs)
g3.set_xticklabels(['Clinical', 'Synthetic'], fontsize=fs, fontweight='bold')

g4 = sns.boxplot(x="type", y="fja_mean", data=df, ax=ax[2])
g4.set_title('FJA', fontsize=fs, fontweight='bold')
g4.set(xlabel=None)
#g4.set(ylabel=None)
g4.set_ylabel('[%]', fontsize=fs)
g4.set_xticklabels(['Clinical', 'Synthetic'], fontsize=fs, fontweight='bold')
ax[2].yaxis.set_tick_params(labelsize = fs)


g5 = sns.boxplot(x="type", y="sfd_mean", data=df, ax=ax[3])
g5.set_title('SFD', fontsize=fs, fontweight='bold')
g5.set(xlabel=None)
#g5.set(ylabel=None)
g5.set_ylabel('[-]', fontsize=fs)
ax[3].yaxis.set_tick_params(labelsize = fs)
g5.set_xticklabels(['Clinical', 'Synthetic'], fontsize=fs, fontweight='bold')


g6 = sns.boxplot(x="type", y="rfi", data=df, ax=ax[4])
g6.set_title('RFI', fontsize=fs, fontweight='bold')
g6.set(xlabel=None)
#g6.set(ylabel=None)
g6.set_ylabel('[%]', fontsize=fs)
ax[4].yaxis.set_tick_params(labelsize = fs)
g6.set_xticklabels(['Clinical', 'Synthetic'], fontsize=fs, fontweight='bold')


fig.set_size_inches(15, 4)
fig.tight_layout()
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.65)
#plt.show()
plt.savefig(osp.join('D:/InletProfileStudy/SSM/figures', 'boxplot2_kaihong.png'), dpi=600)







