#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:52:23 2017
Basic demo for the OnACID algorithm using CNMF initialization. For a more
complete demo check the script demo_OnACID_mesoscope.py

@author: jfriedrich & epnev
"""

import numpy as np
import pylab as pl
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.visualization import view_patches_bar, plot_contours
from copy import deepcopy
from scipy.special import log_ndtr

#%% load data

fname = './example_movies/demoMovie.tif'
Y = cm.load(fname).astype(np.float32)                   #
# used as a background image
Cn = cm.local_correlations(Y.transpose(1, 2, 0))

#%% set up some parameters

# frame rate (Hz)
fr = 10
# approximate length of transient event in seconds
decay_time = 0.5
# expected half size of neurons
gSig = [6, 6]
# order of AR indicator dynamics
p = 1
# minimum SNR for accepting new components
min_SNR = 3.5
# correlation threshold for new component inclusion
rval_thr = 0.90
# number of background components
gnb = 3

# set up some additional supporting parameters needed for the algorithm (these are default values but change according to dataset characteristics)

# number of shapes to be updated each time (put this to a finite small value to increase speed)
max_comp_update_shape = np.inf
# maximum number of expected components used for memory pre-allocation (exaggerate here)
expected_comps = 50
# number of timesteps to consider when testing new neuron candidates
N_samples = np.ceil(fr * decay_time)
# exceptionality threshold
thresh_fitness_raw = log_ndtr(-min_SNR) * N_samples
# total length of file
T1 = Y.shape[0]

# set up CNMF initialization parameters

# merging threshold, max correlation allowed
merge_thresh = 0.8
# number of frames for initialization (presumably from the first file)
initbatch = 400
# size of patch
patch_size = 32
# amount of overlap between patches
stride = 3
# max number of components in each patch
K = 4

#%% obtain initial batch file used for initialization

# memory map file (not needed)
fname_new = Y[:initbatch].save('demo.mmap', order='C')
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Cn_init = cm.local_correlations(np.reshape(Yr, dims + (T,), order='F'))


#%% RUN (offline) CNMF algorithm on the initial batch
pl.close('all')
cnm_init = cnmf.CNMF(2, k=K, gSig=gSig, merge_thresh=merge_thresh,
                     p=p, rf=patch_size // 2, stride=stride, skip_refinement=False,
                     normalize_init=False, options_local_NMF=None,
                     minibatch_shape=100, minibatch_suff_stat=5,
                     update_num_comps=True, rval_thr=rval_thr,
                     thresh_fitness_delta=-50, gnb=gnb,
                     thresh_fitness_raw=thresh_fitness_raw,
                     batch_update_suff_stat=True, max_comp_update_shape=max_comp_update_shape)

cnm_init = cnm_init.fit(images)

print(('Number of components:' + str(cnm_init.A.shape[-1])))

pl.figure()
crd = plot_contours(cnm_init.A.tocsc(), Cn_init, thr=0.9)


#%% run (online) OnACID algorithm

cnm = deepcopy(cnm_init)
cnm._prepare_object(np.asarray(Yr), T1, expected_comps)
t = cnm.initbatch

Y_ = cm.load(fname)[initbatch:].astype(np.float32)
for frame_count, frame in enumerate(Y_):
    cnm.fit_next(t, frame.copy().reshape(-1, order='F'))
    t += 1

#%% extract the results

C, f = cnm.C_on[cnm.gnb:cnm.M], cnm.C_on[:cnm.gnb]
A, b = cnm.Ab[:, cnm.gnb:cnm.M], cnm.Ab[:, :cnm.gnb]
print(('Number of components:' + str(A.shape[-1])))

#%% pass through the CNN classifier with a low threshold (keeps clearer neuron shapes and excludes processes)
use_CNN = True
if use_CNN:
    # threshold for CNN classifier
    thresh_cnn = 0.1
    from caiman.components_evaluation import evaluate_components_CNN
    predictions, final_crops = evaluate_components_CNN(
        A, dims, gSig, model_name='use_cases/CaImAnpaper/cnn_model')
    A_exclude, C_exclude = A[:, predictions[:, 1] <
                             thresh_cnn], C[predictions[:, 1] < thresh_cnn]
    A, C = A[:, predictions[:, 1] >=
             thresh_cnn], C[predictions[:, 1] >= thresh_cnn]
    noisyC = cnm.noisyC[cnm.gnb:cnm.M]
    YrA = noisyC[predictions[:, 1] >= thresh_cnn] - C
else:
    YrA = cnm.noisyC[cnm.gnb:cnm.M] - C

#%% plot results
pl.figure()
crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)

view_patches_bar(Yr, A, C, b, f,
                 dims[0], dims[1], YrA, img=Cn)
