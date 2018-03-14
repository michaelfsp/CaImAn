##
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')


try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import caiman as cm
import numpy as np
import scipy.ndimage as ndi
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise

ipyparallel = True
if ipyparallel:
    c,dview,n_processes = cm.cluster.setup_cluster(backend = 'local', n_processes = 40, single_thread = False)
else:
    dview = None

from tqdm import tqdm

##
#-----------------------------------------------------------------------------
#                               Aux. Functions
#-----------------------------------------------------------------------------
def load_inscopix(xml_fname, start_time=0, subindices=None, bottom=0, top=0,
                  left=0, right=0, resize=None):
    '''Load a multi TIFF Inscopix session by concatenation, using the XML metadata file.
    '''
    #TODO: make it possible to load from a single TIF file name instead. Useful
    # for when nVista crashes without producing an XML file
    import xml.etree.ElementTree as ET
    import os.path

    et = ET.parse(xml_fname)
    root = et.getroot()
    attrs, decompressed = root.getchildren()

    attr_list = attrs.getchildren()

    metadata = {}
    for a in attr_list:
        metadata[a.get('name')] = a.text

    files = decompressed.getchildren()
    frames = []
    filenames = []
    dir_name = os.path.dirname(xml_fname)
    for f in files:
        frames.append(f.get('frames'))
        filenames.append(os.path.join(dir_name, f.text))

    return cm.load_movie_chain(filenames, fr=metadata['fps'], start_time=start_time,
                                   meta_data=metadata, subindices=subindices,
                                   #bottom=bottom, top=top, left=left, right=right,
                                   #resize=resize)
                                   bottom=bottom, top=top, left=left, right=right)


def mem_efficient_norm(m):
    norms = np.zeros(m.shape[0])
    for i, f in enumerate(m):
        norms[i] = np.sqrt(np.sum(f.astype(np.int)**2))
    return norms

##
#-----------------------------------------------------------------------------
#                               Parameters
#-----------------------------------------------------------------------------
min_mov = 0

n_bits = 16#for detection of corrupt frames
n_stds = 6#for detection of corrupt frames

#fdir = '/home/michael/data/priority/black1/'
#xml = 'recording_20160717_143955.xml'
#xml = 'recording_20160720_160238.xml'
#
#fdir = '/home/michael/data/priority/green1/'
#xml = 'recording_20160716_172303.xml'
#
fdir = '/media/hdd_data1/michael/priority/black2/'
bottom, top, left, right = (1, 30, 100, 80)
#xml = 'recording_20160716_162425.xml'
xml = 'recording_20160721_182115.xml'

fname = fdir + xml

method = 'rank'

##
# Check where neurons are for the cropping
#
'''
m_max = np.max(m, axis=0)
m_avg = np.mean(m, axis=0)

m_maxmean = m_max - m_avg

pl.figure()
pl.imshow(m_maxmean)
##
'''

##
#-----------------------------------------------------------------------------
#                               Init Variables
#-----------------------------------------------------------------------------
report = []
#
# Green 1 - 20160716_172303
#m = load_inscopix(fname, bottom=10, top=140, left=195, right=185)
# Black 2
#m = load_inscopix(fname, bottom=1, top=30, left=100, right=80)
# Black 1
m = load_inscopix(fname, bottom=bottom, top=top, left=left, right=right)


##
#-----------------------------------------------------------------------------
#                         Compute Baseline Corr. Img.
#-----------------------------------------------------------------------------
old_corr = cm.summary_images.local_correlations(m, False, False)

pl.figure()
pl.imshow(old_corr)
pl.title('Raw data correlation image')

##
#-----------------------------------------------------------------------------
#                            Detect Corrupt Frames
#-----------------------------------------------------------------------------
#TODO: make sure we don't need to crop such that we can detect frames that are
# corrupted on the edges?
if method == 'diff_to_avg_frame':
    max_val = m.max()
    if max_val < 2**(n_bits-1):
        m *= int(np.floor(2**(n_bits-1)/max_val))
    else:
        m //= int(np.ceil(2**(n_bits-1)/max_val))
        # //= for old division, i.e. int/int=int

    if n_bits == 16:
        arr_type = np.int16
    elif n_bits == 8:
        arr_type = np.int8
    m = m.astype(arr_type, copy=False)

    frame_avg = np.mean(m, axis=(1,2)).astype(arr_type)
    m -= frame_avg[:,np.newaxis,np.newaxis]

    rms = mem_efficient_norm(m)
    rms -= rms.mean()
    rms_std = np.std(rms)

    #TODO: test for different movies if using a nuclear norm instead of Frobenius
    # norm would make a difference
    crp = np.nonzero(np.abs(rms > n_stds * rms_std))[0]

    pl.figure()
    pl.plot(rms)


elif method == 'smallest_singular_value':
    ma = np.array(m)# currently necessary due to a bug in CaImAn
    del(m)# save memory

    s = []
    for f in tqdm(ma):
        s.append(np.min(np.linalg.svd(f, compute_uv=False)))

    s_mean = np.mean(s)
    s_std = np.std(s)

    crp = np.nonzero(np.abs(s < s_mean - 4*s_std))[0]

    pl.figure()
    pl.plot(s)


elif method == 'rank':
    crp = []# indices of corrupt frames
    f_shape = m.shape[1:3]
    smallest_len = min(f_shape)
    largest_len = max(f_shape)
    largest_dim = np.argmax(f_shape)

    for i, f in enumerate(tqdm(m)):
        if np.linalg.matrix_rank(f) < smallest_len:
            crp.append(i)
        else:
            if largest_dim == 0:
                f_1 = f[0:largest_len//2, :]
                f_2 = f[largest_len//2:, :]

            elif largest_dim == 1:
                f_1 = f[:, 0:largest_len//2]
                f_2 = f[:, largest_len//2:]

            f_1_rank = np.linalg.matrix_rank(f_1)
            f_2_rank = np.linalg.matrix_rank(f_2)

            if f_1_rank < min(f_1.shape) or f_2_rank < min(f_2.shape):
                crp.append(i)


#TODO: instead of using first this to detect corrupt frames and then the
#difference to the average between frame before and after use already the second
#to detect corrupt frames

##
#-----------------------------------------------------------------------------
#                               Reload Movie
#-----------------------------------------------------------------------------
if method != 'rank':
    m = load_inscopix(fname, bottom=bottom, top=top, left=left, right=right)


##
#-----------------------------------------------------------------------------
#                            Fix Corrupt Frames
#-----------------------------------------------------------------------------
if crp:
    to_fix = np.copy(crp)

    for i in to_fix:
        pl.figure()
        pl.imshow(m[i,:,:])

    if 0 in to_fix:
        m[0,:,:] = m[1,:,:]
        to_fix = to_fix[1:]

    if m.shape[0] in to_fix:
        m[-1,:,:] = m[-2,:,:]
        to_fix = to_fix[:-1]

    to_fix_before = np.copy(to_fix)

    err = m.shape[1]*m.shape[2]
    #rmse = np.linalg.norm(m[to_fix,:,:] - np.mean((m[to_fix-1,:,:],
    #                                               m[to_fix+1,:,:]), axis=0),
    #                      axis=(1,2))
    #while np.any(rmse > err):
    while len(to_fix) > 0:
        print(to_fix)
        for i in to_fix:
    # in case it is the first or last frame there's nothing to interpolate from
    # so we cheat and substitute by the neighbouring frame
            m[i,:,:] = np.mean((m[i-1,:,:], m[i+1,:,:]), axis=0)

        i_still_to_fix = np.nonzero(np.any(np.abs(m[to_fix_before,:,:] -
                                                  np.mean((m[to_fix_before-1,:,:],
                                                   m[to_fix_before+1,:,:]), axis=0)) > 0.5,
                          axis=(1,2)))[0]
        to_fix = to_fix_before[i_still_to_fix]

# we need to add back its neighbours if they were fixed before too

#TODO: code this using linspace. it is a bit hacky as is right now


#TODO: don't let any frame drop until all are fixed
#    if len(i_still_to_fix) == 0:
#        m[i,:,:] = np.mean((m[i-1,:,:], m[i+1,:,:]), axis=0)
#        i_still_to_fix = np.nonzero(np.any(m[to_fix,:,:] - np.mean((m[to_fix-1,:,:],
#                                                   m[to_fix+1,:,:]), axis=0) > 0.5,
#                          axis=(1,2)))[0]
#
    #    rmse = np.linalg.norm(m[to_fix,:,:] - np.mean((m[to_fix-1,:,:],
#                                               m[to_fix+1,:,:]), axis=0),
#                      axis=(1,2))

##
#-----------------------------------------------------------------------------
#                               Save NPY
#-----------------------------------------------------------------------------
#fname_npy = fname.rstrip('.xml') + '_concatenated.npy'
fname_tif = fname.rstrip('.xml') + '_concatenated.tif'

m.save(fname_tif, to32=False, clip_percentiles=None)
#FIXME: change tif name


##
#-----------------------------------------------------------------------------
#                            Compute pre-MC CI
#-----------------------------------------------------------------------------
first_cc = cm.summary_images.local_correlations(m, False, False)

##
#-----------------------------------------------------------------------------
#                            Motion Correction
#-----------------------------------------------------------------------------
shifts_opencv = False
min_mov = 0
max_shifts = (10,10)

max_dev = 0
strides = (64,64)
overlaps = (32,32)
upsample_factor_grid = 4 #4*max_dex

g_sigma_smooth = 1.4
g_sigma_background = 14.

k_std_smooth = 6# kernel size will be kernel_std * sigma
k_std_background = 4

fname_precorr = fname_tif

#fname = '/media/hdd_data1/michael/andreas/test_snippet2.tif'
#fname = '/home/michael/data/sample_bin_512.tif'

#FIXME: delete this line!!

#mcf = cm.motion_correction.MotionCorrect1P(fname, min_mov, max_shifts=(10,10), shifts_opencv=False, dview=dview, niter_rig=3, max_deviation_rigid=max_dev, g_sigma_smooth=g_sigma_smooth, g_sigma_background=g_sigma_background, k_std_smooth=k_std_smooth, k_std_background=k_std_background)

#mcf.motion_correct_rigid(save_movie=True)
#fname_corr = mcf.fname_tot_rig

mcf = cm.motion_correction.MotionCorrect1P(fname_precorr, min_mov=min_mov, max_shifts=max_shifts, shifts_opencv=shifts_opencv, dview=dview, niter_rig=2, splits_rig=40, max_deviation_rigid=max_dev, strides=strides, overlaps=overlaps, upsample_factor_grid=upsample_factor_grid, niter_els=1, num_splits_to_process_els=[None], g_sigma_smooth=g_sigma_smooth, g_sigma_background=g_sigma_background, k_std_smooth=k_std_smooth, k_std_background=k_std_background)

#mcf.motion_correct_pwrigid(save_movie=True, allow_nans=True)
mcf.motion_correct_rigid(save_movie=True, remove_blanks=True)

##
#-----------------------------------------------------------------------------
#                            Save corr. movie
#-----------------------------------------------------------------------------

fname_corr = fname_tif.rstrip('.tif') + '_mcorr_fft.tif'
mcorr = cm.load(mcf.fname_tot_rig)

##
t,y,x=np.nonzero(np.isnan(mcorr))
mcorr = mcorr.crop(crop_top=2, crop_bottom=2, crop_left=2, crop_right=2)
mcorr = mcorr.crop(crop_bottom=2, crop_top=3)

mcorr.save(fname_corr, to32=False, clip_percentiles=None)

#report_pwr.append(cm.motion_correction.compute_metrics_filter(mcf.fname_tot_els, final_x_size, final_y_size, False))

##
##


# kernel sizes have to be odd ints for openCV gaussianBlur
next_odd_int = lambda x: int(np.ceil(x) // 2 * 2 + 1)
k_size_smooth = next_odd_int(k_std_smooth*g_sigma_smooth)
k_size_background = next_odd_int(k_std_background*g_sigma_background)

bound = int(np.ceil(max(k_size_smooth, k_size_background)/2))

m.gaussian_blur_2D(*2*[k_size_smooth], *2*[g_sigma_smooth])

m += int(m.max())

kernel = cv2.getGaussianKernel(k_size_background, g_sigma_background)
kernel = kernel*kernel.transpose()

center_i = int(k_size_background/2)-1
kernel[center_i, center_i] = 0
kernel /= np.sum(kernel)
kernel = -kernel
kernel[center_i, center_i] = 1.# this is equivalent to adding a delta fun.

m.filter_2D(kernel)
m = m.crop(bound, bound, bound, bound)

##
new_corr = cm.summary_images.local_correlations(m, False, False)

pl.figure()
pl.imshow(first_cc)
pl.title('Original')
pl.figure()
pl.imshow(new_corr)
pl.title('After filtering')

##
fname_filt = '/home/michael/data/filtered_movie.tif'
m.save(fname_filt)

c,dview,n_processes = cm.cluster.setup_cluster(backend = 'local',n_processes = 20, single_thread = False)

##
# Compute metrics
final_x_size = m.shape[1]
final_y_size = m.shape[2]

metrics_b = cm.motion_correction.compute_metrics_motion_correction(fname_corr,
                                                       final_x_size,
                                                       final_y_size, False)


template_b, correlations_b, flows_b, norms_b, smoothness_b, img_corr_b, smoothness_corr_b = metrics_b

#flow_rms_b = np.sqrt(np.mean([i**2 for i in norms_b]))
##
#print(('batch\t',smoothness_b, smoothness_corr_b, '\n'))
report = ['batch\t',smoothness_b, smoothness_corr_b, '\n'] + report


#m = load_inscopix(fname_xml, bottom=25, top=150, left=220, right=174,
#                  resize=(0.25, 0.25))
#m = m.crop(bound, bound, bound, bound)

m = cm.load(fname_tif) #reload movie

max_h,max_w = np.ceil(np.max(mcf.shifts_rig,axis=0)).astype('int')
min_h,min_w = np.floor(np.min(mcf.shifts_rig,axis=0)).astype('int')

m = m.crop(crop_top=max_h, crop_bottom=-min_h, crop_left=max_w
               ,crop_right=-min_w, crop_begin=0,crop_end=0)

fname_or_crop = 'original_crop.npz'
m.save(fname_or_crop)

final_x_size = m.shape[1]
final_y_size = m.shape[2]

metrics_o = cm.motion_correction.compute_metrics_motion_correction(fname_or_crop,
                                                       final_x_size,
                                                       final_y_size, False)

template_o, correlations_o, flows_o, norms_o, smoothness_o, img_corr_o, smoothness_corr_o = metrics_o

report = ['original\t',smoothness_o, smoothness_corr_o, '\n'] + report

print(*report)

pl.show()
