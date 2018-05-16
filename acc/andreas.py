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

parallel = True
if parallel:
    if 'dview' in locals():
        dview.terminate()
    c, dview, n_processes = cm.cluster.setup_cluster(backend = 'local', n_processes = None, single_thread = False)
    import mkl
    mkl.set_num_threads(4)
else:
    import mkl
    mkl.set_num_threads(48)
    dview = None

from tqdm import tqdm

import glob, os

def sgolay2d ( z, window_size, order, derivative = None ):
    """
    """
    import scipy.signal
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2 ) / 2.0

    if  window_size % 2 == 0:
        raise ValueError( 'window_size must be odd' )

    if window_size ** 2 < n_terms:
        raise ValueError( 'order is too high for the window size' )

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ ( k - n, n ) for k in range( order + 1 ) for n in range( k + 1 ) ]

    # coordinates of points
    ind = np.arange( -half_size, half_size + 1, dtype = np.float64 )
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1] ).reshape( window_size ** 2, )

    # build matrix of system of equation
    A = np.empty( ( window_size ** 2, len( exps ) ) )
    for i, exp in enumerate( exps ):
        A[:, i] = ( dx ** exp[0] ) * ( dy ** exp[1] )

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros( ( new_shape ) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs( np.flipud( z[1:half_size + 1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs( np.flipud( z[-half_size - 1:-1, :] ) - band )
    # left band
    band = np.tile( z[:, 0].reshape( -1, 1 ), [1, half_size] )
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size + 1] ) - band )
    # right band
    band = np.tile( z[:, -1].reshape( -1, 1 ), [1, half_size] )
    Z[half_size:-half_size, -half_size:] = band + np.abs( np.fliplr( z[:, -half_size - 1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs( np.flipud( np.fliplr( z[1:half_size + 1, 1:half_size + 1] ) ) - band )
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs( np.flipud( np.fliplr( z[-half_size - 1:-1, -half_size - 1:-1] ) ) - band )

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs( np.flipud( Z[half_size + 1:2 * half_size + 1, -half_size:] ) - band )
    # bottom left corner
    band = Z[-half_size:, half_size].reshape( -1, 1 )
    Z[-half_size:, :half_size] = band - np.abs( np.fliplr( Z[-half_size:, half_size + 1:2 * half_size + 1] ) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv( A )[0].reshape( ( window_size, -1 ) )
        return scipy.signal.fftconvolve( Z, m, mode = 'valid' )
    elif derivative == 'col':
        c = np.linalg.pinv( A )[1].reshape( ( window_size, -1 ) )
        return scipy.signal.fftconvolve( Z, -c, mode = 'valid' )
    elif derivative == 'row':
        r = np.linalg.pinv( A )[2].reshape( ( window_size, -1 ) )
        return scipy.signal.fftconvolve( Z, -r, mode = 'valid' )
    elif derivative == 'both':
        c = np.linalg.pinv( A )[1].reshape( ( window_size, -1 ) )
        r = np.linalg.pinv( A )[2].reshape( ( window_size, -1 ) )
        #import ipdb; ipdb.set_trace()
        #return Z, scipy.signal.fftconvolve( Z, -r, mode = 'valid' ), scipy.signal.fftconvolve( Z, -c, mode = 'valid' )
        #return scipy.signal.filtfilt( Z, -r), scipy.signal.filtfilt( Z, -c)
        #return scipy.ndimage.filters.convolve(z, -r, mode = 'nearest'), scipy.ndimage.filters.convolve(z, -c, mode = 'nearest')
        return scipy.signal.fftconvolve( z, -r, mode = 'valid' ), scipy.signal.fftconvolve( z, -c, mode = 'valid' )


##
#-----------------------------------------------------------------------------
#                               Aux. Functions
#-----------------------------------------------------------------------------
def acc_correct(fdir, fname_xml, crop_bottom=0, crop_top=0, crop_left=0, crop_right=0, resize_factor=1):
    #fdir = '/media/hdd_data1/michael/google_drive/2step_inscopix/two_step_Black2/'
    m = cm.utils.one_photon.load_inscopix(fname_xml, bottom=crop_bottom, top=crop_top, left=crop_left, right=crop_right)

    crp = cm.utils.one_photon.fix_corrupt_frames(m, dview=dview)

    fname_tif = fdir+fname_xml.rstrip('.xml') + '_concat.tif'

    m.save(fname_tif, to32=False, clip_percentiles=None)

    del(m)

    shifts_opencv = False
    min_mov = 0
    max_shifts = (10,10)

    max_dev = 0
    strides = (64,64)
    overlaps = (32,32)
    upsample_factor_grid = 4 #4*max_dex

    g_sigma_smooth = 1.2
    g_sigma_background = 12.

    k_std_smooth = 6# kernel size will be kernel_std * sigma
    k_std_background = 4

    mcf = cm.motion_correction.MotionCorrect1P(fname_tif, min_mov=min_mov, max_shifts=max_shifts, shifts_opencv=shifts_opencv, dview=dview, niter_rig=2, splits_rig=40, max_deviation_rigid=max_dev, strides=strides, overlaps=overlaps, upsample_factor_grid=upsample_factor_grid, niter_els=1, num_splits_to_process_els=[7, None], g_sigma_smooth=g_sigma_smooth, g_sigma_background=g_sigma_background, k_std_smooth=k_std_smooth, k_std_background=k_std_background)

    mcf.motion_correct_rigid(save_movie=True, remove_blanks=True)

    m = cm.load(mcf.fname_tot_rig)

    if resize_factor != 1:
        m = m.resize(fx=resize_factor, fy=resize_factor)

    m.save(fname_tif.rstrip('.tif')+'_resize_factor_{}.tif'.format(resize_factor), to32=False, clip_percentiles=None)

    del m


def acc_correct_single_file(fdir, fname_tif, crop_bottom=0, crop_top=0, crop_left=0, crop_right=0, resize_factor=1):
    #fdir = '/media/hdd_data1/michael/google_drive/2step_inscopix/two_step_Black2/'
    m = cm.load(fdir+fname_tif)

    crp = cm.utils.one_photon.fix_corrupt_frames(m, dview=dview)

    fname_tif = fdir+fname_tif.rstrip('.tif') + '_concat.tif'

    m.save(fname_tif, to32=False, clip_percentiles=None)

    del(m)

    shifts_opencv = True
    min_mov = 0
    max_shifts = (10,10)

    max_dev = 10
    #strides = (64,64)
    #overlaps = (32,32)
    strides = (64,64)
    overlaps = (32,32)
    upsample_factor_grid = 4 #4*max_dex

    g_sigma_smooth = .4
    g_sigma_background = 14.

    k_std_smooth = 4# kernel size will be kernel_std * sigma
    k_std_background = 4

    #mcf = cm.motion_correction.MotionCorrect1P(fname_tif, min_mov=min_mov, max_shifts=max_shifts, shifts_opencv=shifts_opencv, dview=dview, niter_rig=2, splits_rig=40, max_deviation_rigid=max_dev, strides=strides, overlaps=overlaps, upsample_factor_grid=upsample_factor_grid, niter_els=1, num_splits_to_process_els=[None], g_sigma_smooth=g_sigma_smooth, g_sigma_background=g_sigma_background, k_std_smooth=k_std_smooth, k_std_background=k_std_background)

    #mcf.motion_correct_rigid(save_movie=True, remove_blanks=True)

    #mcf = cm.motion_correction.MotionCorrect1P(fname_tif, min_mov=min_mov, max_shifts=max_shifts, shifts_opencv=shifts_opencv, dview=dview, niter_rig=2, splits_rig=4, max_deviation_rigid=max_dev, strides=strides, overlaps=overlaps, upsample_factor_grid=upsample_factor_grid, niter_els=2, splits_els=4, num_splits_to_process_els=[None, None], g_sigma_smooth=g_sigma_smooth, g_sigma_background=g_sigma_background, k_std_smooth=k_std_smooth, k_std_background=k_std_background, window_size=59, order=2, savgolay=True)


    #mcf = cm.motion_correction.MotionCorrect(fname_tif, min_mov=min_mov, max_shifts=max_shifts, shifts_opencv=shifts_opencv, dview=dview, niter_rig=2, splits_rig=4, max_deviation_rigid=max_dev, strides=strides, overlaps=overlaps, upsample_factor_grid=upsample_factor_grid, splits_els=4, num_splits_to_process_els=[None, None], gSig_filt=[14,14])

    #import warnings
    #warnings.simplefilter('error')

    import mkl
    mkl.set_num_threads(6)

    #mcf.motion_correct_pwrigid(save_movie=True)#, remove_blanks=True)
    #m = cm.load(mcf.fname_tot_els)

    mcf = cm.motion_correction.motion_correct_oneP_nonrigid(fname_tif, [int(g_sigma_background)]*2, max_shifts, strides, overlaps, 4, 4, max_dev, dview, 4)

    m = cm.load(mcf.fname_tot_els)

    #mcf.motion_correct_rigid(save_movie=True, remove_blanks=True)
    #m = cm.load(mcf.fname_tot_rig)

    #import ipdb; ipdb.set_trace()

    if resize_factor != 1:
        m = m.resize(fx=resize_factor, fy=resize_factor)

    m.save(fname_tif.rstrip('.tif')+'_resize_factor_{}.tif'.format(resize_factor), to32=False, clip_percentiles=None)

    del m


##
#-----------------------------------------------------------------------------
#                               Parameters
#-----------------------------------------------------------------------------

#fdir = '/media/hdd_data1/michael/google_drive/2step_inscopix/two_step_Black1/'
#crop = (0, 6, 15, 54)

#fdir = '/media/hdd_data1/michael/google_drive/2step_inscopix/two_step_Black2/'
#crop = (0, 20, 60, 40)

#fdir = '/media/hdd_data1/michael/google_drive/2step_inscopix/two_step_Green1/'
#crop = (0, 38, 67, 15)

#fdir = '/media/hdd_data1/michael/google_drive/2step_inscopix/two_step_Green2/'
#crop = (80, 0, 61, 30)


#os.chdir(fdir)
#xml_files = []
#for file in glob.glob("*.xml"):
#    xml_files.append(file)
#
#for file in xml_files:
#    acc_correct(fdir, file, *crop)

fdir = '/media/hdd_data1/michael/andreas2018/'
fname_tif = 'test.tif'

acc_correct_single_file(fdir, fname_tif)
#fdir = '/home/michael/data/priority/black1/'
#xml = 'recording_20160717_143955.xml'
#xml = 'recording_20160720_160238.xml'
#
#fdir = '/home/michael/data/priority/green1/'
#xml = 'recording_20160716_172303.xml'
#
#fdir = '/media/hdd_data1/michael/priority/black2/'
#bottom, top, left, right = (1, 30, 100, 80)
#xml = 'recording_20160716_162425.xml'
#xml = 'recording_20160721_182115.xml'

#fname = fdir + xml

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

"""

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
#
#import caiman.utils.one_photon

m = cm.utils.one_photon.load_inscopix(fname, bottom=bottom, top=top, left=left, right=right)

crp = cm.utils.one_photon.fix_corrupt_frames(m, dview=dview)
##
#-----------------------------------------------------------------------------
#                         Compute Baseline Corr. Img.
#-----------------------------------------------------------------------------
#old_corr = cm.summary_images.local_correlations(m, False, False)
#
#pl.figure()
#pl.imshow(old_corr)
#pl.title('Raw data correlation image')


fname_tif = fdir + xml.rstrip('.xml') + '_concat.tif'
m.save(fname_tif, to32=False, clip_percentiles=None)
del(m)
#FIXME: change tif name


##
#-----------------------------------------------------------------------------
#                            Compute pre-MC CI
#-----------------------------------------------------------------------------
#first_cc = cm.summary_images.local_correlations(m, False, False)
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

g_sigma_smooth = 1.
g_sigma_background = 10.

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

#m.gaussian_blur_2D(*2*[k_size_smooth], *2*[g_sigma_smooth])

#m += int(m.max())
#
#kernel = cv2.getGaussianKernel(k_size_background, g_sigma_background)
#kernel = kernel*kernel.transpose()
#
#center_i = int(k_size_background/2)
#kernel[center_i, center_i] = 0
#kernel /= np.sum(kernel)
#kernel = -kernel
#kernel[center_i, center_i] = 1.# this is equivalent to adding a delta fun.
#
#m.filter_2D(kernel, dview=dview)
#
#m = m.crop(bound, bound, bound, bound)
#
###
#new_corr = cm.summary_images.local_correlations(m, False, False)
#
#pl.figure()
#pl.imshow(first_cc)
#pl.title('Original')
#pl.figure()
#pl.imshow(new_corr)
#pl.title('After filtering')
#
###
#fname_filt = '/home/michael/data/filtered_movie.tif'
#m.save(fname_filt)

#c,dview,n_processes = cm.cluster.setup_cluster(backend = 'local',n_processes = 20, single_thread = False)

##
# Compute metrics
m = cm.load(fname_corr)
final_x_size = m.shape[1]
final_y_size = m.shape[2]
del(m)

#metrics_b = cm.motion_correction.compute_metrics_motion_correction(fname_corr,
#                                                       final_x_size,
#                                                       final_y_size, False)

metrics_b = cm.motion_correction.compute_metrics_filter(
    fname_corr, final_x_size, final_y_size, swap_dim=False,
    g_sigma_smooth=g_sigma_smooth, g_sigma_background=g_sigma_background,
    k_std_smooth=k_std_smooth, k_std_background=k_std_background, dview=dview)


#template_b, correlations_b, flows_b, norms_b, smoothness_b, img_corr_b, smoothness_corr_b = metrics_b

smoothness, smoothness_corr, rof, rof_std, rof_std_t, rof_std_loc, dct_sharp, dct_sharp_corr, dft_sharp, dft_sharp_corr = metrics_b

#flow_rms_b = np.sqrt(np.mean([i**2 for i in norms_b]))
##
#print(('batch\t',smoothness_b, smoothness_corr_b, '\n'))
report = ['batch\t',smoothness, smoothness_corr, rof, rof_std, '\n'] + report


#m = load_inscopix(fname_xml, bottom=25, top=150, left=220, right=174,
#                  resize=(0.25, 0.25))
#m = m.crop(bound, bound, bound, bound)

#m = cm.load(fname_tif) #reload movie
#
#max_h,max_w = np.ceil(np.max(mcf.shifts_rig,axis=0)).astype('int')
#min_h,min_w = np.floor(np.min(mcf.shifts_rig,axis=0)).astype('int')
#
#m = m.crop(crop_top=max_h, crop_bottom=-min_h, crop_left=max_w
#               ,crop_right=-min_w, crop_begin=0,crop_end=0)
#
#fname_or_crop = 'original_crop.npz'
#m.save(fname_or_crop)
#
#final_x_size = m.shape[1]
#final_y_size = m.shape[2]
#
#metrics_o = cm.motion_correction.compute_metrics_motion_correction(fname_or_crop,
#                                                       final_x_size,
#                                                       final_y_size, False)
#
#template_o, correlations_o, flows_o, norms_o, smoothness_o, img_corr_o, smoothness_corr_o = metrics_o
#
#report = ['original\t',smoothness_o, smoothness_corr_o, '\n'] + report

print(*report)

#1.4    5649.32 5.00986131917 0.238281 0.171508

"""

pl.show()
