##
import cv2
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

import caiman as cm
import numpy as np
import pylab as pl

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
#xml = 'recording_20160716_162425.xml'
xml = 'recording_20160721_182115.xml'
bottom, top, left, right = (50, 90, 95, 225)

fname = fdir + xml

method = 'rank'

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
#                               Save NPY
#-----------------------------------------------------------------------------
#fname_npy = fname.rstrip('.xml') + '_concatenated.npy'
fname_tif = fname.rstrip('.xml') + '_concatenated.tif'

m.save(fname_tif, to32=False, clip_percentiles=None)
#FIXME: change tif name

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
#                            Compute pre-MC CI
#-----------------------------------------------------------------------------
first_cc = cm.summary_images.local_correlations(m, False, False)

pl.figure()
pl.imshow(first_cc)
pl.title('Post-corrupt frame repair correlation image')


pl.show()
