"""
Utilities for analysing one photon imaging videos, particularly the ones
outputted by Inscopix miniature miscroscopes.
"""
__author__ = "Michael Pereira"

import cv2
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

import caiman as cm
import numpy as np
import pylab as pl

from tqdm import tqdm
import functools

class LoadInscopixError(Exception):
    def __init__(self, message):
        super().__init__(message)


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

    if int(metadata['frames']) == 1:
        raise LoadInscopixError('This XML file points to a single frame. Snapshots are not supported')

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

def _check_frame_rank(f_i, smallest_len, largest_len, largest_dim):
    f, i = f_i
    if np.linalg.matrix_rank(f) < smallest_len:
        return i
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
            return i
        else:
            return False

def fix_corrupt_frames(m, method='rank', dview=None):
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
        f_shape = m.shape[1:3]
        smallest_len = min(f_shape)
        largest_len = max(f_shape)
        largest_dim = np.argmax(f_shape)

        check_frame_rank = functools.partial(_check_frame_rank,
                                             smallest_len=smallest_len,
                                             largest_len=largest_len,
                                             largest_dim=largest_dim)
        to_fix_all = []
        if dview:
            if 'multiprocessing' in str(type(dview)):
                crp = dview.imap(check_frame_rank, list(zip(m, range(len(m)))))
            else:
                crp = dview.map_sync(check_frame_rank, m)

            for idx in crp:#tqdm(crp, desc='Searching for corrupt frames by rank', total=len(m)):
                if idx is not False:
                    to_fix_all.append(idx)

        else:
            crp = map(check_frame_rank, list(zip(m, range(len(m)))))
            for idx in crp:#tqdm(crp, desc='Searching for corrupt frames by rank', total=len(m)):
                if idx is not False:
                    to_fix_all.append(idx)

    #TODO: instead of using first this to detect corrupt frames and then the
    #difference to the average between frame before and after use already the second
    #to detect corrupt frames

    ##
    #-----------------------------------------------------------------------------
    #                               Reload Movie
    #-----------------------------------------------------------------------------
#    if method != 'rank':
#        m = load_inscopix(fname, bottom=bottom, top=top, left=left, right=right)


    ##
    #-----------------------------------------------------------------------------
    #                            Fix Corrupt Frames
    #-----------------------------------------------------------------------------
    if to_fix_all:
        to_fix = np.copy(to_fix_all)

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
            for i in to_fix:
        # in case it is the first or last frame there's nothing to interpolate from
        # so we cheat and substitute by the neighbouring frame
                m[i,:,:] = np.mean((m[i-1,:,:], m[i+1,:,:]), axis=0)

            i_still_to_fix = np.nonzero(np.any(np.abs(m[to_fix_before,:,:] -
                                                      np.mean((m[to_fix_before-1,:,:],
                                                       m[to_fix_before+1,:,:]), axis=0)) > 0.5,
                              axis=(1,2)))[0]
            to_fix = to_fix_before[i_still_to_fix]
    return to_fix_all


