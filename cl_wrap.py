#!/usr/bin/env python
"""CDL encoding PCA fitter
CREATED:2013-05-08 16:15:55 by Brian McFee <brm2132@columbia.edu>

Usage:

./cl_wrap.py n_jobs /path/to/octarines

Wrap down to a single octave and re-normalize

Saves output alongside as -wrapCL.npy

"""

import os
import sys
import glob
import numpy as np
from joblib import Parallel, delayed


def process_song(W, song):

    songname = os.path.basename(song)
    songname = songname[:songname.index('-CL.npy')]

    print songname
    X = np.load(song)

    Xhat = X.dot(W)
    zXhat = Xhat.max(axis=1, keepdims = True)
    zXhat[zXhat == 0] = 1.0
    Xhat = Xhat / zXhat
    outname = '%s/%s-wrapCL.npy' % (os.path.dirname(song), songname)
    np.save(outname, Xhat)
    pass

def process_data(n_jobs, W, file_glob):
    files = glob.glob(file_glob)
    files.sort()

    Parallel(n_jobs=n_jobs)(delayed(process_song)(W, song) for song in files)
    pass

if __name__ == '__main__':
    n_jobs = int(sys.argv[1])
    
    W = np.eye(48)
    W = np.tile(W, (4, 1))

    file_glob = '%s/*-CL.npy' % sys.argv[2]
    process_data(n_jobs, W, file_glob)
