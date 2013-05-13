#!/usr/bin/env python
"""CDL encoding transformer fitter
CREATED:2013-05-08 16:15:55 by Brian McFee <brm2132@columbia.edu>

Usage:

./cdl_compress.py n_jobs pca_model.pickle /path/to/octarines/glob

Once we have a transformer model and a set of -encoded.npy files, use transformer to compress them.

Saves output alongside as -encoded-compressed.npy

"""

import os
import sys
import glob
import cPickle as pickle
import numpy as np
from joblib import Parallel, delayed


RETAIN = 0.95

def vectorize(A):
    return A.squeeze().reshape((A.shape[0], -1))

def process_song(transformer, d, song, ext):

    songname = os.path.basename(song)
    songname = songname[:songname.index('-encoded.npy')]

    print songname
    A = vectorize(np.load(song))

    # Transform the data, project to top $RETAIN variance dimensions
    Ahat = transformer.transform(A)[:,:d]

    outname = '%s/%s-encoded-%s.npy' % (os.path.dirname(song), songname, ext)
    np.save(outname, Ahat)
    pass

def process_data(n_jobs, transformer, file_glob, ext):
    files = glob.glob(file_glob)
    files.sort()

    d = -1
    if hasattr(transformer, 'explained_variance_ratio_'):
        d = np.argmax(np.cumsum(transformer.explained_variance_ratio_) >= RETAIN)

    Parallel(n_jobs=n_jobs)(delayed(process_song)(transformer, d, song, ext) for song in files)
    pass

if __name__ == '__main__':
    n_jobs = int(sys.argv[1])
    with open(sys.argv[2], 'r') as f:
        transformer = pickle.load(f)

    file_glob = sys.argv[3]
    if len(sys.argv) > 4:
        ext = sys.argv[4]
    else:
        ext = 'compressed'
    process_data(n_jobs, transformer, file_glob, ext)
