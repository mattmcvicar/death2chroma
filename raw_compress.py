#!/usr/bin/env python
"""CDL encoding PCA fitter
CREATED:2013-05-08 16:15:55 by Brian McFee <brm2132@columbia.edu>

Usage:

./cdl_compress.py n_jobs pca_model.pickle /path/to/octarines

Once we have a PCA model and a set of -encoded.npy files, use PCA to compress them.

Saves output alongside as -encoded-compressed.npy

"""

import os
import sys
import glob
import cPickle as pickle
import numpy as np
from joblib import Parallel, delayed


RETAIN = 0.95

def process_song(PCA, d, song):

    songname = os.path.basename(song)
    songname = songname[:songname.index('-CL.npy')]

    print songname
    X = np.load(song)

    # Transform the data, project to top $RETAIN variance dimensions
    Xhat = PCA.transform(X)[:,:d]

    outname = '%s/%s-raw-compressed.npy' % (os.path.dirname(song), songname)
    np.save(outname, Xhat)
    pass

def process_data(n_jobs, PCA, file_glob):
    files = glob.glob(file_glob)
    files.sort()

    d = np.argmax(np.cumsum(PCA.explained_variance_ratio_) >= RETAIN)

    Parallel(n_jobs=n_jobs)(delayed(process_song)(PCA, d, song) for song in files)
    pass

if __name__ == '__main__':
    n_jobs = int(sys.argv[1])
    with open(sys.argv[2], 'r') as f:
        PCA = pickle.load(f)

    file_glob = '%s/*/*-CL.npy' % sys.argv[3]
    process_data(n_jobs, PCA, file_glob)
