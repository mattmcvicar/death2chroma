#!/usr/bin/env python
"""Octarine CDL encoder
CREATED:2013-05-08 10:52:24 by Brian McFee <brm2132@columbia.edu>

Usage:

./cdl_encoder.py K /path/to/cdl_model.pickle /path/to/octarines/glob

    K:                  number of encoders to run in parallel
    cdl_model.pickle:   pre-trained CDL container object
    octarines glob:     eg /path/to/files/*-CDL.npy

will output as:

    /path/to/octainres/$file-cdl.npy
"""

import os
import sys
import glob
import cPickle as pickle
import numpy as np

from joblib import Parallel, delayed

def load_cdl(path):
    with open(path, 'r') as f:
        coder = pickle.load(f)

    return coder

def process_song(song, coder):

    songname = os.path.basename(song)
    songname = songname[:songname.index('-CL.npy')]

    print songname
    X = np.load(song)

    # Chop off the top octave
    X = X[:,:-48]

    # Pad out dimension
    X = X.reshape((X.shape[0], X.shape[1], 1), order='A')

    # Encode the frames
    A = coder.transform(X)

    # Save the result
    outname = '%s/%s-encoded.npy' % (os.path.dirname(song), songname)
    np.save(outname, A)
    pass

def process_data(n_jobs, coder, fileglob):

    files = glob.glob(fileglob)
    files.sort()

    Parallel(n_jobs=n_jobs)(delayed(process_song)(song, coder) for song in files)
    pass

if __name__ == '__main__':
    n_jobs = int(sys.argv[1])

    coder = load_cdl(sys.argv[2])

    process_data(n_jobs, coder, sys.argv[3])
