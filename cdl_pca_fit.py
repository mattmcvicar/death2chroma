#!/usr/bin/env python
"""CDL encoding PCA fitter
CREATED:2013-05-08 16:15:55 by Brian McFee <brm2132@columbia.edu>

Usage:

./cdl_pca_fit.py output_pca_model.pickle /path/to/octarines

"""


import sys
import glob
import cPickle as pickle
import numpy as np
import sklearn.decomposition

def vectorize(A):
    return A.squeeze().reshape((A.shape[0], -1))


def learn_pca(inpath):

    files = glob.glob('%s/*/*-encoded.npy' % inpath)
    files.sort()

    A = None

    print 'Loading data...'
    for f in files:
        if A is None:
            A = vectorize(np.load(f))
        else:
            A = np.vstack((A, vectorize(np.load(f))))

    print 'Building PCA model...'
    PCA = sklearn.decomposition.PCA()
    PCA.fit(A)
    print 'done.'
    return PCA


if __name__ == '__main__':
    outpath = sys.argv[1]
    inpath  = sys.argv[2]

    PCA     = learn_pca(inpath)

    with open(outpath, 'w') as f:
        pickle.dump(PCA, f, protocol=-1)
