#!/usr/bin/env python
"""CDL encoding FDA fitter
CREATED:2013-05-08 16:15:55 by Brian McFee <brm2132@columbia.edu>

Usage:

./cdl_fda_fit.py output_fda_model.pickle /path/to/octarines/glob label_type

"""


import sys
import glob
import cPickle as pickle
import numpy as np
import FDA

def vectorize(A):
    return A.squeeze().reshape((A.shape[0], -1))

def load_labels(infile, label_type):
    '''filename has some -CL-encoded garbage: get the label instead'''

    infile = '%s-%s.npy' % (infile[:infile.index('-encoded.npy')], label_type)
    return np.load(infile)

def learn_fda(inpath, label_type):

    files = glob.glob(inpath)
    files.sort()

    A = None

    print 'Loading data...'
    for f in files:
        Anew = vectorize(np.load(f))
        Ynew = load_labels(f, label_type)
        if A is None:
            A = Anew
            Y = Ynew
        else:
            A = np.vstack((A, Anew))
            Y = np.hstack((Y, Ynew))

    print 'Building FDA model...'
    transformer = FDA.FDA()
    transformer.fit(A, Y)
    print 'done.'
    return transformer


if __name__ == '__main__':
    outpath = sys.argv[1]
    inpath  = sys.argv[2]
    label_type = sys.argv[3]

    transform = learn_fda(inpath, label_type)

    with open(outpath, 'w') as f:
        pickle.dump(transform, f, protocol=-1)
