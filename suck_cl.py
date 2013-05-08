#!/usr/bin/env python
"""Merge data from chromaluma
CREATED:2013-05-08 13:54:35 by Brian McFee <brm2132@columbia.edu>
 
Usage:

    ./suck_cl.py output_cl.npy input_glob
"""

import sys
import os
import numpy as np
import glob


def merge_data(inglob):

    files = glob.glob(inglob)
    files.sort()

    X = None
    for f in files:
        print os.path.basename(f)
        if X is None:
            X = np.load(f)
        else:
            X = np.vstack((X, np.load(f)))

    return X.astype(np.float32)


if __name__ == '__main__':
    outfile = sys.argv[1]
    inglob = sys.argv[2]

    X = merge_data(inglob)
    np.save(outfile, X)
    pass
