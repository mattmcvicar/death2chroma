# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Functions for computing 2d chroma features, ie chroma with luma.
'''

# <codecell>

import numpy as np
import scipy.weave
import scipy.signal
import librosa

# <codecell>

def apdiff(x, y):
    '''Compute an all-pairs difference matrix:  D[i,j] = x[i] - y[j]
    
    Input:
        x - vector, arbitrary size
        y - vector, arbitrary size
    Output:
        D - difference matrix, size x.shape[0] x y.shape[0]
    '''
    
    nx, ny = len(x), len(y)

    D = np.empty( (nx, ny), dtype=x.dtype)
    
    weaver = r"""
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            D[i * ny + j] = x[i] - y[j];
        }
    }
"""
    scipy.weave.inline(weaver, arg_names=['nx', 'ny', 'x', 'y', 'D'])
    return D

# <codecell>

def logFrequencySpectrum( audioData, fs, **kwargs  ):
    '''
    Compute log-frequency spectrum.  Based on code by DAn Ellis.

    Input:
        audioData - vector of audio samples
        fs - sampling rate
        minNote - minimum note number to consider, default 24
        binsPerOctave - number of magnitude values to compute per octave, default 48
        nOctaves - number of octaves, default 6
        smoothingWindow - window to use to smooth the spectrum, None = don't smooth, default np.hanning( binsPerOctave )
        aWeight - whether or not to a-weight the spectrum, default False
        takeLog - whether or not to take a log, default True
    Output:
        spectrum - log-frequency spectrum
    '''
    
    minNote = kwargs.get( 'minNote', 24 )
    binsPerOctave = kwargs.get( 'binsPerOctave', 48 )
    nOctaves = kwargs.get( 'nOctaves', 6 )
    smoothingWindow = kwargs.get( 'smoothingWindow', np.hanning( binsPerOctave ) )
    aWeight = kwargs.get( 'aWeight', False )
    takeLog = kwargs.get( 'takeLog', True )
    
    minFreq = librosa.feature.midi_to_hz( minNote )
    
    # Number of samples
    N = float(audioData.shape[0])
    
    # Compute FFT
    X = np.fft.rfft( np.hanning( N ) * audioData )
    X = np.abs( X )
    
    # Ratio between adjacent frequencies in log-f axis
    frequencyRatio = 2.0**(1.0/binsPerOctave)
    
    # How many bins in log-f axis
    nBins = np.floor( np.log((fs/2.0)/minFreq)/np.log(frequencyRatio) )
    
    # Freqs corresponding to each bin in FFT
    fftFreqs = np.arange( N/2.0 + 1.0 )*(fs/N)
    
    # Freqs corresponding to each bin in log F output
    logFFTFreqs = minFreq*np.exp( np.log( 2 )*np.arange( nBins )/binsPerOctave)
    
    # Bandwidths of each bin in log F
    logFreqBandwidths = logFFTFreqs*(frequencyRatio - 1)
    
    # .. but bandwidth cannot be less than FFT binwidth
    logFreqBandwidths = np.clip( logFreqBandwidths, fs/N, np.inf )
    
    # Controls how much overlap there is between adjacent bands
    overlapFactor = 0.5475   # Adjusted by hand to make sum(mx'*mx) close to 1.0
    
    # Weighting matrix mapping energy in FFT bins to logF bins
    # is a set of Gaussian profiles depending on the difference in 
    # frequencies, scaled by the bandwidth of that bin
    z = (1.0/(overlapFactor * logFreqBandwidths)).reshape((-1, 1))
    
    fftDiff = z*apdiff(logFFTFreqs, fftFreqs)

    mx = np.exp( -0.5*(fftDiff)**2 )
    
    # Normalize rows by sqrt(E), so multiplying by mx' gets approx orig spec back
    z2 = (2*(mx**2).sum(axis=1))**-0.5
    
    # Perform mapping in magnitude domain
    logFrequencyX = np.sqrt( z2*mx.dot(X) )
    
    if smoothingWindow is not None:
        # Compute a spectral envelope for normalizing the spectrum
        normalization = scipy.signal.fftconvolve( logFrequencyX, smoothingWindow, 'same' ) + 1
        logFrequencyX /= normalization
    
    if aWeight:
        # Compute A-weighting values for the spectrum
        logFFTFreqsSquared = logFFTFreqs**2
        weighting = 12200**2*logFFTFreqsSquared**2
        weighting /= logFFTFreqsSquared + 20.6**2
        weighting /= np.sqrt( (logFFTFreqsSquared + 107.7**2)*(logFFTFreqsSquared + 737.9**2) )
        weighting /= logFFTFreqsSquared + 12200*2
        logFrequencyX *= weighting
        
    if takeLog:
        logFrequencyX = librosa.logamplitude( logFrequencyX )
    
    # Truncate by number of octaves requested
    logFrequencyX = logFrequencyX[:binsPerOctave*nOctaves]
    # Normalize
    logFrequencyX = (logFrequencyX - logFrequencyX.min())/(logFrequencyX.max() - logFrequencyX.min())
    
    return logFrequencyX

