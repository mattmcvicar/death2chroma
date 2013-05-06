# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Functions for computing 2d chroma features, ie chroma with luma.
'''

# <codecell>

import numpy as np
import scipy.weave

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
    
def logFrequencySpectrum( audioData, fs, minFreq, binsPerOctave ):
    '''
    Compute log-frequency spectrum.  Based on code by DAn Ellis.

    Input:
        audioData - vector of audio samples
        fs - sampling rate
        minFreq - minimum frequency to consider
        binsPerOctave - number of magnitude values to compute per octave
    Output:
        spectrum - log-frequency spectrum
    '''
    
    # Number of samples
    N = float(audioData.shape[0])
    
    # Compute FFT
    X = np.fft.rfft( np.hanning( N ) * audioData )
    X = np.abs(X)**2
    
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
    z2 = (2 * (mx**2).sum(axis=1))**-0.5
    
    # Perform mapping in magnitude-squared (energy) domain
    return np.sqrt( z2 * mx.dot(X) )

# <codecell>

def CLSpectrum( audioData, fs, minNote, nOctaves, binsPerOctave, **kwargs ):
    '''
    Computes a 2d chroma-luma matrix
    
    Input:
        audioData - vector of audio samples, size nSamples
        fs - sampling rate
        minNote - lowest MIDI note to get chroma value for
        nOctaves - number of octaves
        binsPerOctave - number of bins in each octave
        log - take a log of the spectrum before chromafying?
    Output:
        chromaLuma - chroma-luma matrix, size nOctaves x binsPerOctave
    '''
    log = kwargs.get( 'log', True )
    # Get log-freq spectrum
    semitrum = logFrequencySpectrum( audioData, fs, midiToHz( minNote ), binsPerOctave )
    # Optionally compute log
    if log:
        semitrum = 20*np.log10( semitrum + 1e-40 )
    # Truncate to the requested number of octaves
    semitrum = semitrum[:binsPerOctave*nOctaves]
    # Wrap into chroma-luma matrix
    return np.reshape( semitrum, (nOctaves, binsPerOctave) )

# <codecell>

def midiToHz( note ):
    '''
    Get the frequency of a MIDI note.
    
    Input:
        note - MIDI note number (can be float)
    Output:
        frequency - frequency in Hz of MIDI note
    '''
    return 440.0*(2.0**((note - 69)/12.0))

