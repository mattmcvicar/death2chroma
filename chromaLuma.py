# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Functions for computing 2d chroma features, ie chroma with luma.
'''

# <codecell>

import numpy as np

# <codecell>

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
    N = audioData.shape[0]*1.0
    # Compute FFT
    X = np.fft.rfft( np.hanning( audioData.shape[0] )*audioData )

    # Ratio between adjacent frequencies in log-f axis
    frequencyRatio = 2.0**(1.0/binsPerOctave);
    # How many bins in log-f axis
    nBins = np.floor( np.log((fs/2.0)/minFreq)/np.log(frequencyRatio) );
    # Freqs corresponding to each bin in FFT
    fftFreqs = np.arange( N/2.0 + 1.0 )*(fs/N);
    fftFreqs = np.reshape( fftFreqs, (1, fftFreqs.shape[0]) )
    nFFTBins = N/2.0 + 1;
    # Freqs corresponding to each bin in log F output
    logFFTFreqs = minFreq*np.exp( np.log( 2 )*np.arange( nBins )/binsPerOctave);
    logFFTFreqs = np.reshape( logFFTFreqs, (1, logFFTFreqs.shape[0]) )
    # Bandwidths of each bin in log F
    logFreqBandwidths = logFFTFreqs*(frequencyRatio - 1);
    # .. but bandwidth cannot be less than FFT binwidth
    logFreqBandwidths = np.clip( logFreqBandwidths, fs/N, np.inf );
    # Controls how much overlap there is between adjacent bands
    overlapFactor = 0.5475;   # Adjusted by hand to make sum(mx'*mx) close to 1.0
    # Weighting matrix mapping energy in FFT bins to logF bins
    # is a set of Gaussian profiles depending on the difference in 
    # frequencies, scaled by the bandwidth of that bin
    freqDiff = (np.tile( logFFTFreqs.T, (1, nFFTBins) ) - np.tile( fftFreqs, (nBins, 1) ))/np.tile( overlapFactor*logFreqBandwidths.T, (1, nFFTBins) );
    mx = np.exp( -0.5*freqDiff**2 );
    # Normalize rows by sqrt(E), so multiplying by mx' gets approx orig spec back
    mx = mx/np.tile( np.sqrt( 2*np.array( [np.sum( mx**2, axis=1 )] ).T ), (1, nFFTBins) );
    # Perform mapping in magnitude-squared (energy) domain
    return np.sqrt( np.dot( mx, (np.abs(X)**2) ) );

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

