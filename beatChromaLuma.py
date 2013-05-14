# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Compute beat-synchronous chroma-luma matrices
'''

# <codecell>

import librosa
import numpy as np
import chromaLuma
import scipy.signal

# <codecell>

def onset_strength_median(y=None, sr=22050, S=None, **kwargs):
    """Extract onsets from an audio time series or spectrogram, using median
        
        Arguments:
        y         -- (ndarray) audio time-series          | default: None
        sr        -- (int)     sampling rate of y         | default: 22050
        S         -- (ndarray) pre-computed spectrogram   | default: None
        
        **kwargs    -- Parameters to mel spectrogram, if S is not provided
        
        See librosa.feature.melspectrogram() for details
        
        Note: if S is provided, then (y, sr) are optional.
        
        Returns onsets:
        onsets    -- (ndarray) vector of onset strength
        
        Raises:
        ValueError -- if neither (y, sr) nor S are provided
        
        """
    
    # First, compute mel spectrogram
    if S is None:
        if y is None:
            raise ValueError('One of "S" or "y" must be provided.')
        
        S   = librosa.feature.melspectrogram(y, sr = sr, **kwargs)
        
        # Convert to dBs
        S   = librosa.logamplitude(S)
    
    
    ### Compute first difference
    onsets  = np.diff(S, n=1, axis=1)
    
    ### Discard negatives (decreasing amplitude)
    #   falling edges could also be useful segmentation cues
    #   to catch falling edges, replace max(0,D) with abs(D)
    onsets  = np.maximum(0.0, onsets)
    
    ### Average over mel bands
    onsets  =  np.median(onsets, axis=0)
    
    return onsets

# <codecell>

def getTuningOffset( spectrogram, fs ):
    '''
    Given a spectrogram of a song, compute its tuning offset in semitones
    
    Input: 
        spectrogram - Magnitude STFT of a song, size nBins x nFrames
        fs - sampling rate of the song
    Output:
        tuningOffset - tuning offset in semitones of this song
    '''
    # Pre-allocate with the largest it could be
    semiDiffs = np.zeros( spectrogram.shape[0]*spectrogram.shape[1] )
    n = 0
    # Freqs corresponding to each bin in FFT
    for X in spectrogram.T:
        # Compute local maxima of DFT
        Xc = X*(X > .1*np.max( X ))#*(X > 2*scipy.signal.medfilt( X, 31 ) )
        localMax = np.flatnonzero( np.logical_and(Xc > np.hstack([Xc[0], Xc[:-1]]), Xc > np.hstack([Xc[1:], Xc[-1]])) )
        # Parabolic interpolation
        alpha = X[localMax - 1]
        beta = X[localMax]
        gamma = X[localMax + 1]
        localMax = localMax + .5*(alpha - gamma)/(alpha - 2*beta + gamma)
        # Get frequencies corresponding to the local max
        localMaxFreqs = localMax*fs/(2.0*(spectrogram.shape[0] - 1))
        # Convert to MIDI note number (Hz)
        localMaxNotes = librosa.feature.hz_to_midi( localMaxFreqs )
        # Throw out values outside of musical range
        localMaxNotes = localMaxNotes[np.logical_and( localMaxNotes >= 24, localMaxNotes < 108 )]
        # Compute semitone differences
        trackSemiDiffs = localMaxNotes - np.round( localMaxNotes )
        semiDiffs[n:n + trackSemiDiffs.shape[0]] = trackSemiDiffs
        n += trackSemiDiffs.shape[0]
    semiDiffs = semiDiffs[:n]
    counts, bins = np.histogram( semiDiffs, 100 )
    bestBin = np.argmax( counts )
    return (bins[bestBin] + bins[bestBin + 1])/2.0

# <codecell>

def beatChromaLuma( filename, **kwargs ):
    '''
    Given a file, get the beat-synchronous chroma-luma matrices
    
    Input:
        filename - full path to file to process
        minNote - minimum note number to consider, default 35.5
        binsPerOctave - number of magnitude values to compute per octave, default 48
        nOctaves - number of octaves, default 4
        smoothingWindow - window to use to smooth the spectrum, None = don't smooth, default np.hanning( binsPerOctave )
        smoothingPower - power to raise spectral envelope to, default 3.0, ignored if smoothingWindow=None
        aWeight - whether or not to a-weight the spectrum, default False
        takeLog - whether or not to take a log, default True
    Output:
        tuning - estimated tuning offset of the song, in semitones
        beatTimes - vector of beat locations, in seconds, size nBeats
        semitrums - matrix of per-beat semitrums, size nBeats x binsPerOctave*nOctaves
    '''

    binsPerOctave = kwargs.get( 'binsPerOctave', 48 )
    nOctaves = kwargs.get( 'nOctaves', 4 )

    # Read in audio data
    audioData, fs = librosa.load( filename, sr=22050 )
    hop = 64
    frameSize = 2048
    # Get beat locations - using modified median filter version for the onset envelope
    _, beats = librosa.beat.beat_track( sr=fs, onsets=onset_strength_median( audioData, fs, hop_length=hop, n_fft=frameSize, n_mels=128 ), hop_length=hop, n_fft=frameSize )
    # Convert beat locations to samples
    beatSamples = beats*hop
    # Get harmonic component of signal
    spectrogram = librosa.stft( audioData, n_fft=frameSize, hop_length=frameSize/4 )
    harmonicSpectrogram, _ = librosa.hpss.hpss_median( np.abs( spectrogram ), win_P=13, win_H=13, p=4 )
    # Compute tuning offset
    tuningOffset = getTuningOffset( harmonicSpectrogram, fs )
    harmonicSpectrogram = harmonicSpectrogram*np.exp( 1j*np.angle( spectrogram ) )
    harmonicData = librosa.istft( harmonicSpectrogram, n_fft=frameSize, hop_length=frameSize/4 )
    # Compute a chroma-luma matrix for each beat
    semitrums = np.zeros( (beats.shape[0], nOctaves*binsPerOctave) )
    # Keep track of semitone differences
    for n, (beatStart, beatEnd) in enumerate( zip( beatSamples[:-1], beatSamples[1:] ) ):
        # Grab audio samples within this beat
        beatData = harmonicData[beatStart:beatEnd]
        semitrums[n] = chromaLuma.logFrequencySpectrum( beatData, fs, **kwargs )
    return librosa.frames_to_time( beats, fs, hop ), semitrums

# <codecell>

def fakeSemigram( labelsFile, binsPerOctave, nOctaves ):
    '''
    Given annotations, generates a synthetic semigram.
    
    Input:
        labelsFile - chord label file of the song in question
        binsPerOctave - number of semigram bins in each octave
        nOctaves - number of octaves to compute
    Output:
        fakeSemigram - a synthetic semigram matrix, size nBins x nBeats
    '''
    import pickle
    with open( "Training_Scripts/dict_minmaj.p" ) as f:
        labelToIntervals = pickle.load( f )[1]
    labels = np.load( labelsFile )
    baseOctave = np.zeros( (binsPerOctave, labels.shape[0]), dtype=np.bool )
    binsPerSemi = binsPerOctave/4
    for n, label in enumerate( labels ):
        for semi in labelToIntervals[label]:
            semis = np.arange( semi*4, (semi+1)*4 )
            baseOctave[semis, n] = 1
    octaves = np.tile( baseOctave, (nOctaves, 1) )
    octaves = np.roll( octaves, -binsPerOctave/24, axis=0 )
    return octaves

# <codecell>

if __name__ == '__main__':
    # Create .npy files for each beatles mp3
    import os
    import glob
    mp3Files = glob.glob( 'data/beatles/*.mp3' )
    for mp3File in mp3Files:
        beats, semitrums = beatChromaLuma( mp3File )
        nameBase = os.path.splitext( mp3File )[0]
        np.save( nameBase + '-beats.npy', beats )
        np.save( nameBase + '-CL-magnitude.npy', semitrums )
    mp3Files = glob.glob( os.path.join( 'data/uspop2002/*.mp3' ) )
    for mp3File in mp3Files:
        beats, semitrums = beatChromaLuma( mp3File )
        nameBase = os.path.splitext( mp3File )[0]
        np.save( nameBase + '-beats.npy', beats )
        np.save( nameBase + '-CL-magnitude.npy', semitrums )

