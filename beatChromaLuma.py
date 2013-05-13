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

def beatChromaLuma( filename, **kwargs ):
    '''
    Given a file, get the beat-synchronous chroma-luma matrices
    
    Input:
        filename - full path to file to process
        minNote - minimum note number to consider, default 36
        binsPerOctave - number of magnitude values to compute per octave, default 48
        nOctaves - number of octaves, default 4
        smoothingWindow - window to use to smooth the spectrum, None = don't smooth, default np.hanning( binsPerOctave )
        smoothingPower - power to raise spectral envelope to, default 3.0, ignored if smoothingWindow=None
        aWeight - whether or not to a-weight the spectrum, default False
        takeLog - whether or not to take a log, default True
    Output:
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
    harmonicSpectrogram = harmonicSpectrogram*np.exp( 1j*np.angle( spectrogram ) )
    harmonicData = librosa.istft( harmonicSpectrogram, n_fft=frameSize, hop_length=frameSize/4 )
    # Compute a chroma-luma matrix for each beat
    semitrums = np.zeros( (beats.shape[0], nOctaves*binsPerOctave) )
    for n, (beatStart, beatEnd) in enumerate( zip( beatSamples[:-1], beatSamples[1:] ) ):
        # Grab audio samples within this beat
        beatData = harmonicData[beatStart:beatEnd]
        semitrums[n] = chromaLuma.logFrequencySpectrum( beatData, fs, **kwargs )
    return librosa.frames_to_time( beats, fs, hop ), semitrums

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

