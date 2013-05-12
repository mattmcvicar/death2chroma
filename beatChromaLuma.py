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
    audioData, fs = librosa.load( filename, sr=None )
    hop = np.floor( 0.003*fs )
    # Get beat locations
    _, beats = librosa.beat.beat_track( audioData, fs, hop_length=hop )
    # Convert beat locations to samples
    beatSamples = beats*hop
    # Get harmonic component of signal
    frameSize = 2**np.ceil( np.log2( .09*fs ) )
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
    dirs = [os.path.join( 'data/beatles', subdir) for subdir in os.listdir( 'data/beatles' )]
    for subdir in dirs:
        mp3Files = glob.glob( os.path.join( subdir, '*.mp3' ) )
        for mp3File in mp3Files:
            beats, semitrums = beatChromaLuma( mp3File )
            nameBase = os.path.splitext( mp3File )[0]
            np.save( nameBase + '-beats.npy', beats )
            np.save( nameBase + '-CL-magnitude.npy', semitrums )
    dirs = [os.path.join( 'data/uspop2002', subdir) for subdir in os.listdir( 'data/uspop2002' )]
    for subdir in dirs:
        mp3Files = glob.glob( os.path.join( subdir, '*.mp3' ) )
        for mp3File in mp3Files:
            beats, semitrums = beatChromaLuma( mp3File )
            nameBase = os.path.splitext( mp3File )[0]
            np.save( nameBase + '-beats.npy', beats )
            np.save( nameBase + '-CL-magnitude.npy', semitrums )

