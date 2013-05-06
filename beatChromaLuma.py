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
        minNote - lowest note to consider, default 24
        nOctaves - number of octaves to include in the chroma-lumas, default 7
        binsPerOctave - number of bins in each octave in the chroma-luma, default 48
        log - take log-chroma? default True
    Output:
        beatTimes - vector of beat locations, in seconds, size nBeats
        semitrums - matrix of per-beat semitrums, size nBeats x binsPerOctave*nOctaves
    '''

    binsPerOctave = kwargs.get( 'binsPerOctave', 48 )
    nOctaves = kwargs.get( 'nOctaves', 6 )

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
    harmonicSpectrogram, _ = librosa.hpss.hpss_median( np.abs( spectrogram ), win_H=13, p=3 )
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
    dirs = [os.path.join( 'mp3s-32k', subdir) for subdir in os.listdir( 'mp3s-32k' )]
    for subdir in dirs:
        mp3Files = glob.glob( os.path.join( subdir, '*.mp3' ) )
        for mp3File in mp3Files:
            beats, semitrums = beatChromaLuma( mp3File )
            nameBase = os.path.splitext( mp3File )[0]
            np.save( nameBase + '-beats.npy', beats )
            np.save( nameBase + '-CL.npy', semitrums )

