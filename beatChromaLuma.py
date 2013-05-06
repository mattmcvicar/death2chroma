# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Compute beat-synchronous chroma-luma matrices
'''

# <codecell>

import librosa
import numpy as np
import separateHarmonicPercussive
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
        CLPatches - tensor of chroma-luma matrices, size nBeats x nOctaves x binPerOctave
    '''
    minNote = kwargs.get( 'minNote', 24 )
    nOctaves = kwargs.get( 'nOctaves', 7 )
    binsPerOctave = kwargs.get( 'binsPerOctave', 48 )
    log = kwargs.get( 'log', True )

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
    seperator = separateHarmonicPercussive.HarmonicPercussiveSeparator( spectrogram )
    harmonicData = librosa.istft( seperator.harmonicSpectrogram, n_fft=frameSize, hop_length=frameSize/4 )
    # Compute a chroma-luma matrix for each beat
    CLPatches = np.zeros( (beats.shape[0], nOctaves, binsPerOctave) )
    for n, (beatStart, beatEnd) in enumerate( zip( beatSamples[:-1], beatSamples[1:] ) ):
        # Grab audio samples within this beat
        beatSamples = harmonicData[beatStart:beatEnd]
        CLPatches[n] = chromaLuma.CLSpectrum( beatSamples, fs, minNote, nOctaves, binsPerOctave, log=log )
    return librosa.frames_to_time( beats, fs, hop ), CLPatches

# <codecell>

if __name__ == '__main__':
    import os
    import glob
    dirs = [os.path.join( 'mp3s-32k', subdir) for subdir in os.listdir( 'mp3s-32k' )]
    for subdir in dirs:
        mp3Files = glob.glob( os.path.join( subdir, '*.mp3' ) )
        for mp3File in mp3Files:
            beats, CLPatches = beatChromaLuma( mp3File )
            nameBase = os.path.splitext( mp3File )[0]
            np.save( nameBase + '-beats.npy', beats )
            np.save( nameBase + '-CL.npy', CLPatches )

