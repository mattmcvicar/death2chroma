death2chroma
============

Experiments on the Beatles dataset to try and beat a chroma-based HMM using cleverer features

========
Contents
========

mp3s32k - 180 Beatles mp3s

chordlabs - 180 ground truth annotations, in the form <start_time> <end_time> <chord_label>
            Times are measured in seconds and have been manually aligned to the audio in 
            mp3s32k. Chord labels are in ascii Chris Harte format and contain many chord
            types. The basic form is root:(interval1,interval2,)/[bass], or root:shorthand.
            Examples include C:(1,#5,7,b9), Bb:(1,3,5)/F, D:min

chordlabs_minmaj - 180 ground truth annotations, reduced to their constituent major and minor
                   triads for convenience. 

create_minmaj_labs - scripts to convert chordlabs to chordlabs_minmaj in MATLAB

Predictions - folder to store predictions. Predictions should be in the same format as the 
              ground truth, with identical filenames with an optional extra extension. 
              Example: '07-Michelle_prediction.lab'  

Evaluations - Folder containing MATLAB scripts for evaluation. Evaluation can be performed
              currently at the minor/major, triad, and quad level.


