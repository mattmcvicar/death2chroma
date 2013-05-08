# Trains a model, given directory of chromaluma and GT

# Parameters
luma_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Beatles_luma/'
GT_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/chordlabs/'
alphabet = 'minmaj'

# get filenames
import os
os.chdir('/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Training_Scripts')

GT_files = os.listdir(GT_dir)
GT_files = [f for f in GT_files if os.path.splitext(f)[1] == '.lab']
n_songs = len(GT_files)

luma_files = os.listdir(luma_dir)
luma_feat_files = [f for f in luma_files if os.path.splitext(f)[0][-len('-CL'):] == '-CL']
luma_beat_files = [f for f in luma_files if os.path.splitext(f)[0][-len('-beats'):] == '-beats']

# Need the Evaluation scripts
import sys
sys.path.append("../Evaluations/Python_Eval/")
import reduce_chords

# For each training GT, first extract the chord labels so that we can map them
all_chords = []
for gt in GT_files:

  # Read file
  lines = open(GT_dir + gt).readlines()
  
  # Get chord labels
  chords = [chord.split()[2].rstrip() for chord in lines]  
  
  # Reduce them to alhabet
  reduced_chords = reduce_chords.reduce_chords(chords,alphabet)
  
  # Append to all chords
  all_chords.extend(list(set(reduced_chords)))
  
# Get unique chords
chord_indices = list(set(all_chords)) 

# Some of these will contain the same pitch classes. So make a map
chord_classes = dict()
for chord in chord_indices:
    
  # get notes
  chordnotes, bassnote, success = reduce_chords.chord2pitchclasses(chord)
  
  # if seen this combination before, add to dict, else init dict
  if tuple(chordnotes) in chord_classes:
    chordnotes.sort()  
    chord_classes[tuple(chordnotes)].append(chord)
  else:    
    chordnotes.sort()  
    chord_classes[tuple(chordnotes)] = [chord]
    
# Now we have a state model. For each GT file, map to the index in chord_classes 
# and then sample according to sample times
import numpy as np
chord_indices = chord_classes.keys()
for (index,gt) in enumerate(GT_files):
    
  # Read file
  lines = open(GT_dir + gt).readlines()
  
  # Get chord labels and times
  chords = [chord.split()[2].rstrip() for chord in lines]  
  starts = [np.double(chord.split()[0].rstrip()) for chord in lines]
  ends = [np.double(chord.split()[1].rstrip()) for chord in lines]
  
  # Reduce them to alhabet
  reduced_chords = reduce_chords.reduce_chords(chords,alphabet)

  # Map to integers
  chord_numbers = []
  for chord in reduced_chords:
    chordnotes, bassnote, success = reduce_chords.chord2pitchclasses(chord)
    chordnotes.sort()
    chord_tuple = tuple(chordnotes)
    chord_numbers.append(chord_indices.index(chord_tuple))
    
  # Load up the corresponding beat times
  beat_times = np.load(luma_dir + luma_beat_files[index])
  
  # For each beat, get the beat interval and find the dominant chord
  # in this region
  annotation_sample_times = np.vstack((starts,ends))   
    
  