# Trains a model, given directory of chromaluma and GT

# Parameters
luma_dir_uspop = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/USpopmp3/uspop2002/'
GT_dir_uspop = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/USpoplabs_flat/'

luma_dir_beatles = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/beatles/'
GT_dir_beatles = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/chordlabs/'

# Alphabet
alphabet = 'minmaj'

# get filenames
import os
os.chdir('/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Training_Scripts')

GT_files = os.listdir(GT_dir_beatles); GT_files = [GT_dir_beatles + f for f in GT_files]
GT_files_USpop = os.listdir(GT_dir_uspop); GT_files_USpop = [GT_dir_uspop + f for f in GT_files_USpop]
GT_files.extend(GT_files_USpop)

GT_files = [f for f in GT_files if os.path.splitext(f)[1] == '.lab']
n_songs = len(GT_files)

luma_files = os.listdir(luma_dir_beatles); luma_files = [luma_dir_beatles + f for f in luma_files]
luma_files_USpop = os.listdir(luma_dir_uspop); luma_files_USpop = [luma_dir_uspop + f for f in luma_files_USpop]
luma_files.extend(luma_files_USpop)

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
  lines = open(gt).readlines()
  
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
print 'getting information on chords...'

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
import sample_gt

Annotations = []
filenames = []
Beat_times = []
for (index,gt) in enumerate(GT_files):
  
  print 'Getting chord annotation for song: ' + str(index+1) + ' of ' + str(n_songs)
  # Read file
  lines = open(gt).readlines()
  
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
  beat_times = np.load(luma_beat_files[index])
  
  # Sample the beats
  annotation_sample_times = np.vstack((starts,ends))
  n_states = chord_indices.index(())
  
  # Sample annotations. Strip the first label to make same length as beat_times
  Annotations.append(sample_gt.sample_annotations_beat(chord_numbers,annotation_sample_times,beat_times,n_states)[1:])
  Beat_times.append(beat_times)  
  
  filenames.append(gt[:-len('.lab')]+'-labels-'+alphabet)
  
# Save the annotations
beatles_savedir = './beatles_indices/'
uspop_savedir = './uspop_indices/'

import re

for index,file in enumerate(filenames):
    
  save_name = re.split('/',file)[-1]
  if file[:len(GT_dir_beatles)] == GT_dir_beatles:
    np.save(beatles_savedir + save_name, Annotations[index])
  else:
    np.save(uspop_savedir + save_name, Annotations[index])
      
# And the dictionary
import cPickle
cPickle.dump((chord_classes, chord_indices), open( "dict_" + alphabet + ".p", "wb" ) )