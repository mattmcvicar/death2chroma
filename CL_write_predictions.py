# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Code to make .lab from beat-labels

# <codecell>

def print_ground_truth(annotation, beat_times, filename, chord_classes, chord_indices):
    
  # Open file for editing
  file = open(filename,'wb')
  
  # Append a fake last beat
  beat_times = list(beat_times)
  beat_times.append(beat_times[-1] + 0.1)
  for index,chord in enumerate(annotation):
      
    # Get chord text. The first one will do (evaulation deals with that)
    chord_symbol = chord_classes[chord_indices[chord]][0]
    file.write(str(beat_times[index]) + ' ' + str(beat_times[index+1]) + ' ' + 
         chord_symbol + '\n')

# <codecell>

# Write predictions to file

# get filenames
import os
#os.chdir('/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Training_Scripts')

pred_name = ''
pred_dir = '/home/bmcfee/git/death2chroma/data/predictions/' + pred_name + '/'
output_dir = '/home/bmcfee/git/death2chroma/data/predictions_lab/' + pred_name + '/'


files = os.listdir(pred_dir)

# Load up the chord indices
import cPickle
chord_dict = '/home/bmcfee/git/death2chroma/data/dict_minmaj.p'
pkl_file = open(chord_dict, 'rb')
data = cPickle.load(pkl_file)

chord_classes = data[0]
chord_indices = data[1]


NOCHORD = filter(lambda z: not chord_indices[z], range(len(chord_indices)))[0]
print NOCHORD

import numpy as np

import sys
sys.path.append("../")
#import print_ground_truth

# <codecell>

for file in files:
  print 'Writing:' + file

  # Load annotation
  fullname = pred_dir + file
  print file
  P = np.load(fullname)
  
  # Convert to right format
  beat_times = [i[0] for i in P]
  Annotation = [int(i[1]) for i in P]
  
  
  # Pad in no-chord
  if i[0] != 0.0:
    beat_times.insert(0, 0)
    Annotation.insert(0, NOCHORD)
    
  # Write to file
  Pred_name = os.path.join(output_dir, file[:-len('.pyc')]) + '.lab'
  print_ground_truth(Annotation, beat_times, Pred_name, chord_classes, chord_indices)

# <codecell>

import os
import numpy as np
import reduce_chords
import cPickle
import re

# Directories for prediction and GT
GT_dir = '/home/bmcfee/git/death2chroma/data/beatles/'
#Predict_dir = '/home/bmcfee/git/death2chroma/data/predictions_lab'
#appended = '-CL-magnitude-fda_minmaj_bass-hmm-prediction-minmaj.lab'

Predict_dir = '/home/bmcfee/git/death2chroma/data/Matt_predictions/'
appended = '_prediction.txt'
# Is something appended to the predictions? (ie '_prediction')?


# Get filenames

GT_files = os.listdir(GT_dir)
Predict_files = os.listdir(Predict_dir)

GT_files.sort()
Predict_files.sort()


# Only process the right file types
file_ext = '.lab'
GT_files = [f for f in GT_files if os.path.splitext(f)[1] == file_ext]
Predict_files = [f for f in Predict_files if os.path.splitext(f)[1] == file_ext]


# Is something appended?
#appended = Predict_files[0][len(GT_files[0][:-len(file_ext)]):-len(file_ext)] + file_ext

# Alphabet
alphabet = 'minmaj';

# Dictionary (not explicity used to score, just for visualisation)

#chord_dict = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Training_Scripts/dict_minmaj.p'
#pkl_file = open(chord_dict, 'rb')
#data = cPickle.load(pkl_file)

#chord_classes = data[0]
#chord_indices = data[1]

# Store results
CP_Overlap = []
NP_Overlap = []
MIREX_Overlap = []
song_lengths = []

# Store sampled sequences
Display_P = []
Display_GT = []
 
# Ignore these files
#ignore = [150] # Revolution 9 for Beatles
ignore = [] # USpop use all

# Main loop

for (index,GT_file) in enumerate(GT_files):

  print 'Evaluating song: ' + str(index + 1) + ' of ' + str(len(GT_files)) + ': ' + GT_file

  # Get expected Prediction name
  localname, extension = os.path.splitext(GT_file)
  Predict_name = os.path.join(Predict_dir, localname + appended)
    
  # Read GT and Prediction
  GT = open(os.path.join(GT_dir, GT_file)).readlines()
  P  = open(Predict_name).readlines()

  GT_chords = []
  GT_start_times = []; GT_end_times = []
  for line in GT:
    start_time,end_time,chord = line.split() 
    
    GT_chords.append(chord)
    GT_start_times.append(np.double(start_time)) 
    GT_end_times.append(np.double(end_time)) 

  
  P_chords = []
  P_start_times = []; P_end_times = []
  for line in P:
    start_time,end_time,chord = line.split() 
    
    P_chords.append(chord)
    P_start_times.append(np.double(start_time)) 
    P_end_times.append(np.double(end_time)) 

  # Fix prediction to length of GT
  if P_end_times[-1] < GT_end_times[-1]:
    P_start_times.append(P_start_times[-1])  
    P_end_times.append(GT_end_times[-1])
    P_chords.append('N')
  elif P_end_times[-1] > GT_end_times[-1]:  
    remove_rows = [i for (i,t) in enumerate(P_start_times) if t > GT_end_times[-1] or P_end_times[i] > GT_end_times[-1]]
    
    # Remove them
    del P_start_times[remove_rows[0]:]
    del P_end_times[remove_rows[0]:]
    del P_chords[remove_rows[0]:]
  else:
    # Fine
    pass

  
    
  # Sample so they're the same alphabet 
  GT_chords_reduce = reduce_chords.reduce_chords(GT_chords,alphabet)
  P_chords_reduce = reduce_chords.reduce_chords(P_chords,alphabet)

  # Get the bass notes from the full chord labels
  GT_bass = reduce_chords.reduce_chords(GT_chords,'bass')
  P_bass = reduce_chords.reduce_chords(P_chords,'bass')

  # Now sample each chord at 1 kHz. Store pitch classes and sorted pitch classes
  # for NP and CP
  #unique_chords = list(set(np.hstack((GT_chords_reduce,P_chords_reduce))))
  GT_chord_sample_CP = []; GT_chord_sample_NP = []; GT_chord_sample_MIREX = []
  P_chord_sample_CP = []; P_chord_sample_NP = []; P_chord_sample_MIREX = []
  
  # Also for display purposes
  GT_chord_sample_disp = []; P_chord_sample_disp = []
  for c_index,chord in enumerate(GT_chords_reduce):
    duration = int(round(1000*(GT_end_times[c_index] - GT_start_times[c_index])))
       
    # Need bass symbol for CP, first two notes for MIREX
    sym = reduce_chords.chord2pitchclasses(chord)[0]
    bass_sym = GT_bass[c_index]
    MIREX_sym = reduce_chords.chord2pitchclasses(GT_chords[c_index])[0]
    if len(MIREX_sym) < 2:
      pass
    else:
      MIREX_sym = MIREX_sym[:2]
        
    GT_chord_sample_NP.extend([sym]*duration)
    GT_chord_sample_CP.extend([[sym, bass_sym]]*duration)
    GT_chord_sample_MIREX.extend([MIREX_sym]*duration)
    
    # Look up chord index for visualisation
    for k in chord_classes:
      if chord in chord_classes[k]:   
        GT_chord_sample_disp.extend([chord_indices.index(k)]*duration)
        break
   
  for c_index,chord in enumerate(P_chords_reduce):
    duration = int(round(1000*(P_end_times[c_index] - P_start_times[c_index])))
      
    # Need bass symbol for CP, first two notes for MIREX
    sym = reduce_chords.chord2pitchclasses(chord)[0]
    bass_sym = P_bass[c_index]
    MIREX_sym = reduce_chords.chord2pitchclasses(P_chords[c_index])[0]
    if len(MIREX_sym) < 2:
      pass
    else:
      MIREX_sym = MIREX_sym[:2]
        
    P_chord_sample_NP.extend([sym]*duration)
    P_chord_sample_CP.extend([[sym, bass_sym]]*duration)
    P_chord_sample_MIREX.extend([MIREX_sym]*duration)

    # Look up chord index for visualisation
    for k in chord_classes:
      if chord in chord_classes[k]:   
        P_chord_sample_disp.extend([chord_indices.index(k)]*duration)
        break
    
  # Still can have rounding effects, but only a maximum of 
  # nchords/1000 seconds..
  minlen = np.min([len(P_chord_sample_CP),len(GT_chord_sample_CP)])
  GT_chord_sample_CP = GT_chord_sample_CP[:minlen] 
  P_chord_sample_CP = P_chord_sample_CP[:minlen]
  
  minlen = np.min([len(P_chord_sample_NP),len(GT_chord_sample_NP)])
  GT_chord_sample_NP = GT_chord_sample_NP[:minlen]
  P_chord_sample_NP = P_chord_sample_NP[:minlen]
 
  minlen = np.min([len(P_chord_sample_MIREX),len(GT_chord_sample_MIREX)])
  GT_chord_sample_MIREX = GT_chord_sample_MIREX[:minlen]
  P_chord_sample_MIREX = P_chord_sample_MIREX[:minlen]
  
  minlen = np.min([len(P_chord_sample_disp),len(GT_chord_sample_disp)])
  GT_chord_sample_disp = GT_chord_sample_disp[:minlen]
  P_chord_sample_disp = P_chord_sample_disp[:minlen]
    
  
  # Store
  Display_P.append(P_chord_sample_disp)  
  Display_GT.append(GT_chord_sample_disp)  
  
  # Display?
  #import matplotlib.pyplot as plt
  #print len(GT_chord_sample_disp)
  #figure(figsize=(16,10))
  #plt.imshow(np.vstack((np.array(GT_chord_sample_disp),np.array(P_chord_sample_disp))),interpolation="nearest",aspect="auto")
  #yticks(range(2), ['True', 'Predicted'])
  #plt.show() 
  
  # Finally, output score
  CP_correct = [GT_chord_sample_CP[i] == p for (i,p) in enumerate(P_chord_sample_CP)]
  NP_correct = [GT_chord_sample_NP[i] == p for (i,p) in enumerate(P_chord_sample_NP)]
  MIREX_correct = [GT_chord_sample_MIREX[i] == p for (i,p) in enumerate(P_chord_sample_MIREX)]
  
  CP_Overlap.append(100*np.mean(CP_correct))
  NP_Overlap.append(100*np.mean(NP_correct))
  MIREX_Overlap.append(100*np.mean(MIREX_correct))
  song_lengths.append(minlen)
  
  
# Remove some songs
CP_Overlap = [o for index,o in enumerate(CP_Overlap) if index not in ignore]
NP_Overlap = [o for index,o in enumerate(NP_Overlap) if index not in ignore]
MIREX_Overlap = [o for index,o in enumerate(MIREX_Overlap) if index not in ignore]

song_lengths = [s for index,s in enumerate(song_lengths) if index not in ignore]  
  
# Normalise song lengths  
song_lengths = np.true_divide(song_lengths,np.sum(song_lengths))

# Display

Model = re.split('/',Predict_dir)[-2]
print '***************************'
print Model + ' - ' + alphabet
print '---------------------------'
print 'Mean Chord overlap: ' + str(round(np.mean(CP_Overlap),2)) + '%'
print 'Mean Note overlap: ' + str(round(np.mean(NP_Overlap),2)) + '%'
print 'Mean MIREX overlap: ' + str(round(np.mean(MIREX_Overlap),2)) + '%'
print 'Total Chord overlap: ' + str(round(np.dot(song_lengths,CP_Overlap),2)) + '%'
print 'Total Note overlap: ' + str(round(np.dot(song_lengths,NP_Overlap),2)) + '%'
print 'Total MIREX overlap: ' + str(round(np.dot(song_lengths,MIREX_Overlap),2)) + '%'
print '***************************'

# <headingcell level=1>

# Evaluation

# <codecell>

import cPickle as pickle

# <codecell>

with open('/home/bmcfee/git/death2chroma/data/performance_train_matt.pickle', 'w') as f:
#with open('/home/bmcfee/git/death2chroma/data/performance_train_hmm_cl_fda_minmaj.pickle', 'w') as f:
    pickle.dump({'Display_P': Display_P, 'Display_GT': Display_GT, 'NP_Overlap': NP_Overlap, 'GT_files': GT_files}, f, protocol=-1)

# <codecell>

def loadpickle(fname):
    with open(fname, 'r') as f:
        D = pickle.load(f)
    return D

# <codecell>

D_matt = loadpickle('/home/bmcfee/git/death2chroma/data/performance_train_matt.pickle')
D_cl = loadpickle('/home/bmcfee/git/death2chroma/data/performance_train_hmm_cl_fda_minmaj.pickle')

# <codecell>

D_matt['NP_Overlap'] = np.array(D_matt['NP_Overlap'])
D_matt['Display_P'] = map(np.array, D_matt['Display_P'])
D_cl['NP_Overlap'] = np.array(D_cl['NP_Overlap'])
D_cl['Display_P'] = map(np.array, D_cl['Display_P'])

# <codecell>

bads = np.argwhere(D_matt['NP_Overlap'] < 50)
[D_matt['GT_files'][i] for i in bads]

# <codecell>

print len(D_cl['Display_P'][53]), len(D_matt['Display_P'][53]), len(D_cl['Display_GT'][53]), len(D_matt['Display_GT'][53])

# <codecell>

#err = bads[0]
#err = 53
err = 0

# <codecell>

figure(figsize=(20, 5))

M = np.zeros_like(D_cl['Display_GT'][err])
M[:D_matt['Display_P'][err].size] = D_matt['Display_P'][err]
                          
imshow(np.vstack([D_cl['Display_GT'][err], D_cl['Display_P'][err], M.reshape((1,-1))]), aspect='auto', interpolation='nearest')
plt.colorbar()
yticks(range(3), ['True',  'CL', 'Matt'])
pass

# <codecell>

plot(map(len, D_matt['Display_P']))
plot(map(len, D_cl['Display_P']))

# <codecell>

D_matt.keys()

# <codecell>

figure(figsize=(20, 4))
plot(sorted(D_matt['NP_Overlap']))
plot(sorted(D_cl['NP_Overlap']))
#plot(D_matt['Display_P'][53])
#plot(D_cl['Display_P'][53])
legend(['Matt', 'CL'])

