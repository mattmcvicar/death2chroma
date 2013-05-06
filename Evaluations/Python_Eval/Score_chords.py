# Directories for prediction and GT
GT_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/chordlabs/'
Predict_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Predictions/Matt_pretrained_minmaj/'

# Is something appended to the predictions? (ie '_prediction')?
appended = '_prediction.txt'

# Get filenames
import os
GT_files = os.listdir(GT_dir)
Predict_files = os.listdir(Predict_dir)

# Only process the right file types
file_ext = '.lab'
GT_files = [f for f in GT_files if os.path.splitext(f)[1] == file_ext]
Predict_files = [f for f in Predict_files if os.path.splitext(f)[1] == file_ext]


# Alphabet
alphabet = 'minmaj';

# Store results
Overlap = []
song_lengths = []

# Main loop
import numpy as np
import reduce_chords
for (index,GT_file) in enumerate(GT_files):

  print 'Evaluating song: ' + str(index + 1) + ' of ' + str(len(GT_files)) + ': ' + GT_file

  # Get expected Prediction name
  localname,extension = os.path.splitext(GT_file)
  Predict_name = Predict_dir + localname + appended

  # Read GT and Prediction
  GT = open(GT_dir + GT_file).readlines()
  P = open(Predict_name).readlines()

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

  # Now sample each chord at 1 kHz
  unique_chords = list(set(np.hstack((GT_chords_reduce,P_chords_reduce))))
  GT_chord_sample = []
  P_chord_sample = []
  for index,chord in enumerate(GT_chords_reduce):
    sym = unique_chords.index(chord)
    duration = int(round(1000*(GT_end_times[index] - GT_start_times[index])))
    GT_chord_sample.extend(np.tile(sym,duration))

  for index,chord in enumerate(P_chords_reduce):
    sym = unique_chords.index(chord)
    duration = int(round(1000*(P_end_times[index] - P_start_times[index])))
    P_chord_sample.extend(np.tile(sym,duration))

  # Still can have rounding effects, but only a maximum of 
  # nchords/1000 seconds...
  minlen = np.min([len(P_chord_sample),len(GT_chord_sample)])
  GT_chord_sample = GT_chord_sample[:minlen]
  P_chord_sample = P_chord_sample[:minlen]

  # Display?
  # import matplotlib.pyplot as plt
  # plt.imshow(np.vstack((np.array(GT_chord_sample),np.array(P_chord_sample))),interpolation="nearest",aspect="auto"); plt.show() 
  
  # Finally, output score
  correct = [GT_chord_sample[i] == p for (i,p) in enumerate(P_chord_sample)]
  Overlap.append(100*np.mean(correct))
  song_lengths.append(minlen)
  
song_lengths = np.true_divide(song_lengths,np.sum(song_lengths))

print '**********************'
print 'Mean Overlap: ' + str(np.mean(Overlap)) + '%'
print 'Total Overlap: ' + str(np.dot(song_lengths,Overlap)) + '%'
print '**********************'

