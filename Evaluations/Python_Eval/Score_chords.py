# Directories for prediction and GT
GT_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/chordlabs/'
Predict_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Predictions/Matt_pretrained_complex/'

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
CP_Overlap = []
NP_Overlap = []
MIREX_Overlap = []
song_lengths = []

# Ignore these files
ignore = [150] # Revolution 9

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
  
  # Get the bass notes from the full chord labels
  GT_bass = reduce_chords.reduce_chords(GT_chords,'bass')
  P_bass = reduce_chords.reduce_chords(P_chords,'bass')

  # Now sample each chord at 1 kHz. Store pitch classes and sorted pitch classes
  # for NP and CP
  unique_chords = list(set(np.hstack((GT_chords_reduce,P_chords_reduce))))
  GT_chord_sample_CP = []; GT_chord_sample_NP = []; GT_chord_sample_MIREX = []
  P_chord_sample_CP = []; P_chord_sample_NP = []; P_chord_sample_MIREX = []
  
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
    
  # Still can have rounding effects, but only a maximum of 
  # nchords/1000 seconds...
  minlen = np.min([len(P_chord_sample_CP),len(GT_chord_sample_CP)])
  GT_chord_sample_CP = GT_chord_sample_CP[:minlen]; 
  P_chord_sample_CP = P_chord_sample_CP[:minlen]
  
  GT_chord_sample_NP = GT_chord_sample_NP[:minlen]
  P_chord_sample_NP = P_chord_sample_NP[:minlen]
 
  GT_chord_sample_MIREX = GT_chord_sample_MIREX[:minlen]
  P_chord_sample_MIREX = P_chord_sample_MIREX[:minlen]
  
  # Display?
  # import matplotlib.pyplot as plt
  # plt.imshow(np.vstack((np.array(GT_chord_sample),np.array(P_chord_sample))),interpolation="nearest",aspect="auto"); plt.show() 
  
  # CP = chords have the same notes
  
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

print '***************************'
print 'Mean Chord overlap: ' + str(round(np.mean(CP_Overlap),2)) + '%'
print 'Mean Note overlap: ' + str(round(np.mean(NP_Overlap),2)) + '%'
print 'Mean MIREX overlap: ' + str(round(np.mean(MIREX_Overlap),2)) + '%'
print 'Total Chord overlap: ' + str(round(np.dot(song_lengths,CP_Overlap),2)) + '%'
print 'Total Note overlap: ' + str(round(np.dot(song_lengths,NP_Overlap),2)) + '%'
print 'Total MIREX overlap: ' + str(round(np.dot(song_lengths,MIREX_Overlap),2)) + '%'
print '***************************'

