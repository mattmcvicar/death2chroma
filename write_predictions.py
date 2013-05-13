# Write predictions to file

# get filenames
import os
os.chdir('/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Training_Scripts')

pred_name = 'USpop_raw_full_linear_minmaj'
pred_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Brian_predicts/' + pred_name + '/'
output_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Predictions/' + pred_name + '/'


files = os.listdir(pred_dir)

# Load up the chord indices
import numpy as np
np.load('/Users/mattmcvicar/Desktop/Work/New_chroma_features/chord_indices_minmaj.npy')

import sys
sys.path.append("../")
import print_ground_truth

for file in files:
  print 'Writing:' + file

  # Load annotation
  fullname = pred_dir + file
  print file
  P = np.load(fullname)
  
  # Convert to right format
  beat_times = [i[0] for i in P]
  Annotation = [int(i[1]) for i in P]
  
  # Write to file
  Pred_name = output_dir + file[:-len('pyc')] + 'lab'
  print_ground_truth.print_ground_truth(Annotation, beat_times, Pred_name, 
                                         chord_classes, chord_indices)
                                           

  