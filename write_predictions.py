# Write predictions to file

# get filenames
import os
os.chdir('/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Training_Scripts')

pred_name = 'Beatles_CL_magnitude_fda_minmaj_bass_hmm_prediction_minmaj'
pred_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Brian_predicts/' + pred_name + '/'
output_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Predictions/' + pred_name + '/'


files = os.listdir(pred_dir)

# Load up the chord indices
import cPickle
chord_dict = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/Training_Scripts/dict_minmaj.p'
pkl_file = open(chord_dict, 'rb')
data = cPickle.load(pkl_file)

chord_classes = data[0]
chord_indices = data[1]

import numpy as np

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
                                           

  