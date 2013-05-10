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
        