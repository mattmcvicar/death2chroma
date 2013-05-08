def sample_annotations_beat(annotations,annotation_sample_times,sample_times,numStates):

  # 1. Initialisation 
  import numpy as np
  number_samples = annotation_sample_times.shape[1]
  number_windows = len(sample_times)
  sampled = np.zeros((1,number_windows+1)) # Store the annotations of a song
  t_anns = 1
  t_prev_anns = 1
  t_sample = 1
    
    
  # 1.1 For the first frame, if it is less than the start time, then no
  # chord
  while (sample_times[t_sample - 1] < annotation_sample_times[0][0]):
    sampled[t_sample - 1] = numStates
    t_sample = t_sample + 1
    
  # 1.2 Assure that t_sample falls in a chord region
  while (annotation_sample_times[t_anns-1,1] < sample_times[t_sample-1]):
        t_anns = t_anns + 1
    
  # 2. go though all time grid
  while (t_sample <= number_windows and t_anns <= number_samples):
    # 2.1 If TS(ts)<TA(ta), then this frame falls in this chord region
    if (sample_times[t_sample - 1] <= annotation_sample_times[t_anns][1]):
      # A. if the interval between two beats has more than 1 chord
      if (t_prev_anns < t_anns):
        # Majority vote
        countInterval = 1                   
        intervalC = np.zeros((1,t_anns-t_prev_anns+1))
                
        # First interval
        if t_sample == 1:
          intervalC[countInterval - 1] = annotation_sample_times[t_prev_anns - 1][1] - annotation_sample_times[t_prev_anns - 1][0]
        else:
          intervalC[countInterval - 1] = annotation_sample_times[t_prev_anns - 1][1] - sample_times[t_sample - 2];
                
        countInterval=countInterval + 1
                
        # Between intervals
        for j in range(t_prev_anns+1,t_anns):                      
          intervalC[countInterval - 1] = annotation_sample_times[j - 1][1] - annotation_sample_times[j - 1][0]                        
          countInterval = countInterval + 1                   
                
        # Last interval
        intervalC[countInterval - 1] = sample_times[t_sample - 1] - annotation_sample_times[t_anns - 1][0]      
                
        # Majority vote
        maxInterval = np.max(intervalC)
        maxIndex = np.argmax(intervalC)
        sampled[t_sample - 1] = annotations[t_prev_anns -1 + maxIndex - 1]
        t_prev_anns = t_anns
        
      # B. if the interval between two beats falls in 1 chord
      else:
        sampled[t_sample - 1] = annotations[t_anns - 1]
      
      t_sample=t_sample + 1
        
    # 2.2 Else, find the chord interval this beat falls in
    else:
      while (t_anns <= number_samples and annotation_sample_times[t_anns - 1][1] < sample_times[t_sample - 1]):
        t_anns = t_anns + 1
    
  # 3. if there are still samples left, assign no chord
  if t_sample <= number_windows:
    sampled[t_sample-1:] = numStates
    
  if (t_anns == number_samples): # The last chord after final beats
    sampled[number_windows] = annotations[t_anns - 1]
  elif t_anns < number_samples:
    countInterval = 1                    
    intervalC = np.zeros((1,number_samples-t_anns+1))
    
    # Majority vote
    intervalC[countInterval - 1] = annotation_sample_times[t_anns-1][1] - sample_times[number_windows - 1]
    countInterval = countInterval + 1
    for j in range(t_anns + 1, number_samples + 1):
      intervalC[countInterval - 1] = annotation_sample_times[j - 1][1] - annotation_sample_times[j - 1][0]
      countInterval = countInterval + 1

    maxIndex = np.argmax(intervalC)
    sampled[number_windows] = annotations[t_anns - 1 + maxIndex - 1]
             
  # 4. return the annotation samples
  return sampled
