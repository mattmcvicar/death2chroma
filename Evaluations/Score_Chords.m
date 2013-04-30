clear all
clc

% Directories for prediction and GT
GT_dir = '../chordlabs_minmaj/';
Predict_dir = '../Predictions/Matt_pretrained_minmaj/';

% Path for toolbox
addpath(genpath('../'))

% Is something appended to the predictions? (ie '_prediction')?
appended = '_prediction.txt';

% Get filenames
GT_files = getAllFiles(GT_dir);
Predict_files = getAllFiles(Predict_dir);

% Alphabet
alphabet = 'minmaj';

% Store results
Overlap = [];
song_lengths = [];

% Main loop
for i=1:length(GT_files)

  disp(['Evaluating song: ' num2str(i) ' of ' num2str(length(GT_files)) ': ' GT_files{i}])

  % Get expected Prediction name
  [trash,localname,extension] = fileparts(GT_files{i});
  Predict_name = [Predict_dir localname appended];

  % Read GT and Prediction
  f_GT = fopen(GT_files{i}); GT = textscan(f_GT,'%s%s%s'); fclose(f_GT);
  f_P = fopen(Predict_name); P = textscan(f_P,'%s%s%s'); fclose(f_P);

  GT_chords = GT{3};
  P_chords = P{3};

  GT_times = [];
  for t=1:length(GT{1})
    GT_times = [GT_times ; [str2num(GT{1}{t}) str2num(GT{2}{t})]]; 
  end

  P_times = [];
  for t=1:length(P{1})
    P_times = [P_times ; [str2num(P{1}{t}) str2num(P{2}{t})]];
  end

  % Fix prediction to length of GT
  if P_times(end,end) < GT_times(end,end)
    P_times = [P_times ; [P_times(end-1,end) GT_times(end,end)]];
    P_chords = [P_chords ; 'N'];
  elseif P_times(end,end) > GT_times(end,end)
    bad_rows = find(P_times(:,1) > GT_times(end,end));
    P_times(bad_rows,:) = [];
    P_chords(bad_rows) = [];
    P_times(end,end) = GT_times(end,end); 
  else
    % Fine
  end

  % Sample so they're the same alphabet
  switch alphabet 
    case 'minmaj'
      GT_chords = reduce_to_minmaj(GT_chords);
      P_chords = reduce_to_minmaj(P_chords);
    case 'triads'
      GT_chords = reduce_to_triads(GT_chords);
      P_chords = reduce_to_triads(P_chords);
    case 'quads'
      GT_chords = reduce_to_quads(GT_chords);
      P_chords = reduce_to_quads(P_chords);
  end
 
  % Now sample each chord at 1 kHz
  unique_chords = unique([GT_chords P_chords]);
  GT_chord_sample = [];
  P_chord_sample = [];
  for t=1:length(GT_chords)
    sym = strmatch(GT_chords{t},unique_chords);
    duration = round(1000*(GT_times(t,2) - GT_times(t,1)));
    GT_chord_sample = [GT_chord_sample repmat(sym,[1, duration])];
  end

  for t=1:length(P_chords)
    sym = strmatch(P_chords{t},unique_chords);
    duration = round(1000*(P_times(t,2) - P_times(t,1)));
    P_chord_sample = [P_chord_sample repmat(sym,[1, duration])];
  end

  % Still can have rounding effects, but only a maximum of 
  % nchords/1000 seconds...
  minlen = min([length(P_chord_sample) length(GT_chord_sample)]);
  GT_chord_sample = GT_chord_sample(1:minlen);
  P_chord_sample = P_chord_sample(1:minlen);

  % Finally, output score
  Overlap(i) = 100*mean(eq(GT_chord_sample,P_chord_sample));
  song_lengths(i) = minlen;
end

song_lengths = song_lengths/sum(song_lengths);

disp('***********')
disp(['Mean Overlap: ' num2str(mean(Overlap)) '%'])
disp(['Total Overlap: ' num2str(song_lengths*Overlap') '%'])
disp('***********')

