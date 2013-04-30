clear all
clc

% Path to rich labs
lab_path = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/chordlabs/';
labs = getAllFiles(lab_path);

% New directory (flat)
output_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/chordlabs_minmaj/';

% Pesky '.DS_Store'...
if labs{1}(end-length('DS_Store'):end) == '.DS_Store'
 labs = labs(2:end);
end

% Alphabet
alphabet = 0;
keytab = {'N', ...
          'C','C#','D','D#','E','F','F#','G','G#','A','A#','B', ...
          'C:min','C#:min','D:min','D#:min','E:min','F:min', ...
          'F#:min','G:min','G#:min','A:min','A#:min','B:min'};

% Read each file, then use DAn's parser
for f = 1:length(labs)

 % Open output file
 [dir,localname] = fileparts(labs{f});
 fID = fopen([output_dir localname '.lab'],'w');
 
 disp(['Writing: ' num2str(f) ' of ' num2str(length(labs)) ': ' localname])
 
 % Split each chord line by whitespace
 D = importdata(labs{f}); 
 for c = 1:length(D)
  split_data = regexp(D{c},'\s','split');
  
  start_time = split_data{1};
  end_time = split_data{2};
  chord = split_data{3};
  
  new_chord = normalize_labels({chord}, alphabet);
  new_chord_label = keytab{new_chord+1};
  
  % Write
  fprintf(fID,'%s',[start_time ' ']);
  fprintf(fID,'%s',[end_time ' ']);
  fprintf(fID,'%s\n',new_chord_label);  
 end
  
  fclose(fID);
end

