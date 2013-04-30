function [CH_chords]=transform_to_CHformat(MM_chords)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: 
%[CH_chords]=transform_to_CHformat(MM_chords)
% 
%1. This function transforms the prediction of Sonic Visualiser (Chordino) into
% CH's format. 
%2. Or transforms other chords (from web/new dataset/etc) into CH' format.
%
% INPUTS 
%MM_chords - the chords appeared.
%
% OUTPUTS 
%CH_chords - the chords in CH's format.
%
%---------------------------------------------
%Function created by Y. Ni
%Function revised by M. McVicar
%Intelligent Systems Lab
%University of Bristol
%U.K.
%2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%% PATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1. Type patch
% plain majors:
CH_form_types{1}={'','maj'};

% numerals ( dominant 7s,9s,11ths,13ths):
CH_form_types{2}={'11','11'};  % 11
CH_form_types{3}={'13','13'};  % 13
CH_form_types{4}={'2','sus4'}; % 2!!
CH_form_types{5}={'4','sus4'}; % 4!!
CH_form_types{6}={'5','(5)'}; % 5
CH_form_types{7}={'5+','(5)'}; % 5+ % doesnt like #5, or b6 use 5 for now?!
CH_form_types{8}={'6','maj6'}; % 6
CH_form_types{9}={'7','7'}; % 7
CH_form_types{10}={'7+','(3,#5,b7)'}; % 7+
CH_form_types{11}={'79','9'}; % 79!!
CH_form_types{12}={'7m','min7'}; % 7m!!
CH_form_types{13}={'7sus','(2,4,7)'}; % 7sus 
CH_form_types{14}={'7sus4','(4,7)'}; % 7sus4 (as above)
CH_form_types{15}={'9','9'}; % 9

% Adds and dims:
CH_form_types{16}={'add2','(2,3,5)'};  % add2
CH_form_types{17}={'add4','(2,4,5)'};  % add4
CH_form_types{18}={'add9','(3,5,9)'}; % add9
CH_form_types{19}={'dim','dim'}; % dim 

% Minors:
CH_form_types{20}={'m','min'};  % m
CH_form_types{21}={'m11','min11'};  % m11
CH_form_types{22}={'m4','min'}; % m4!!
CH_form_types{23}={'m5','(5)'}; % m5!!
CH_form_types{24}={'m5+','(b3,#5)'}; % m5+
CH_form_types{25}={'m5-','dim'}; % m5- 
CH_form_types{26}={'m6','min6'}; % m6
CH_form_types{27}={'m7','min7'}; % m7
CH_form_types{28}={'m7+','(b3,#5,b7)'}; % m7+
CH_form_types{29}={'m7m','min7'}; % m7m!!
CH_form_types{30}={'m9','(1,b3,5,b7,9)'}; % min9
CH_form_types{31}={'madd9','(b3,5,9)'}; % madd9

% Majors:
CH_form_types{32}={'maj9(#11)','(1,3,5,7,9,#11)'}; % maj9add#11
CH_form_types{33}={'maj7','maj7'}; % maj7
CH_form_types{34}={'maj9','maj9'}; % maj9

% Diminished symbol and Suspended:
CH_form_types{35}={'o','dim'}; % o
CH_form_types{36}={'sus','(2,4,5)'}; % sus
CH_form_types{37}={'sus2','sus2'}; % sus2
CH_form_types{38}={'sus4','sus4'}; % sus4
CH_form_types{39}={'sus9','sus2'}; % sus9

%%%%%%%% First patch:
CH_form_types{40}={'7#9','(3,5,b7,#9)'}; % 7#9
CH_form_types{41}={'o7','dim7'}; % diminished 7
CH_form_types{42}={'m2','min'}; % minor

%%%%%%%%% Second patch
CH_form_types{43}={'+','aug'}; % augmented
CH_form_types{44}={'0','dim'}; % augmented
CH_form_types{45}={'add','maj'}; % augmented
CH_form_types{46}={'7|','7'}; % augmented
CH_form_types{47}={'dim7','dim7'}; % augmented

%%%%%%%%%%Oasis Patch
CH_form_types{48}={'add6','(3,5,6)'};
CH_form_types{49}={'m7add4','(b3,5,b7,4)'};
% 
% %%%%%%%%% Patch to pass all beatles 17/09/10
CH_form_types{50}={'sus7','(4,5,b7)'};

% Patch to pass Queen 23/09/10
CH_form_types{51}={'3','maj'};
CH_form_types{52}={'aug','(1,3,#5)'};
CH_form_types{53}={'6+','maj6'};
CH_form_types{54}={'g','maj'};
CH_form_types{55}={'msus4','sus4'};
CH_form_types{56}={'sus6','(3,6)'};
CH_form_types{57}={'addy','maj'};
CH_form_types{58}={'msus2-d','sus2'};
CH_form_types{59}={'madd7','(b3,5,7)'};
CH_form_types{60}={'m9+','(b3,5,b7,#9)'};
CH_form_types{61}={'maj','maj'};
CH_form_types{62}={'min','min'};

%Patch to pass SALAMI dataset 15/03/12
CH_form_types{63}={'1','(1)'};
CH_form_types{64}={'1(#5)','(1,#5)'};
CH_form_types{65}={'1(11)','(1,11)'};
CH_form_types{66}={'1(11,9)','(1,9,11)'};
CH_form_types{67}={'1(13)','(1,13)'};
CH_form_types{68}={'1(3)','(1,3)'};
CH_form_types{69}={'1(3,7)','(1,3,7)'};
CH_form_types{70}={'1(b3)','(1,b3)'};	
CH_form_types{71}={'1(b3,b7,11,9)','(1,b3,b7,9,11)'};
CH_form_types{72}={'1(b5,11)','(1,b5,11)'};
CH_form_types{73}={'1(b5,b7,3)','(1,3,b5,b7)'};
CH_form_types{74}={'1(b7)','(1,b7)'};
CH_form_types{75}={'5(13)','(1,5,13)'};
CH_form_types{76}={'5(b13)','(1,5,b13)'};
CH_form_types{77}={'5(b7)','(1,5,b7)'};
CH_form_types{78}={'7(#11)','(1,3,5,b7,#11)'};
CH_form_types{79}={'7(#9)','(1,3,5,b7,#9)'};
CH_form_types{80}={'7(11)','(1,3,5,b7,11)'};
CH_form_types{81}={'7(b13)','(1,3,5,b7,b13)'};
CH_form_types{82}={'7(b13,#9)','(1,3,5,b7,#9,b13)'};
CH_form_types{83}={'7(b9)','(1,3,5,b7,b9)'};
CH_form_types{84}={'7(b9,#11)','(1,3,5,b7,b9,#11)'};
CH_form_types{85}={'7(b9,b13)','(1,3,5,b7,b9,b13)'};
CH_form_types{86}={'9(#11)','(1,3,5,b7,9,#11)'};
CH_form_types{87}={'9(13)','(1,3,5,b7,9,13)'};
CH_form_types{88}={'aug(7)','(1,3,#5,7)'};
CH_form_types{89}={'aug(b7)','(1,3,#5,b7)'};
CH_form_types{90}={'aug(b7,9)','(1,3,#5,b7,9)'};
CH_form_types{91}={'dim(b13)','(1,3,b5,b13)'};
CH_form_types{92}={'hdim7','hdim7'};
CH_form_types{93}={'maj(#11)','(1,3,5,#11)'};
CH_form_types{94}={'maj(#9)','(1,3,5,#9)'};
CH_form_types{95}={'maj(11)','(1,3,5,11)'};
CH_form_types{96}={'maj(9)','(1,3,5,9)'};
CH_form_types{97}={'maj(b9)','(1,3,5,b9)'};
CH_form_types{98}={'maj13','(1,3,5,7,9,13)'}; %Different interpretations?
CH_form_types{99}={'maj6','(1,3,5,6)'};
CH_form_types{100}={'maj6(7)','(1,3,5,6,7)'};
CH_form_types{101}={'maj6(7,#11)','(1,3,5,6,7,#11)'};
CH_form_types{102}={'maj6(9)','(1,3,5,6,9)'};
CH_form_types{103}={'maj6(b7)','(1,3,5,6,b7)'};
CH_form_types{104}={'maj6(b7,11)','(1,3,5,6,b7,11)'};
CH_form_types{105}={'maj7(#11)','(1,3,5,7,#11)'};
CH_form_types{106}={'maj7(b7)','(1,3,5,7,b7)'};
CH_form_types{107}={'maj9(13)','(1,3,5,7,9,13)'};
CH_form_types{108}={'maj9(13,#11)','(1,3,5,7,9,#11,13)'};
CH_form_types{109}={'min(11)','(1,b3,5,11)'};
CH_form_types{110}={'min(9)','(1,b3,5,9)'};
CH_form_types{111}={'min(b13)','(1,b3,5,b13)'};
CH_form_types{112}={'min11','(1,b3,5,b7,9,11)'}; %Different interpretations?
CH_form_types{113}={'min13','(1,b3,5,b7,9,13)'}; %Different interpretations?
CH_form_types{114}={'min6','(1,b3,5,6)'};
CH_form_types{115}={'min6(#11)','(1,b3,5,6,#11)'};
CH_form_types{116}={'min7','(1,b3,5,b7)'};
CH_form_types{117}={'min7(11)','(1,b3,5,b7,11)'};
CH_form_types{118}={'min7(b13)','(1,b3,5,b7,b13)'};
CH_form_types{119}={'min9','(1,b3,5,b7,9)'};
CH_form_types{120}={'min9(b13)','(1,b3,5,b7,9,b13)'};
CH_form_types{121}={'minmaj7','(1,b3,5,7)'};
CH_form_types{122}={'sus2(b7)','(1,2,5,b7)'};
CH_form_types{123}={'sus4(9)','(1,4,5,9)'};
CH_form_types{124}={'sus4(b7)','(1,4,5,b7)'};
CH_form_types{125}={'sus4(b7,9)','(1,4,5,b7,9)'};
CH_form_types{126}={'sus4(b7,9,#11)','(1,4,5,b7,9,#11)'};
CH_form_types{127}={'sus4(b7,9,13)','(1,4,5,b7,9,13)'};

%Patch to pass MIREX dataset 15/03/12
CH_form_types{128}={'(1,2,4)','(1,2,4)'};
CH_form_types{129}={'(1)','(1)'};
CH_form_types{130}={'(1,5)','(1,5)'};
CH_form_types{131}={'7(*5,13)','7'};
CH_form_types{132}={'7(13)','(1,3,5,b7,13)'};
CH_form_types{133}={'9(11)','(1,3,5,b7,9,11)'};
CH_form_types{134}={'sus4(2)','(1,2,4,5,9)'};
CH_form_types{135}={'min(4)','min(4)'};
CH_form_types{136}={'maj(*1)','maj(*1)'};
CH_form_types{137}={'9(*3,11)','9(*3,11)'};


%2. Bass patch
% Numerals
CH_bass_types{1}={'',''};
CH_bass_types{2}={'/11','/11'};
CH_bass_types{3}={'/13','/13'};
CH_bass_types{4}={'/13-','/b13'};
CH_bass_types{5}={'/4','/4'};
CH_bass_types{6}={'/4/9','/4'};
CH_bass_types{7}={'/5+','/b6'};   % can't parse #5, so do b6 instead
CH_bass_types{8}={'/5-','/b5'};
CH_bass_types{9}={'/6','/6'};
CH_bass_types{10}={'/7','/7'};
CH_bass_types{11}={'/7/9','/7'};
CH_bass_types{12}={'/9','/9'};
CH_bass_types{13}={'/9+','/#9'};
CH_bass_types{14}={'/9-','/b9'};
CH_bass_types{15}={'/9/11','/9'};

CH_bass_types{20}={'/b5','/b5'};
CH_bass_types{21}={'/b9','/b9'};

% The rest are notes.
CH_bass_types{16}={'/a','note'};
CH_bass_types{17}={'/a#','note'};
CH_bass_types{18}={'/ab','note'};
CH_bass_types{19}={'/b','note'};

CH_bass_types{22}={'/bb','note'};
CH_bass_types{23}={'/c','note'};
CH_bass_types{24}={'/c#','note'};
CH_bass_types{25}={'/d','note'};
CH_bass_types{26}={'/d#','note'};
CH_bass_types{27}={'/db','note'};
CH_bass_types{28}={'/e','note'};
CH_bass_types{29}={'/eb','note'};
CH_bass_types{30}={'/f','note'};
CH_bass_types{31}={'/f#','note'};
CH_bass_types{32}={'/g','note'};
CH_bass_types{33}={'/g#','note'};

%%%%%%%%%%% First patch:
CH_bass_types{34}={'/9/4','/9'};

%%%%%%%%%% second patch

%%%%  Oasis Patch
CH_bass_types{35}={'/7/2','/7'};

%%%%%%%%% Patch to pass all beatles 17/09/10
CH_bass_types{36}={'/#9','/#9'};

% Patch to pass Queen 23/09/10
CH_bass_types{37}={'/gb','note'};
CH_bass_types{38}={'/sus4/b9','/b9'};
CH_bass_types{39}={'/e#','note'};


%Patch to pass SALAMI dataset 15/03/12
CH_bass_types{40}={'/b7','/b7'};
CH_bass_types{41}={'/5','/5'};
CH_bass_types{42}={'/3','/3'};
CH_bass_types{43}={'/b3','/b3'};
CH_bass_types{44}={'/b7','/b7'};
CH_bass_types{45}={'/b11','/b11'};
CH_bass_types{46}={'/b13','/b13'};
CH_bass_types{47}={'/#11','/#11'};
CH_bass_types{48}={'/b1','/b1'};
CH_bass_types{49}={'/bb7','/bb7'};

%Patch to pass MIREX dataset 15/03/12
CH_bass_types{50}={'/2','/2'};
CH_bass_types{51}={'/#4','/#4'};
CH_bass_types{52}={'/b6','/b6'};
CH_bass_types{53}={'/#5','/#5'};
CH_bass_types{54}={'/b2','/b2'};
CH_bass_types{55}={'/#1','/#1'};


%%%%%%%%%%Construct the type/bass dictionary%%%%%%%%%%%%%%%%%%
CH_bass_dict=containers.Map();
CH_type_dict=containers.Map();
for i=1:length(CH_bass_types)
    CH_bass_dict(CH_bass_types{i}{1})=CH_bass_types{i}{2};
end

for i=1:length(CH_form_types)
    CH_type_dict(CH_form_types{i}{1})=CH_form_types{i}{2};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%3. For each Chord representation, decompose it into root/type/bass format
N=length(MM_chords);
CH_chords=cell(1,N);

for i=1:N
    temp_chord=lower(MM_chords{i});
    chordLen=length(temp_chord);

    %3.1 Get the root of the chord
    root=upper(temp_chord(1));
    % initalise the position of the suffix to be the second place
    suffix_pos=2;
    % If there's an accidental, add it, and increase the suffix position
    if chordLen>1 && (temp_chord(2)=='#' || temp_chord(2)=='b')
       root=[root temp_chord(2)]; 
       suffix_pos=3;
    end
    
    if (chordLen>=suffix_pos)
        if (temp_chord(suffix_pos)==':')
            suffix_pos=suffix_pos+1;
        end
    end
    
    %3.2 Get the type of the chord
    suffix_content=temp_chord(suffix_pos:end);
    slash_pos=strfind(suffix_content,'/');
    if slash_pos==1  % major chord (i.e. chord type='')
        type_content='maj';
    elseif ~isempty(slash_pos)  % found a slash, take up to slash (type)
        type_content=suffix_content(1:slash_pos-1);
    else     % found no slash, take all the content (no bass)
        type_content=suffix_content;
    end
    
    %Transform type to the CH format
    if (isKey(CH_type_dict,type_content))
        type_content=CH_type_dict(type_content);
    else
        %Check whether the type can be parsed by the toolbox, if so keep
        %it. Else give error.
        if (isempty(chord2notes(['C:',type_content])))
            error(['Unknown type: ' type_content,', keep the type for further analysis.']);
        end
    end
    
    %3.3 Get the bass
    bass_content=suffix_content(slash_pos:end);
    
    %Transform bass to the CH format
    if isempty(bass_content)
        temp_s='';
    elseif (isKey(CH_bass_dict,bass_content))
        temp_s=CH_bass_dict(bass_content);
    else
        warning(['Unknown bass: ' bass_content,', dispose the bass.']);
        temp_s='';
    end
    
    %Post-process the bass content
    if (strcmp(temp_s,''))
        bass_content='';
    elseif (strcmp(temp_s,'note')) %If it is a note
        % capatalise note (but keep the sharp # and flat b as lower case)
        if length(bass_content(2:end))==1
            note=upper(bass_content(2:end));
        else
            note=[upper(bass_content(2)) bass_content(3:end)];
        end  
        
        %Use CH toolbox to extract degree using root
        bass_content=['/' note2degree(note,root)];
    else % Or a numeral
        bass_content=temp_s;
    end
    
    %3.4 Return the CH format for this chord
    CH_chords{i}=[root,':',type_content,bass_content];
end

return;
    
    
   