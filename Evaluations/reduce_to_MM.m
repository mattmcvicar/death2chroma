function [reduced_chords,reducedT_chords,chord_type_noMapping]=reduce_to_MM(CH_chords)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: 
%reduced_chords=reduce_to_MM(CH_chords)
%
%Convert a cell array of CH format chords to MM's representations (121 classes).
%Prerequisite: chord2quality.m in CH toolbox
%
% Inputs
%          - CH_chords. The unique chords in Christ Harte's format.
% 
% Outputs
%          - reduced_chords. The reduced MM's chords.
%          - reducedT_chords. The reduced MM's chords without bass (for KCB
%            model).
%          - chord_type_noMapping. The chord types that do not have a map
%            in the predefined library.
%
%---------------------------------------------
%Function created by M. McVicar
%Function revised by Y. Ni
%Intelligent Systems Lab
%University of Bristol
%U.K.
%2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


chord_type_map=containers.Map();
chord_type_noMapping=containers.Map();

%The 121 chord types
%Column 1
chord_type_map('N')='N';
chord_type_map('')='maj';
chord_type_map('min')='min';
chord_type_map('7')='7';
chord_type_map('min7')='min7';
chord_type_map('/5')='maj/5';
chord_type_map('maj')='maj';
chord_type_map('/3')='maj/3';
chord_type_map('maj7')='maj7';
chord_type_map('sus4')='maj';
chord_type_map('min/5')='min';
chord_type_map('maj6')='maj6';
chord_type_map('9')='7';
chord_type_map('min/b3')='min';
chord_type_map('/2')='maj';
chord_type_map('aug')='aug';
%Variation: (3,#5)=aug
chord_type_map('(3,#5)')='aug';
chord_type_map('/7')='maj';
chord_type_map('dim')='dim';
chord_type_map('(1)')='X';
chord_type_map('min/b7')='min7';
chord_type_map('/b7')='maj';
chord_type_map('6')='X';
chord_type_map('min6')='min';
chord_type_map('dim7')='dim';
chord_type_map('7(#9)')='7';
chord_type_map('/4')='maj';
chord_type_map('(1,5)')='maj';
chord_type_map('/6')='maj';
chord_type_map('7/3')='maj/3';
chord_type_map('min/6')='min';
chord_type_map('maj7/3')='maj/3';
chord_type_map('min7/b7')='min7';
chord_type_map('min/4')='min';
chord_type_map('maj9')='maj7';
chord_type_map('hdim7')='dim';
chord_type_map('min9')='min7';
chord_type_map('sus4(b7)')='maj';
chord_type_map('maj(9)')='maj';
chord_type_map('maj/9')='maj';
chord_type_map('7/b7')='7';
%Variation one: maj/b7=7/b7
chord_type_map('maj/b7')='7';
chord_type_map('min/2')='min';
chord_type_map('maj6/5')='maj/5';
chord_type_map('7/5')='maj/5';
chord_type_map('minmaj7/b3')='min';
chord_type_map('min(*b3)')='min';
chord_type_map('maj(9)/9')='maj';
chord_type_map('sus4/5')='maj/5';
chord_type_map('sus2')='maj';
chord_type_map('min7/4')='min7';
chord_type_map('min(9)')='min';
chord_type_map('sus4(2)')='maj';
chord_type_map('min(6)')='min';
chord_type_map('dim/b3')='dim';
chord_type_map('/9')='maj';
chord_type_map('(1,4)')='maj';
chord_type_map('min/7')='min';
chord_type_map('maj6/2')='maj';
%Column 2
chord_type_map('hdim7/b7')='dim';
%Variation: dim7/b7 = dim
chord_type_map('dim7/b7')='dim';
chord_type_map('min7/b3')='min7';
chord_type_map('maj(2)')='maj';
chord_type_map('min6/5')='min';
chord_type_map('9(*3)')='7';
chord_type_map('min7(4)/b7')='min7';
chord_type_map('maj(4)')='maj';
chord_type_map('6/2')='X';
chord_type_map('maj(*3)')='maj';
chord_type_map('minmaj7')='min';
chord_type_map('min7(9)')='min7';
chord_type_map('maj7(9)')='maj7';
chord_type_map('maj(9)/3')='maj/3';
chord_type_map('maj(#4)/5')='maj/5';
chord_type_map('7/2')='7';
chord_type_map('min7(4)')='min';
chord_type_map('maj7/5')='maj/5';
chord_type_map('maj(2)/2')='maj';
chord_type_map('maj(*1)/#1')='maj';
chord_type_map('9(11)')='7';
chord_type_map('minmaj7/5')='min';
chord_type_map('min7(2,*b3,4)')='min7';
chord_type_map('min6/6')='min';
chord_type_map('maj6/3')='maj/3';
chord_type_map('maj(b9)')='maj';
chord_type_map('maj(*5)')='maj';
chord_type_map('dim7/b3')='dim';
chord_type_map('dim/b5')='dim';
chord_type_map('7(b9)')='7';
chord_type_map('7(13)')='7';
chord_type_map('/b3')='maj';
chord_type_map('sus4(9)')='maj';
chord_type_map('sus4(2)/2')='maj';
chord_type_map('sus2(b7)')='min';
chord_type_map('min7(*5,b6)')='min7';
chord_type_map('min(2)')='min';
chord_type_map('min(*5)')='min';
chord_type_map('maj/2')='maj';
chord_type_map('maj(*1)/5')='maj/5';
chord_type_map('maj(#11)')='maj';
chord_type_map('dim7/b9')='dim';
chord_type_map('dim7/5')='dim';
chord_type_map('7sus')='X';
chord_type_map('/#4')='maj';
chord_type_map('(6)')='X';
chord_type_map('min7/5')='min7';
chord_type_map('min7(4)/5')='min7';
chord_type_map('min7(*b3)')='min7';
chord_type_map('min6/b3')='min';
chord_type_map('min/3')='min';
chord_type_map('min(b6)/5')='min';
chord_type_map('min(9)/b3')='min';
chord_type_map('maj6/b7')='maj6';
chord_type_map('maj/5')='maj/5';
chord_type_map('maj/3')='maj/3';
chord_type_map('maj(9)/6')='maj';
chord_type_map('maj(9)/5')='maj/5';
%Column 3
chord_type_map('hdim7/b3')='dim';
chord_type_map('dim7/2')='dim';
chord_type_map('aug/3')='aug';
chord_type_map('aug/#5')='aug';
chord_type_map('9/5')='maj/5';
chord_type_map('7/b3')='7';
chord_type_map('7/b2')='maj';
chord_type_map('7/#5')='7';
chord_type_map('4')='X';
chord_type_map('/b6')='maj';
chord_type_map('(1,5,9)')='min';
chord_type_map('(1,4,b7)')='7';
chord_type_map('(*5)')='X';
chord_type_map('sus4/4')='maj';
chord_type_map('sus')='maj';
chord_type_map('min7/7')='min7';
chord_type_map('min7(*5)/b7')='min7';
chord_type_map('min(4)')='min';
chord_type_map('min(*b3)/5')='min';
chord_type_map('min(*5)/b7')='min';
chord_type_map('min(*3)/5')='min';
chord_type_map('maj9(*7)')='maj7';
chord_type_map('maj7/7')='maj7';
chord_type_map('maj7/2')='maj7';
chord_type_map('maj7(*b5)')='maj7';
chord_type_map('maj7(*5)')='maj7';
chord_type_map('maj6(9)')='maj6';
chord_type_map('maj(13)')='maj';
chord_type_map('maj(11)')='maj';
chord_type_map('dim7/7')='dim';
chord_type_map('dim/b7')='dim';
chord_type_map('dim/5')='dim';
chord_type_map('aug(9,11)')='aug';
chord_type_map('9(*3,11)')='7';
chord_type_map('7(*5,13)')='7';
chord_type_map('(b6)')='X';
chord_type_map('(b3,5)')='min';
chord_type_map('(7)')='X';
chord_type_map('(5)')='X';
chord_type_map('(4,b7,9)')='X';
chord_type_map('(3)')='X';
chord_type_map('(1,b7)/b7')='X';
chord_type_map('(1,b7)')='X';
chord_type_map('(1,b3,4)/b3')='X';
chord_type_map('(1,b3)/b3')='X';
chord_type_map('(1,b3)')='X';
chord_type_map('(1,4,b5)')='X';
chord_type_map('(1,2,5,b6)')='X';
chord_type_map('(1,2,4)')='X';

%Patch (for new SALAMI chord dataset)
%  chord_type_map('1(b7)/b7')='X';
%  chord_type_map('1(#5)')='X';
%  chord_type_map('1')='maj'; %1 note is regard as major chord


%2. Get the chord information and map it using MM's chord dictionary
num_chord=length(CH_chords);
reduced_chords=cell(1,num_chord);
reducedT_chords=cell(1,num_chord);


for chord=1:num_chord
    %2.1 If it is an unknown chord
    if (CH_chords{chord}(1)=='X')
        reduced_chords{chord}='X';
        
    %2.2 Else, match the rootnote and the types
    else
        rootnote=getchordinfo(CH_chords{chord});
        
        if (length(rootnote)==length(CH_chords{chord}))
            chord_type='';
        else
            chord_type=CH_chords{chord}(length(rootnote)+1:end);
            chord_type=regexprep(chord_type,':','');
        end
        
        
        if (isKey(chord_type_map,chord_type))
            %Extract the type+bass
            reduced_chord_type=chord_type_map(chord_type);
            
            %Extract only the type
            temp_pos=strfind(reduced_chord_type,'/');
            
            if (strcmp(reduced_chord_type,'X'))
                reduced_chords{chord}='X'; %Undefined on the dictionary side
            else
                reduced_chords{chord}=[rootnote,':',reduced_chord_type];
                if (~isempty(temp_pos))
                    reducedT_chords{chord}=[rootnote,':',reduced_chord_type(1:temp_pos(1)-1)];
                end
            end
        else
            [~,~,~,~,success_flag]=getchordinfo(CH_chords{chord});
            if (success_flag)
                reduced_chords{chord}=CH_chords{chord};
            else
                warning(['In function reduce_to_MM.m: no such chord type in the dictionary: ', chord_type,'.']);
                reduced_chords{chord}='X';%CH_chords{chord};% %No match between the prediction and the dictionary
                chord_type_noMapping(chord_type)=1;
            end
        end
    end
end

return;





