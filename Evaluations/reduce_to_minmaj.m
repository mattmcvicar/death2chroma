function reduced_chords=reduce_to_minmaj(CH_chords)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: 
%reduced_chords=reduce_to_minmaj(CH_chords)
%
%Convert a cell array of CH format chords to minor/maj representations.
%Prerequisite: chord2quality.m in CH toolbox
%
% Inputs
%          - CH_chords. The unique chords in Christ Harte's format.
% 
% Outputs
%          - reduced_chords. The reduced major/minor chords.
%
%---------------------------------------------
%Function created by M. McVicar
%Function revised by Y. Ni
%Intelligent Systems Lab
%University of Bristol
%U.K.
%2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


reduced_chords=cell(1,length(CH_chords));

for chord=1:length(CH_chords)
    
    quality=chord2quality(CH_chords{chord});
    
    switch(quality)
        
        case 0  % minor
            type='maj';
            
        case 1 % major
            type='min';
            
        case 2 % diminished
            type='min';
            
        case 3 % augmented
            type='maj';
            
        case 4 % suspended
            type='maj';
            
    end % end switch
    
    % get rootnote and append type
    rootnote=getchordinfo(CH_chords{chord});
    
    reduced_chords{chord}=[rootnote ':' type];
    
end % end chord





