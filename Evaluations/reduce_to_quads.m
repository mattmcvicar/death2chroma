function reduced_chords=reduce_to_quads(CH_chords)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: 
%reduced_chords=reduce_to_quads(CH_chords)
%
% Convert a cell array of CH format chords to quad representations.
% Prerequisite: chord2quality.m in CH toolbox
%
% Inputs
%          - CH_chords. The unique chords in Christ Harte's format.
% 
% Outputs
%          - reduced_chords. The reduced major/minor/diminished/augmented/suspended chords.
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
    
    chordnotes=chord2notes(CH_chords{chord});
    
    % If it's NC, move on.
    if isempty(chordnotes)
        reduced_chords{chord}='N';
        continue
    end
    
    % look at the root and go one or two back to look for 7ths
    rootnote=getchordinfo(CH_chords{chord});
    minor7th=degree2note('b7',rootnote);
    major7th=degree2note('7',rootnote);
    
    min7flag=0;
    maj7flag=0;
    % search for them in chordnotes
    if ~isempty(strmatch(minor7th{1},chordnotes,'exact'))
        min7flag=1;
    end
    
    if ~isempty(strmatch(major7th{1},chordnotes,'exact'))
        maj7flag=1;
    end
    
    % Now get the rest of the chord
    quality=chord2quality(CH_chords{chord});
    
    switch(quality)
        
        case 0  % minor
            type='maj';
            
        case 1 % major
            type='min';
            
        case 2 % diminished
            type='dim';
            
        case 3 % augmented
            type='aug';
            
        case 4 % suspended
            type='sus';
            
    end % end switch
    
    % Get the degreelist for the triad
    reduced_chords{chord}=[rootnote ':(' shorthand2degrees(type)];
    
    % see if min7 or maj7 have been flagged, else close bracket
    if min7flag
        reduced_chords{chord}=[reduced_chords{chord} ',b7)'];
    elseif maj7flag
        reduced_chords{chord}=[reduced_chords{chord} ',7)'];
    else
        reduced_chords{chord}=[reduced_chords{chord} ')'];
    end
    
end % end chord