function reduced_chords=reduce_to_triads(CH_chords)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: 
%reduced_chords=reduce_to_triads(CH_chords)
%
%Convert a cell array of CH format chords to triad representations.
%Prerequisite: chord2quality.m in CH toolbox
%
% Inputs
%          - CH_chords. The unique chords in Christ Harte's format.
% 
% Outputs
%          - reduced_chords. The reduced triads chords.
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
            type='dim';
            
        case 3 % augmented
            type='aug';
            
        case 4 % suspended

            % can be sus2 or sus4, find out which
            susfind=regexp(CH_chords{chord},':sus');
            sustype=CH_chords{chord}(susfind+4);
            
            % can also be like this A:(1,2,4). Meh, make it major
            if isempty(sustype)
                type='maj';
            else
                
            % append 2 or 4
                type=['sus' sustype];
            end
            
    end % end switch
    
    % get rootnote and append type
    rootnote=getchordinfo(CH_chords{chord});
    
    reduced_chords{chord}=[rootnote ':' type];
    
end % end chord