 function BassNotes = reduce_to_bass(chords)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: 
%BassNotes = reduce_to_bass(chords)
%
%Convert a cell array of CH format chords to bass representations.
%Prerequisite: parsechord.m, note2pitchclass.m, degree2note.m in CH toolbox.
%
% Inputs
%          - chords. The unique (full) chords in Christ Harte's format.
% 
% Outputs
%          - BassNotes. The bass note of the chords.
%
%---------------------------------------------
%Function created by M. McVicar
%Function revised by Y. Ni
%Intelligent Systems Lab
%University of Bristol
%U.K.
%2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
 BassNotes= zeros(1,length(chords));
 for t=1:length(chords)
     C=chords{t};
     %If it is an unknown chord, then bass_note=14;
     if (C(1)=='X')
         BassNotes(t)=14;
     else
         % get chord info
         [rootnote, shorthand, degreelist,bass] = parsechord(C);
         
         % no chord
         if strcmp(rootnote,'N')
             BassNotes(t)=13;
         %Patch 1: no chord
         elseif strcmp(rootnote,'&pause')
             BassNotes(t)=13;
             
             % no inversion, bass = rootnote
         elseif isempty(bass)
             BassNotes(t)=note2pitchclass(rootnote)+1;
             
             % there's a bassnote
         else
             BassNotes(t)=note2pitchclass(bass)+1;
         end
         
         % check if it was an interval
         if BassNotes(t)==0
             % get the note wrt the root
             bnote=degree2note(bass,rootnote);
             bnote=bnote{1};
            
             
             % convert the note to a pitch class
             BassNotes(t)=note2pitchclass(bnote)+1;
         end
     end
 end