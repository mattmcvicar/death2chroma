function [reduced_keys]=reduce_to_key(keys)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: 
%[reduced_keys]=reduce_to_key(keys)
%
%Convert a cell array of keys (format restricted) to key indices.
%
% Inputs
%          - keys. The unique keys (format restricted, see below).
% 
% Outputs
%          - reduced_keys. The reduced key labels.
%
%---------------------------------------------
%Function created by Y. Ni
%Intelligent Systems Lab
%University of Bristol
%U.K.
%2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


key_map=containers.Map();

%%%%%%%%%%%% Enumerate all possible key types (can be modified) %%%%%%%%%%%
%1. No key
key_map('n')='N';
key_map('silence')='N';

%2. A:min
key_map('key a:minor')='A:min';
key_map('key a:min')='A:min';
key_map('a:min')='A:min';
key_map('a:minor')='A:min';
key_map('am')='A:min';

%3. A:maj
key_map('key a:major')='A:maj';
key_map('key a:maj')='A:maj';
key_map('a:maj')='A:maj';
key_map('a:major')='A:maj';
key_map('a')='A:maj';

%4. A#:min
key_map('key a#:minor')='A#:min';
key_map('key a#:min')='A#:min';
key_map('a#:min')='A#:min';
key_map('a#:minor')='A#:min';
key_map('a#m')='A#:min';

key_map('key bb:minor')='A#:min';
key_map('key bb:min')='A#:min';
key_map('bb:min')='A#:min';
key_map('bb:minor')='A#:min';
key_map('bbm')='A#:min';


%5. A#:maj
key_map('key a#:major')='A#:maj';
key_map('key a#:maj')='A#:maj';
key_map('a#:maj')='A#:maj';
key_map('a#:major')='A#:maj';
key_map('a#')='A#:maj';

key_map('key bb:major')='A#:maj';
key_map('key bb:maj')='A#:maj';
key_map('bb:maj')='A#:maj';
key_map('bb:major')='A#:maj';
key_map('bb')='A#:maj';

%6. B:min
key_map('key b:min')='B:min';
key_map('key b:minor')='B:min';
key_map('b:min')='B:min';
key_map('b:minor')='B:min';
key_map('bm')='B:min';


%7. B:maj
key_map('key b:maj')='B:maj';
key_map('key b:major')='B:maj';
key_map('b:maj')='B:maj';
key_map('b:major')='B:maj';
key_map('b')='B:maj';


%8. C:min
key_map('key c:min')='C:min';
key_map('key c:minor')='C:min';
key_map('c:min')='C:min';
key_map('c:minor')='C:min';
key_map('cm')='C:min';


%9. C:maj
key_map('key c:maj')='C:maj';
key_map('key c:major')='C:maj';
key_map('c:maj')='C:maj';
key_map('c:major')='C:maj';
key_map('c')='C:maj';


%10. C#:min
key_map('key c#:minor')='C#:min';
key_map('key c#:min')='C#:min';
key_map('c#:min')='C#:min';
key_map('c#:minor')='C#:min';
key_map('c#m')='C#:min';


key_map('key db:minor')='C#:min';
key_map('key db:min')='C#:min';
key_map('db:min')='C#:min';
key_map('db:minor')='C#:min';
key_map('dbm')='C#:min';


%11. C#:maj
key_map('key c#:major')='C#:maj';
key_map('key c#:maj')='C#:maj';
key_map('c#:maj')='C#:maj';
key_map('c#:major')='C#:maj';
key_map('c#')='C#:maj';


key_map('key db:major')='C#:maj';
key_map('key db:maj')='C#:maj';
key_map('db:maj')='C#:maj';
key_map('db:major')='C#:maj';
key_map('db')='C#:maj';


%12. D:min
key_map('key d:min')='D:min';
key_map('key d:minor')='D:min';
key_map('d:min')='D:min';
key_map('d:minor')='D:min';
key_map('dm')='D:min';


%13. D:maj
key_map('key d:maj')='D:maj';
key_map('key d:major')='D:maj';
key_map('d:maj')='D:maj';
key_map('d:major')='D:maj';
key_map('d')='D:maj';


%14. D#:min
key_map('key d#:minor')='D#:min';
key_map('key d#:min')='D#:min';
key_map('d#:min')='D#:min';
key_map('d#:minor')='D#:min';
key_map('d#m')='D#:min';


key_map('key eb:minor')='D#:min';
key_map('key eb:min')='D#:min';
key_map('eb:min')='D#:min';
key_map('eb:minor')='D#:min';
key_map('ebm')='D#:min';


%15. D#:maj
key_map('key d#:major')='D#:maj';
key_map('key d#:maj')='D#:maj';
key_map('d#:maj')='D#:maj';
key_map('d#:major')='D#:maj';
key_map('d#')='D#:maj';


key_map('key eb:major')='D#:maj';
key_map('key eb:maj')='D#:maj';
key_map('eb:maj')='D#:maj';
key_map('eb:major')='D#:maj';
key_map('eb')='D#:maj';


%16. E:min
key_map('key e:min')='E:min';
key_map('key e:minor')='E:min';
key_map('e:min')='E:min';
key_map('e:minor')='E:min';
key_map('em')='E:min';


%17. E:maj
key_map('key e:maj')='E:maj';
key_map('key e:major')='E:maj';
key_map('e:maj')='E:maj';
key_map('e:major')='E:maj';
key_map('e')='E:maj';


%18. F:min
key_map('key f:min')='F:min';
key_map('key f:minor')='F:min';
key_map('f:min')='F:min';
key_map('f:minor')='F:min';
key_map('fm')='F:min';


%19. F:maj
key_map('key f:maj')='F:maj';
key_map('key f:major')='F:maj';
key_map('f:maj')='F:maj';
key_map('f:major')='F:maj';
key_map('f')='F:maj';


%20. F#:min
key_map('key f#:minor')='F#:min';
key_map('key f#:min')='F#:min';
key_map('f#:min')='F#:min';
key_map('f#:minor')='F#:min';
key_map('f#m')='F#:min';


key_map('key gb:minor')='F#:min';
key_map('key gb:min')='F#:min';
key_map('gb:min')='F#:min';
key_map('gb:minor')='F#:min';
key_map('gbm')='F#:min';


%21. F#:maj
key_map('key f#:major')='F#:maj';
key_map('key f#:maj')='F#:maj';
key_map('f#:maj')='F#:maj';
key_map('f#:major')='F#:maj';
key_map('f#')='F#:maj';


key_map('key gb:major')='F#:maj';
key_map('key gb:maj')='F#:maj';
key_map('gb:maj')='F#:maj';
key_map('gb:major')='F#:maj';
key_map('gb')='F#:maj';


%22. G:min
key_map('key g:min')='G:min';
key_map('key g:minor')='G:min';
key_map('g:min')='G:min';
key_map('g:minor')='G:min';
key_map('gm')='G:min';


%23. G:maj
key_map('key g:maj')='G:maj';
key_map('key g:major')='G:maj';
key_map('g:maj')='G:maj';
key_map('g:major')='G:maj';
key_map('g')='G:maj';


%24. G#:min
key_map('key g#:minor')='G#:min';
key_map('key g#:min')='G#:min';
key_map('g#:min')='G#:min';
key_map('g#:minor')='G#:min';
key_map('g#m')='G#:min';


key_map('key ab:minor')='G#:min';
key_map('key ab:min')='G#:min';
key_map('ab:min')='G#:min';
key_map('ab:minor')='G#:min';
key_map('abm')='G#:min';


%25. G#:maj
key_map('key g#:major')='G#:maj';
key_map('key g#:maj')='G#:maj';
key_map('g#:maj')='G#:maj';
key_map('g#:major')='G#:maj';
key_map('g#')='G#:maj';


key_map('key ab:major')='G#:maj';
key_map('key ab:maj')='G#:maj';
key_map('ab:maj')='G#:maj';
key_map('ab:major')='G#:maj';
key_map('ab')='G#:maj';

%26. Patch: Cb
key_map('cb')='B:maj';
key_map('cbm')='B:min';
key_map('cb:maj')='B:maj';
key_map('cb:min')='B:min';

%%%%%%%%%%%%Enumerate all possible key types (can be modified) %%%%%%%%%%%%


%Get the key information and reduced to the 25-class key labels
num_keys=length(keys);
reduced_keys=cell(1,num_keys);

for i=1:num_keys
    key_lower_case=lower(keys{i});
    if (isKey(key_map,key_lower_case))
        reduced_keys{i}=key_map(key_lower_case);
    else
        warning(['Warning in reduce_to_key.m: Can not find the key ',keys{i},' in key map. Change it to no-key label.']);
        reduced_keys{i}='N';
    end
end

