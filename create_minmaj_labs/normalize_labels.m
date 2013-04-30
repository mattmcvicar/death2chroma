function Y = normalize_labels(X, ruleset)
%  Y = normalize_labels(X, R)
%      Convert a cell array of rich chord labels into a 
%      list of integers in a reduced set (0..24)
%      R = ruleset. 0 = dpwe default, 1 = mirex 2009, 2 = include diminished
% 2008-08-11 Dan Ellis dpwe@ee.columbia.edu

if nargin < 2;  ruleset = 0; end

% Canonical chord label index
keytab = {'N', ...
          'C','C#','D','D#','E','F','F#','G','G#','A','A#','B', ...
          'C:min','C#:min','D:min','D#:min','E:min','F:min', ...
          'F#:min','G:min','G#:min','A:min','A#:min','B:min', ...
          'C:7','C#:7','D:7','D#:7','E:7','F:7', ...
          'F#:7','G:7','G#:7','A:7','A#:7','B:7' };

ul = unique(X);

flats = {'Ab','Bb','Cb','Db','Eb','Fb','Gb'};
shrps = {'G#','A#','B','C#','D#','E','F#'};

Y = zeros(1,length(X));

if ruleset == 0
  patterns = {'min'};
  replaces = {':min'};
elseif ruleset == 1
  patterns = {'min','dim','sus2'};
  replaces = {':min',':min',':min'};
elseif ruleset == 2
  patterns = {'min','7'};
  replaces = {':min',':7'};
end

for i = 1:length(ul)
  u = ul{i};
  %disp(u)
  % Convert this label
  [r,s] = splitchordlab(u);
  ix = find(strcmp(flats,r));
  if length(ix) > 0
    r = shrps{ix};
  end
  % default mapping
  u2 = r; 
  for m = 1:length(patterns)
    if strncmp(s, patterns{m}, length(patterns{m}))
      u2 = [r,replaces{m}];
    end
  end
  % find its index
  un = find(strcmp(keytab, u2));
  % report unrecognized label
  if length(un) == 0
    error(['unrecognized label ',u,' (converts to ',u2,')']);
  end
  % write it in
  Y(strcmp(X,u)) = un-1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [r,s] = splitchordlab(L)
s = '';
x = min(find(L=='/'));
if length(x)
  L = L(1:x-1);
end
r = L;
x = min(find(L==':'));
if length(x)
  r = L(1:x-1);
  s = L(x+1:end);
end
