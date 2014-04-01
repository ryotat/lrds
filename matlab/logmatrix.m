function xcv = logmatrix(xcv)
% logmatrix - takes logm for each trial.
%
% Syntax:
%  xcv = logmatrix(xcv)
  
[C1,C2,n] = size(xcv);

if C1~=C2
  error('Input is not square.')
end

for ii=1:n
  xcv(:,:,ii) = logm(xcv(:,:,ii));
end
