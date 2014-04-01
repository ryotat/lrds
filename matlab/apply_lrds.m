function out = apply_lrds(X, cls)
% apply_lrds - applies the classifier
%
% Syntax:
%  out = apply_lrds(X, cls)
  
[C1, C2, n] = size(X);

if C1==1
  C = sqrt(C2);
  X = shiftdim(X);
elseif C1~=C2
  error('Input is not square.');
else
  C = C1;
  X = reshape(X, [C^2, n]);
end

out = reshape(cls.Ww*cls.W*cls.Ww', [1,C^2])*X + cls.bias;