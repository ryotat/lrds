function C=getfieldarray(A, field)
% getfieldarray - get field from a structured array.
%
% Syntax:
%  C=getfieldarray(A, field)
C=cell(size(A));

for i=1:prod(size(A))
  C{i}=getfield(A(i), field);
end
