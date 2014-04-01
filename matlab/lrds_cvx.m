function [W, bias, z, status]=lrds_cvx(X, Y, lambda, bQuiet, precision)
% lrds_cvx - logistic regression with dual spectral regularization
%            for symmetric matrix inputs (using CVX)
% Syntax:
%  [W, bias, z, status]=lrds_cvx(X, Y, lambda, bQuiet, precision)
%
% Inputs:
%  X      : Input matrices.
%           [C,C,n] array: Each X(:,:,i) assumed to be symmetric.
%           [C^2,n] array: Each reshape(X(:,i), sqrt(size(X,1))) assumed to be symmetric.
%  Y      : Binary lables. +1 or -1. [n,1] matrix. n is the number of samples.
%  lambda : Regularization constant.
%  bQuiet : binary. true suppresses outputs. (default)
%  precision : precision used by CVX. See CVX manual. (optional)
%
% Outputs:
%  W      : Weight matrix. [C,C] matrix.
%  bias   : Bias term
%  z      : Classifier outputs. [n,1] matrix.
%  status : Status returned by CVX.
%
% Reference:
% "Classifying Matrices with a Spectral Regularization",
% Ryota Tomioka and Kazuyuki Aihara,
% Proc. ICML2007, pp. 895-902, ACM Press; Oregon, USA, June, 2007.
% http://www.machinelearning.org/proceedings/icml2007/papers/401.pdf
%
% CVX: Matlab Software for Disciplined Convex Programming  
% Michael Grant, Stephen Boyd, Yinyu Ye
% http://www.stanford.edu/~boyd/cvx/
%
% This software is distributed from:
% http://www.sat.t.u-tokyo.ac.jp/~ryotat/lrds/index.html
%
% Ryota Tomioka 2007.


if ~exist('bQuiet','var') | isempty(bQuiet)
  bQuiet = true;
end

if ~exist('precision','var') | isempty(precision)
  precision = 'default';
end

m = size(X,1);
n = length(Y);

cvx_begin sdp
cvx_quiet(bQuiet);
cvx_precision(precision);
variable W(m,m) symmetric;
variable U(m,m) symmetric;
variable bias;
variable z(n);
minimize sum(log(1+exp(-z))) + lambda*trace(U);
subject to
for i=1:n
    Y(i)*(trace(W*X(:,:,i))+bias)==z(i);
end
U >= W;
U >= -W;
cvx_end
status = cvx_status;

if isempty(strfind(status,'Solved'))
  fprintf('CVX_STATUS [%s]. All coefficients set to zero.\n', status);
  W    = zeros(m,m);
  bias = 0;
end
