function [W, bias, z, status]=lrds_dual(X, Y, lambda, varargin)
% lrds_dual - Logistic regression with dual spectral regularization
%             for symmetric input matrix.
%
% Objective:
%    Solves the regularized ERM problem:
%        sum(loss(f(X_i),y_i)) + lambda*sum(svd(W))
%    where
%        f(X) = trace(W'*X) + bias
%
% Syntax:
%  [W, bias, z, status]=lrds_dual(X, Y, lambda, <opt>)
%
% Inputs:
%  X      : Input matrices.
%           [C,C,n] array: Each X(:,:,i) assumed to be symmetric.
%           [C^2,n] array: Each reshape(X(:,i), sqrt(size(X,1))) assumed to be symmetric.
%  Y      : Binary lables. +1 or -1. [n,1] matrix. n is the number of samples.
%  lambda : Regularization constant.
%  <opt>  : Options.
%    .tol     : absolute tolerance for duality gap (1e-6)
%    .tolX    : relative tolerance for step size (1e-6)
%    .tmul    : barrier parameter multiplier (20)
%    .maxiter : maximum number of iteration (1000)
%    .display : 'all'(4), 'every'(3), 'iter'(2), 'final'(1), or 'none'(0) ('iter')
%
% Outputs:
%  W      : Weight matrix. [C,C] matrix.
%  bias   : Bias term.
%  z      : Classifier outputs. [n,1] matrix.
%  status : Miscellaneous numbers.
%
% Reference:
% "Classifying Matrices with a Spectral Regularization",
% Ryota Tomioka and Kazuyuki Aihara,
% Proc. ICML2007, pp. 895-902, ACM Press; Oregon, USA, June, 2007.
% http://www.machinelearning.org/proceedings/icml2007/papers/401.pdf
%
% This software is distributed from:
% http://www.sat.t.u-tokyo.ac.jp/~ryotat/lrds/index.html
%
% Ryota Tomioka 2007.

try
  opt = propertylist2struct(varargin{:});
catch
  if nargin>3
    opt = varargin{1};
  else
    opt = [];
  end
end

opt = setDefaults(opt, struct('tol', 1e-6, ...
                        'tolX', 1e-6, ...
                        'tmul', 20, ...
                        'maxiter', 1000, ...
			       'display', 'iter'));

if ~isnumeric(opt.display)
  opt.display = find(strcmp(opt.display,...
                            {'none','final','iter','every','all'}))-1;
end


if ndims(X)==3 & size(X,1)==size(X,2)
  [C,Cd, n]=size(X);
  Xf=reshape(X, [C*C, n]);
elseif ndims(X)==3 & size(X,1)
  [Cd,C, n]=size(X);
  Xf=shiftdim(X);
  X =reshape(X,[C,C,n]);
else
  [CC,n]=size(X);
  C=sqrt(CC);
  Xf = X;
  X  = reshape(X,[C,C,n]);
end

if n~=length(Y)
  error('Sample size mismatch!');
end

Y=shiftdim(Y);

if isfield(opt,'alpha')
  alpha = opt.alpha;
else
  alpha = zeros(n,1);
  alpha(Y>0) = min(lambda*0.01,1)/sum(Y>0);
  alpha(Y<0) = min(lambda*0.01,1)/sum(Y<0);
end

cc = 0;
display_line_search = opt.display==4;

if isfield(opt,'t')
  t = opt.t;
else
  t = 2*(C+n)/(n*log(2));
end

cc0 = 0;
time0 = cputime;
time00 = time0;

while cc<opt.maxiter
  while cc<opt.maxiter
    cc = cc + 1;
    
    A = reshape(Xf*(alpha.*Y), [C,C]); A=(A+A')/2;

    R1 = chol(lambda*eye(C)-A); % X = R'*R (R is upper diagonal)
    R2 = chol(lambda*eye(C)+A);

    
    [loss, gl, hl]=lossDual(alpha);


    %% Calculate the gradient and Hessian associated to the
    %% barrier log det term
    SX1 = zeros(C*C,n);
    SX2 = zeros(C*C,n);
    gS  = zeros(n,1);
    for i=1:n
      D1 = R1'\(Y(i)*X(:,:,i))/R1;
      D2 = R2'\(Y(i)*X(:,:,i))/R2;
      SX1(:,i)=reshape(D1, [C*C,1]);
      SX2(:,i)=reshape(D2, [C*C,1]);
      gS(i) = sum(diag(D1-D2));
    end
    
    H1 = SX1'*SX1;
    H2 = SX2'*SX2;

    %% Gradient
    g = gl+((2*alpha-1)./(alpha.*(1-alpha)) + gS)/t;

    %% Hessian
    Hd = hl + (alpha.^(-2)+(1-alpha).^(-2))/t;
    Hr = (H1+H2)/t;
    H = diag(Hd)+Hr;
    
    HIg = H\g;
    HIy = H\Y;

    %% Lagrangian multp. assoc. to the equality constraint
    nu = - (Y'*HIg)/(Y'*HIy);
    
    %% Newton direction
    delta = -H\(g+Y*nu);

    alpha0 = alpha;

    %% Line search to determine the stepsize s
    Sd0 = reshape(Xf*(delta.*Y), [C,C]); Sd0=(Sd0+Sd0')/2;
    Sd1 = -eig(R1'\Sd0/R1);
    Sd2 = eig(R2'\Sd0/R2);

    [s, dloss] = lineSearch(alpha, delta, t, Sd1, Sd2, Y*nu, opt.tolX/max(abs(delta)./abs(alpha)), display_line_search);

    
    %% Update
    alpha = alpha0 + s*delta;

    A = reshape(Xf*(alpha.*Y), [C,C]); A=(A+A')/2;
    
   
    %% Weight matrix

    RR = chol(lambda^2*eye(C)-A*A');

    W = 2*(RR\((RR')\A))/t;
    W = (W+W')/2;
    trQ = 2*lambda*trace(((RR')\eye(C))/RR)/t;

    %% trQ = trace(Q1) + trace(Q2), where
    %% (lambda*eye(C) - A) * Q1 = 1/t *eye(C)
    %% (lambda*eye(C) + A) * Q2 = 1/t *eye(C)
    %% 
    %% Note that trQ is not exactly sum(svd(W)) until convergence

    %% Lag. multp. assoc. to the box constraints 0<=alpha<=1
    beta1= 1./(t*alpha);
    beta2= 1./(t*(1-alpha));

    %% Bias term (= Lag. multp. assoc. to the equality constraint)
    bias = nu;
    
    z=Y.*(reshape(W,[1,C*C])*Xf+bias)'-beta1+beta2;

    %% Primal objective
    loss_prim = lossPrime(z)+sum(beta2)+lambda*trQ;
    
    %% Dual objective (at new alpha)
    loss=lossDual(alpha);
    
    %% Primal - dual gap
    gap(cc) = loss_prim - (-loss);

    %% Interior-point objective function
    obj(cc) = loss +1/t*(-2*sum(log(diag(RR)))...
                         -sum(log(alpha))-sum(log(1-alpha)));
    
    %% IP first order optimality
    gg(cc)  = max(abs(g+Y*nu));

    
    %% Check validity
    if 0
    [Va, Da]=eig(A);
    da=diag(Da);
    lmW = 2*da./(t*(lambda^2-da.^2));

    W0  = Va*diag(lmW)*Va';
    trQ0 = sum(2*lambda./(t*(lambda^2-da.^2)));
    z0=Y.*(reshape(W0,[1,C*C])*Xf+bias)'-beta1+beta2;
    loss_prim0 = lossPrime(z0)+sum(beta2)+lambda*trQ0;

    obj0 = loss +1/t*(-sum(log(lambda-da))-sum(log(lambda+da))...
                         -sum(log(alpha))-sum(log(1-alpha)));

    fprintf('!!! |W-W0|=%g dz=%g, dtrQ=%g dloss_prim=%g dobj=%g\n',...
            max(abs(rangeof(W-W0))),...
            max(abs(rangeof(z-z0))),...
            trQ-trQ0,...
            loss_prim-loss_prim0,...
            obj(cc)-obj0);
    end
    

    if opt.display>=3
      fprintf('[%d] t=%g gap=%g(>=%g) gg=%g y*alpha=%g nu=%g Hmin=%g s=%g obj=%g',...
            cc, t, gap(cc), 2*(C+n)/t, gg(cc), Y'*alpha, nu, min(eig(H)), s, obj(cc));
    
      if cc>1
      fprintf(' dloss=(%g/%g)\n', obj(cc)-obj(cc-1), dloss);
    else
      fprintf('\n');
      end
    end
    

    if gg(cc)<opt.tol | ((gg(cc)<opt.tol*min(100,gap(cc)/opt.tol) | gap(cc)<opt.tol) & s* ...
                         max(abs(delta)./abs(alpha))<opt.tolX);

      tlap = cputime-time0;

      if opt.display>=2
        fprintf('t=%g: gap=%g gg=%g nsteps=%d time=%g\n',...
                t,gap(cc),gg(cc),cc-cc0,tlap);
      end
      cc0 = cc;
      time0 = cputime;
        
      break;
    end
  end
  if gap(cc)<opt.tol
    break;
  else
    t = t*opt.tmul;
  end
end


status = struct('opt',opt,...
                'niter',cc,...
                't',t,...
                'gap',gap,...
                'obj',obj,...
                'beta1',beta1,...
                'beta2',beta2,...
                'alpha', alpha,...
                'time', cputime-time00);



if opt.display>0
  fprintf('[%d] gap=%g total time=%g\n', cc, gap(end),cputime-time00);
end


function loss = lossPrime(z)
z1 = z(z<0);
z2 = z(z>=0);
loss = sum(log(exp(z1)+1))-sum(z1)+sum(log(1+exp(-z2)));

loss0=sum(log(1+exp(-z)));

if ~isinf(loss0) & abs(loss-loss0)>1e-9
  error;
end


function [loss, g, h] = lossDual(alpha)

ix = alpha~=0 & alpha~=1;

loss = zeros(size(alpha));
g    = zeros(size(alpha));
h    = zeros(size(alpha));

loss(~ix) = 0;
loss(ix) = alpha(ix).*log(alpha(ix)) + (1-alpha(ix)).*log(1-alpha(ix));

g(~ix)= nan;
g(ix) = log(alpha(ix)./(1-alpha(ix)));

h(~ix)= nan;
h(ix) = 1./alpha(ix) + 1./(1-alpha(ix));

loss = sum(loss);


function [s,dloss] = lineSearch(alpha,delta,t,Sd1,Sd2,gnu,tolX,display);
snew = 1;
s1 = 0;
s2 = nan;
s_best = 0;
alpha0 = alpha;
loss0 = lossDual(alpha0);

dloss_best = 0;
cc = 1;


while 1
  s = snew;
  cc = cc +1;
  
  if display
    if s_best > s
      fprintf(' %02d: s1=%.2f   s=%.2f * s2=%.2f',cc,s1,s,s2);
    elseif s_best < s
      fprintf(' %02d: s1=%.2f * s=%.2f   s2=%.2f',cc,s1,s,s2);
    else
      fprintf(' %02d: _s1=%.2f   s=%.2f   s2=%.2f',cc,s1,s,s2);
    end
  end
  
  lm1 = 1+s*Sd1;
  lm2 = 1+s*Sd2;

  alpha = alpha0 + s*delta;
  
  isfeas = 0;
  if any(lm1<=0) | any(lm2<=0) | any(alpha<=0) | any(alpha>=1)
    ss = '!';
    s2 = min(s, s2);
    snew = max((s1+s2)/2,s/2);
  else
    %% feasible
    isfeas = 1;
     dloss = lossDual(alpha)-loss0...
            +1/t*(-sum(log(lm1))-sum(log(lm2))...
            -sum(log(1+s*delta./alpha0))-sum(log(1-s*delta./(1-alpha0))))...
             +s*gnu'*delta;

        
    if dloss < dloss_best
      ss = '-';
      dloss_best = dloss;
      if s_best<s
        s1 = s_best;
      elseif s<s_best
        s2 = s_best;
      end
      
      s_best = s;
      
    else
      ss = '+';
      if s_best<s
        s2 = s;
      elseif s<s_best
        s1 = s;
      end
    end

    if isnan(s2)
      snew = s*2;
    else
      r = 0.5 + rand(1)*0.1-0.05;
      snew = s1*r+s2*(1-r);
    end

    % obj(cc) = loss +1/t*(-log(det(S1))-log(det(S2))-sum(log(alpha))-sum(log(1-alpha)));

  end

  if display
    fprintf(' dloss_best=%g (%s)\n',dloss_best, ss);
  end
  
  if (isfeas & s2-s1<0.01)
    break;
  end
  if (isfeas & isnan(s1) & s<tolX)
    break;
  end
end


function obj = objectiveLocal(alpha0, delta, x, Xf, Y, lambda, t)
%
% [Xl,Yl]=ndgrid(-1:0.1:1);
% xl = [shiftdim(Xl,-1); shiftdim(Yl,-1)];
% r = max(abs(delta))/max(abs(g+Y*nu))
% P = eye(n)-Y*Y'/n;
% objl = objectiveLocal(alpha, [delta, -P*g*r*0.001], xl, Xf, Y, lambda, t);
% figure, surf(Xl, Yl, objl, 'edgecolor','none')



sz = size(x); sz(1)=1;

obj = squeeze(zeros(sz));

C = sqrt(size(Xf,1));

for i=1:prod(size(obj))
  alpha = alpha0 + delta*x(:,i);
  A = reshape(Xf*(alpha.*Y), [C,C]); A=(A+A')/2;
  S1 = lambda*eye(C)-A;
  S2 = lambda*eye(C)+A;
  
  
  obj(i) = lossDual(alpha) +1/t*(-sum(log(eig(S1)))-sum(log(eig(S2)))-sum(log(alpha))-sum(log(1-alpha)));

  if any(alpha<0) | any(alpha>1) | any(eig(S1)<0) | any(eig(S2)<0)
    obj(i)=nan;
  end
end
 
