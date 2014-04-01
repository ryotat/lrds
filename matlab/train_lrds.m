function cls = train_lrds(X, Y, lambda)
% train_lrds - wrapper function that produces the classifier struct
%
% Syntax:
%  cls = train_lrds(X, Y, lambda)
Ivalid = find(~isnan(Y));

X = X(:,:,Ivalid);
Y = Y(Ivalid);

[X, Ww] = whiten(X);

[W, bias] = lrds_dual(X, Y, lambda);

cls = struct('W', W, 'bias', bias, 'Ww', Ww);