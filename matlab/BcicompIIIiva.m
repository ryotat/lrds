% BcicompIIIiva.m - main script file that applies the method to BCI
% competition III dataset IVa

file   = 'data_set_IVa_%s.mat';
file_t = 'data_set_IVa_%s_truth.mat';


subjects = {'aa','al','av','aw','ay'};

opt.ival= [500 3500];
opt.filtOrder= 5;
opt.band = [7 30];
opt.logm = 0;

%% Reduced set of 49 channels
opt.chanind = [14, 15, 16, 17, 18, 19, 20, 21, 22, 33, 34, 35, 36, 37, 38, ...
               39, 50, 51, 52, 53, 54, 55, 56, 57, 58, 68, 69, 70, 71, 72, ...
               73, 74, 75, 76, 87, 88, 89, 90, 91, 92, 93, 94, 95, 104, 106,...
              108, 112, 113, 114];

%% Load precomputed filter coefficients (7-30Hz Butterworth filter)
butter = load('butter730.mat');

lambda = exp(linspace(log(0.01), log(100), 20));

memo = repmat(struct('lambda',[],'cls',[],'out',[],'loss',[]),...
              [length(subjects), length(lambda)]);

for jj=1:length(subjects)
  fprintf('Subject: %s\n', subjects{jj});
  
  %% Load a dataset and preprocess
  load(sprintf(file, subjects{jj}));

  %% Select channels and covert cnt into double
  cnt  = 0.1*double(cnt(:,opt.chanind));
  clab = nfo.clab(opt.chanind);
  C = length(clab);
  
  %% Apply a band-pass filter
  cnt = filter(butter.b, butter.a, cnt);
  
  %% Cut EEG into tirals
  xepo = cutoutTrials(cnt, mrk.pos, opt.ival, nfo.fs);
  X = covariance(xepo);
  Y = (mrk.y-1.5)*2;  % convert {1,2} -> {-1, 1}
  
  %% Find indices of training and test set
  Itrain = find(~isnan(Y));
  Itest  = find(isnan(Y));

  %% Whiten the training data
  Xtr = X(:,:,Itrain);
  Ytr = Y(Itrain);
  [Xtr, Ww] = whiten(Xtr);

  if opt.logm
    Xtr = logmatrix(Xtr);
  end
    
  %% Train the classifier for various values of lambda
  for ii=1:length(lambda)
    [W, bias] = lrds_dual(Xtr, Ytr, lambda(ii));
    memo(jj,ii).lambda = lambda(ii);
    if ~opt.logm
      memo(jj,ii).cls    = struct('W',W,'bias',bias,'Ww',Ww);
    else
      memo(jj,ii).cls    = struct('W',W,'bias',bias,'Ww',eye(C));
    end
  end
  
  
  %% Load the true label of the test set
  load(sprintf(file_t, subjects{jj}));
  true_y = (true_y(Itest)-1.5)*2;

  Xte = X(:,:,Itest);
  
  if opt.logm
    Xte = logmatrix(matmultcv(Xte, Ww));
  end
    
  %% Apply the classifier
  fprintf('Subject: %s\n', subjects{jj});
  fprintf('lambda\t loss\n------------------------------------\n');
  for ii=1:length(lambda)
    memo(jj,ii).out  = apply_lrds(Xte, memo(jj,ii).cls);
    memo(jj,ii).loss = loss_0_1(true_y, memo(jj,ii).out);
    
    fprintf('%g\t%g\n', lambda(ii), memo(jj,ii).loss);
  end
end

loss=cell2mat(getfieldarray(memo,'loss'));

figure, plot(log(lambda), 100*(1-loss)', 'linewidth',2)
set(gca,'fontsize',20)
set(gca,'xtick',log(0.01):log(10):log(100))
set(gca,'xticklabel', {'0.01', '0.1', '1.0', '10', '100'})
grid on;
hold on;
plot(log(lambda), 100*(1-mean(loss)), 'color',[.7 .7 .7], 'linewidth', 2);
leg = [subjects {'average'}];
legend(leg);
xlabel('Regularization constant \lambda')
ylabel('Classification accuracy')
