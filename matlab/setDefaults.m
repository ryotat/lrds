function opt = setDefaults(opt, optdef)
% setDefaults - set defaults.
%
% Syntax:
%  opt = setDefaults(opt, optdef)
  
flds = fieldnames(optdef);

for ii=1:length(flds)
  if ~isfield(opt, flds{ii})
    opt = setfield(opt, flds{ii}, getfield(optdef, flds{ii}));
  end
end
