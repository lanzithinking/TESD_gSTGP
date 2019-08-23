% This is to read ADNI data for image analysis
% for multiple subjects

function [t,y]=read_data(typ,grp,dur,loc)
if ~exist('typ','var') || isempty(typ)
    typ='PET';
end
if ~exist('grp','var') || isempty(grp)
    grp='MCI';
end
if ~exist('dur','var') || isempty(dur)
    dur=5;
end
if ~exist('loc','var') || isempty(loc)
    loc='./data/';
end

J=dur;
folder=loc;
files=dir(folder);
nfiles=length(files)-2;
keywds=['AD',typ,'_',grp];
if contains(typ,'PET')
    keywds=[keywds,'_uniform'];
end
keywds=[keywds,'_J',num2str(J)];
found=false;
for k=1:nfiles
    if contains(files(k+2).name,keywds)
        % load data
        load([folder,files(k+2).name],'K','t','y');
        fprintf('%s loaded.\n',files(k+2).name);
        found=true; break;
    end
end
if found
    % print sumamry
    fprintf('There are %d subject(s) in %s group read for %s images at %d time points.\n', K,grp,typ,J);
else
    warning('No specified data found!');
end

end