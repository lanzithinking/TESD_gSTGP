% This is to generate simulated data of a spatio-temporal process
% for multiple trials

function [x,t,y]=generate_data(N,K,d,seedNO)
if ~exist('N','var')
    N=[200,100]; % discretization sizes for space and time domains
end
if ~exist('K','var')
    K=1; % number of trials
end
if ~exist('d','var')
    d=1; % space dimension
end
if ~exist('seed','var')
    seedNO=2018;
end

I=N(1)+1; J=N(2)+1;
folder='./data/';
files=dir(folder);
nfiles=length(files)-2;
keywds={['sim_STproc_I',num2str(I),'_J',num2str(J)],['_d',num2str(d)]};
found=false;
for k=1:nfiles
    if contains(files(k+2).name,keywds{1}) && contains(files(k+2).name,keywds{2})
        N_trial=extractBetween(files(k+2).name,'_K','_d');
        if K<=str2num(N_trial{1})
            % load data
            load([folder,files(k+2).name],'x','t','y');
            y=y(:,:,1:K);
            fprintf('%s loaded.\n',files(k+2).name);
            found=true; break;
        end
    end
end
if ~found
    % or generate data
    [x,t,y,~]=sim_STproc(N,K,d,seedNO,true,false);
end

end