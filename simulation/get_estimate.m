% This is to obtain estimation of parameters, mean and covariance functions

clear;
sufx={'','_mtimesx','_gpu'};
stgp_ver=['STGP',sufx{2}];
addpath('../util/');
if contains(stgp_ver,'mtimesx')
    addpath('../util/mtimesx/');
end
addpath('../util/tight_subplot/');
addpath(genpath('../util/boundedline-pkg/'));
% Random Numbers...
seedNO = 2018;
seed = RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% model options
stationary=false;
models={'sep','kron_prod','kron_sum'};
L_mdl=length(models);
upthypr=1;
alg_name='MCMC';
if upthypr>=2
    alg_name=['opt',alg_name];
    if upthypr==3
        alg_name=['jt',alg_name];
    end
end
intM=false;
hold_out=false;

%% data

% parameters setting
N=[200,100]; % discretization sizes for space and time domains
% N=[20,500]; % discretization sizes for space and time domains
trials=[100,1000]; % number of trials
% trials=[1]; % number of trials
L_trial=length(trials);
d=1; % space dimension
% load or simulate data
[x,t,y]=generate_data(N,trials(end),d,stationary,seedNO);
l_x=0.5; l_t=0.3; l_xt=sqrt(l_x*l_t);
sigma2_n=1e-2; % noise variance

% thin the mesh
thin=[50,1];
% thin=[5,1];
x=x(1:thin(1):end,:); t=t(1:thin(2):end); y=y(1:thin(1):end,1:thin(2):end,:);
I=size(x,1); J=size(t,1); L=I;
if hold_out
    % hold out some data (15% in time direction) for prediction
    t(end-floor(J*.2)+1:2:end)=[]; y(:,end-floor(J*.2)+1:2:end,:)=[];
    t(end-floor(J*.05)+1:end)=[]; y(:,end-floor(J*.05)+1:end,:)=[];
    I=size(x,1); J=size(t,1); L=I;
end

% parameters of kernel
s=2; % smoothness
% kappa=1.2; % decaying rate for dynamic eigenvalues
% spatial kernel
jit_x=1e-6;
% temporal kernels
jit_t=1e-6;


%% estimation
sigma2_estm=cell(L_mdl,L_trial); [sigma2_estm{:}]=deal(zeros(1,3)); sigma2_estd = sigma2_estm;
eta_estm=cell(L_mdl,L_trial); [eta_estm{:}]=deal(zeros(1,3)); eta_estd = eta_estm;
Lambda_estm=cell(L_mdl,L_trial); [Lambda_estm{:}]=deal(zeros(J,L)); Lambda_estd = Lambda_estm;
M_estm=cell(L_mdl,L_trial); [M_estm{:}]=deal(zeros(I,J)); M_estd = M_estm;
C_estm=cell(L_mdl,L_trial); [C_estm{:}]=deal(sparse(I*J,I*J)); C_estd = C_estm;

% estimate the mean and joint covariance kernel from MCMC samples
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
for mdl=1:L_mdl
    keywd = {[alg_name,'_',repmat('intM_',1,intM),models{mdl},'_I',num2str(I),'_J',num2str(J)],['_L',num2str(L),'_d',num2str(d)]};%,'_',repmat('non',1,~stationary),'stationary']};
    f_name = ['estxft_',keywd{:}];
    for tr=1:L_trial
        found=false;
        for k=1:nfiles
            if contains(files(k+2).name,join(keywd,['_K',num2str(trials(tr))]))
                load(strcat(folder, files(k+2).name));
                fprintf('%s loaded.\n',files(k+2).name);
                found=true; break;
            end
        end
        if found
            sigma2_estm{mdl,tr}=mean(samp_sigma2,1); sigma2_estd{mdl,tr}=std(samp_sigma2,0,1);
            eta_estm{mdl,tr}=mean(samp_eta,1); eta_estd{mdl,tr}=shiftdim(std(samp_eta,0,1));
            Lambda_estm{mdl,tr}=shiftdim(mean(samp_Lambda,1)); Lambda_estd{mdl,tr}=shiftdim(std(samp_Lambda,0,1));
            M_estm{mdl,tr}=shiftdim(mean(samp_M,1)); M_estd{mdl,tr}=shiftdim(std(samp_M,0,1));
            if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
                stgp=mgC.stgp;
            else
                ker{1}=feval([stgp_ver,'.GP'],x,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_x,true);
                ker{2}=feval([stgp_ver,'.GP'],t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},optini.Lambda,kappa,mdl-1); [ker{1:2}]=deal([]);
            end
            N_samp=size(samp_sigma2,1);
            for n=1:N_samp
                stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),shiftdim(samp_Lambda(n,:,:)));
                [~,C_z_n]=stgp.tomat();
                C_estm{mdl,tr}=C_estm{mdl,tr}+C_z_n; C_estd{mdl,tr}=C_estd{mdl,tr}+C_z_n.^2;
            end
            C_estm{mdl,tr}=C_estm{mdl,tr}./N_samp; C_estd{mdl,tr}=sqrt(C_estd{mdl,tr}./N_samp - C_estm{mdl,tr}.^2);
        end
    end
    
end


% save the estimation results
save([folder,'est_',[alg_name,repmat('_intM_',1,intM),'_I',num2str(I),'_J',num2str(J),'_L',num2str(L),'_d',num2str(d),'_',repmat('non',1,~stationary),'stationary'],'.mat'],...
     'sigma2_estm','sigma2_estd','eta_estm','eta_estd','Lambda_estm','Lambda_estd','M_estm','M_estd','C_estm','C_estd');