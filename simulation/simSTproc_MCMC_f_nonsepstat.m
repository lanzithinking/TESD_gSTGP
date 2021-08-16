% inference of a simulated spatio-temporal process model using MCMC
% for multiple-trials
% optimize sigma2, eta or sample them using slice sampler,
% sample Lambda, (M) using ESS, (Gibbs) resp.

function []=simSTproc_MCMC_f_nonsepstat(num_trl,stationary,mdl_opt,upthypr,sampleM,intM,hold_out,spdapx,seedNO)
sufx={'','_mtimesx','_gpu'};
stgp_ver=['STGP',sufx{2}];
addpath('../util/','../sampler/');
% addpath(['../util/+',stgp_ver,'/']);
if contains(stgp_ver,'mtimesx')
    addpath('../util/mtimesx/');
end
% if ~isdeployed
%     addpath ~/Projects/STGP/code/util/;
%     addpath ~/Projects/STGP/code/sampler/;
%     addpath ~/Projects/STGP/code/util/STGP_mtimesx/;
%     addpath ~/Projects/STGP/code/util/mtimesx/;
% end
% if isdeployed
%     num_trl=str2num(num_trl);
%     stationary=str2num(stationary);
%     mdl_opt=str2num(mdl_opt);
%     upthypr=str2num(upthypr);
%     sampleM=str2num(sampleM);
%     intM=str2num(intM);
%     hold_out=str2num(hold_out);
%     spdapx=str2num(spdapx);
%     seedNO=str2num(seedNO);
% end

% settings
if ~exist('num_trl','var') || isempty(num_trl)
    num_trl=100;
end
if ~exist('stationary','var') || isempty(stationary)
    stationary=false;
end
if ~exist('mdl_opt','var') || isempty(mdl_opt)
    mdl_opt=0;
end
if ~exist('upthypr','var') || isempty(upthypr)
    upthypr=1; % 0: no update; 1: sample; 2; optimize; 3: optimize jointly
end
if ~exist('sampleM','var') || isempty(sampleM)
    sampleM=true;
end
if ~exist('intM','var') || isempty(intM)
    intM=(mdl_opt==1);
end
if ~exist('hold_out','var') || isempty(hold_out)
    hold_out=false;
end
if ~exist('spdapx','var') || isempty(spdapx)
    spdapx=false; % use speed up (e.g. parfor) or approximation; number of workers when it is numeric
end
if ~exist('seedNO','var') || isempty(seedNO)
    seedNO=2018;
end

% Random Numbers...
seed=RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% model options
models={'nonsepstat'};
alg_name='MCMC';
if upthypr>=2
    alg_name=['opt',alg_name];
    if upthypr==3
        alg_name=['jt',alg_name];
    end
end
% for mdl_opt=1:2;
% compatible setting for M
if ~intM
    sampleM=true;
end

% setting for simulation
N=[200,100]; % discretization sizes for space and time domains
% N=[20,500]; % discretization sizes for space and time domains
K=num_trl; % number of trials
d=1; % space dimension
% load or simulate data
[x,t,y]=generate_data(N,K,d,stationary,seedNO);

% thin the mesh
thin=[50,1];
% thin=[5,1];
x=x(1:thin(1):end,:); t=t(1:thin(2):end); y=y(1:thin(1):end,1:thin(2):end,:);
% N=[size(x,1),size(t,1)];
I=size(x,1); J=size(t,1); L=I;
if hold_out
    % hold out some data (15% in time direction) for prediction
    t(end-floor(J*.2)+1:2:end)=[]; y(:,end-floor(J*.2)+1:2:end,:)=[];
    t(end-floor(J*.05)+1:end)=[]; y(:,end-floor(J*.05)+1:end,:)=[];
    I=size(x,1); J=size(t,1); L=I;
end

% setup parallel pool
if spdapx
    if isnumeric(spdapx)
        num_wkr = spdapx;
    else
        num_wkr = 10;
    end
    clst = parcluster('local');
    max_wkr= clst.NumWorkers;
    poolobj=gcp('nocreate');
    if isempty(poolobj) && num_wkr>1
        poolobj=parpool('local',min([num_wkr,max_wkr]));
    end
end

% parameters of kernel
s=2; % smoothness
kappa=1.2; % decaying rate for dynamic eigenvalues
% kappa=.2; % decaying rate for dynamic eigenvalues
% spatial kernel
jit_x=1e-6;
ker{1}=feval([stgp_ver,'.GP'],x,[],[],s,L,jit_x,true);
% temporal kernels
jit_t=1e-6;
ker{2}=feval([stgp_ver,'.GP'],t,[],[],s,L,jit_t,true,spdapx);
% for hyper-GP
ker{3}=ker{2};
% specify (hyper)-priors
% (a,b) in inv-gamma priors for sigma2_*, * = eps, t, u
% (m,V) in (log) normal priors for eta_*, (eta=log-rho), * = x, t, u
switch mdl_opt
    case {0,1}
        a=ones(1,3); b=[5e0,1e1,1e1];
        m=zeros(1,3); V=[1e-1,1e-1,1e-2];
    case 2
        if ~hold_out
            a=ones(1,3); b=[1e-1,1e0,5e0];
            m=zeros(1,3); V=[1,1,1]; % K=100: V=[1,1,1]; K=1000: V=[1,1,2];
%             a=ones(1,3); b=[1e-1,1e0,5e0];
%             m=zeros(1,3); V=[1e-1,.1,.5];
        else
            a=ones(1,3); b=[1e-1,1e0,1e1];
            m=zeros(1,3); V=[1,1,log10(K)+1]; % K=100: V=[1,1,3]; K=1000: V=[1,1,4];
        end
end

% allocation to save
Nsamp=1e4; burnrate=0.2; thin=2;
Niter=Nsamp*thin; NBurnIn=floor(Niter*burnrate); Niter=Niter+ NBurnIn;
if upthypr
    samp_sigma2=zeros(Nsamp,3);
    samp_eta=zeros(Nsamp,3);
else
    samp_sigma2=[]; samp_eta=[];
end
samp_Lambda=zeros(Nsamp,J,L);
if max([I,J])>Nsamp
    samp_M=zeros(I,J,2);
else
    samp_M=zeros(Nsamp,I,J);
end

if upthypr==3
    engy=zeros(Niter,3);
    nl_sigma2=0; nl_eta=0; 
else
    engy=zeros(Niter,7);
    nl_sigma2=zeros(1,3); nl_eta=zeros(1,3);
end

% initializatioin
% sigma2=1./gamrnd(a,1./b); 
sigma2=invgamrnd(a,b);
% sigma2(1)=sigma2(1).^(mdl_opt~=2);
sigma2(1)=sigma2(1).*(mdl_opt~=2);
eta=normrnd(m,sqrt(V));
for k=1:length(ker)
    ker{k}=ker{k}.update(sigma2(k)^(k~=1),exp(eta(k)));
end
if isnumeric(kappa)
    gamma=(1:L).^(-kappa/2); % induce Cauchy sequence only if kappa>1
elseif contains(kappa,'eigCx')
    [~,gamma]=ker{1}.eigs(L); % eigenvalues of C_x
    gamma=sqrt(gamma');
else
    gamma=ones(1,L);
end
Lambda=ker{3}.rnd([],L).*gamma;
% stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},Lambda,kappa,mdl_opt,[],[],spdapx); [ker{1:2}]=deal([]);
% stgp=nonsepstat(x,t,sigma2(2),exp(eta(1)),1,L,jit_x,[],spdapx);
stgp=nonsepstat(x,t,sigma2(2),exp(eta(1)),1,L,jit_x,0,[],spdapx);
[ker{1:2}]=deal([]);
M=stgp.sample_priM; % (I,J)
% set up marginal stgp kernel
mgC=feval([stgp_ver,'.mg'],stgp,K,sigma2(1),L); mgC.stgp.opt=mdl_opt; mgC.stgp.ker_opt=mgC.stgp.jtkers{mgC.stgp.opt+1}; stgp=[];
if mgC.stgp.opt==2
    mgC.isub=feval([stgp_ver,'.isub'],mgC.stgp,mgC.K,mgC.L,mgC.store_eig);
end
% constant updates
dlta=[I*J*K,I*J*K^(mdl_opt==2),J*L]./2;
alpha=a+dlta;
% center y and reset M
if mdl_opt==2
%     y=y-mean(y,3);
%     M=0.*M;
    M=mean(y,3);
    sampleM=false;
end

% try
%     % load best initials
%     load(['./est_',alg_name,repmat('_intM_',1,intM),'_I',num2str(I),'_J',num2str(J),'_L',num2str(L),'_d',num2str(d),'_',repmat('non',1,~stationary),'stationary','.mat']);
%     tr=find([100,1000]==K);
%     sigma2=sigma2_estm{mdl_opt+1,tr}; eta=eta_estm{mdl_opt+1,tr}; Lambda=Lambda_estm{mdl_opt+1,tr}; M=M_estm{mdl_opt+1,tr};
    optimf=[];
%     fprintf('Read optimal initials.\n');
% catch
%     % optimize initial location
%     [sigma2,eta,Lambda,M,mgC,ker{3},optimf]=opt4ini(sigma2,eta,Lambda,M,mgC,ker{3},y,a,b,m,V,[1,1,1,0],intM,upthypr==3,20);
% end
% record optimized initialization
optini=struct('sigma2',sigma2, 'eta',eta, 'Lambda',Lambda, 'M',M, 'optimf',optimf);
% optimization options
if upthypr
    opts_unc=optimoptions('fminunc','Algorithm','quasi-newton','display','off','MaxIterations',100);
    opts_con=optimoptions('fmincon','Algorithm','sqp','display','off','MaxIterations',100);
end

% MCMC
fprintf('Running MCMC for model %s with%s M integrated on %sstationary simulation data...\n',join([repmat('0',1,mdl_opt==0),repmat('I',1,mdl_opt)]), repmat('out',1-intM), repmat('non',1,~stationary));
prog=0.05:0.05:1;
tic;
for iter=1:Niter
    
    % update sigma2
    if intM
        if mdl_opt~=2
            % sigma2_eps
            logf{1}=@(q)logpost_sigma2([q,sigma2(2)],mgC,y,a(1),b(1),1);
            if upthypr==1
                [sigma2(1),l_sigma2(1)]=slice(sigma2(1),logf{1}(sigma2(1)),logf{1},[],[],[0,+Inf]);
            elseif upthypr==2
                [sigma2(1),nl_sigma2(1)]=fmincon(@(q)-logf{1}(q),sigma2(1),[],[],[],[],0,[],[],opts_con);
            end
        end
        % sigma2_t
        logf{2}=@(q)logpost_sigma2([sigma2(1),q],mgC,y,a(2),b(2),2);
        if upthypr==1
            [sigma2(2),l_sigma2(2)]=slice(sigma2(2),logf{2}(sigma2(2)),logf{2},[],[],[jit_t,+Inf]);
%             nl_sigma2=-l_sigma2;
        elseif upthypr==2
            [sigma2(2),nl_sigma2(2)]=fmincon(@(q)-logf{2}(q),sigma2(2),[],[],[],[],jit_t,[],[],opts_con);
        end
        % joint optmize
        if upthypr==3
            logF=@(q)logpost_sigma2(q,mgC,y,a(1:2),b(1:2),1:2);
            [sigma2(1:2),nl_sigma2]=fmincon(@(q)-sum(logF(q)),sigma2(1:2),[],[],[],[],zeros(2,1),[],[],opts_con);
        end
    else
        switch mdl_opt
            case {0,1}
                dltb(1)=0.5.*sum((y-M).^2,'all');
                dltb(2)=0.5.*(M(:)'*mgC.stgp.solve(M(:))).*sigma2(2);
            case 2
                dltb(1)=0;
%                 dltb(2)=0.5.*sum(M.*mgC.stgp.C_t.solve(M')','all').*sigma2(2);
                y_ctr=reshape(y-M,mgC.stgp.N,[]);
                dltb(2)=0.5.*sum(y_ctr.*mgC.stgp.solve(y_ctr),'all').*sigma2(2);
        end
    end
    % sigma2_u
    U=mgC.stgp.scale_Lambda(Lambda);
    dltb(3)=0.5.*sum(U.*ker{3}.solve(U),'all').*sigma2(3);
    beta=b+dltb;
    idx2upd=(intM*3+(1-intM)*(1+(mdl_opt==2))):3;
    if upthypr==1
%         sigma2_=1./gamrnd(alpha,1./beta); % sample
        sigma2_=invgamrnd(alpha,beta);
        l_sigma2_=log(gampdf(1./sigma2_,alpha,1./beta))-2*log(sigma2_);
        sigma2(idx2upd)=sigma2_(idx2upd); l_sigma2(idx2upd)=l_sigma2_(idx2upd);
    elseif upthypr>=2
        sigma2_=beta./(alpha+1); % optimize
        nl_sigma2_=-(log(gampdf(1./sigma2_,alpha,1./beta))-2*log(sigma2_));
        sigma2(idx2upd)=sigma2_(idx2upd); nl_sigma2(idx2upd)=nl_sigma2_(idx2upd);
    end
    if upthypr==1
        nl_sigma2=-l_sigma2;
    elseif upthypr==3
        nl_sigma2=sum(nl_sigma2);
    end
    
    if upthypr
        % update kernels
        ker{3}=ker{3}.update(sigma2(3));
        mgC=mgC.update(mgC.stgp.update([],mgC.stgp.C_t.update(sigma2(2))),sigma2(1));
    end
    
    % setting for eta and Lambda
    if intM
        M=[];
    end
    
    % update eta
    % eta_x
    logf{1}=@(q)logpost_eta([q,eta(2)],mgC,y,m(1),V(1),1,M);
    if upthypr==1
        [eta(1),l_eta(1)]=slice(eta(1),logf{1}(eta(1)),logf{1});
    elseif upthypr==2
        [eta(1),nl_eta(1)]=fminunc(@(q)-logf{1}(q),eta(1),opts_unc);
    end
%     % eta_t
%     logf{2}=@(q)logpost_eta([eta(1),q],mgC,y,m(2),V(2),2,M);
%     if upthypr==1
%         [eta(2),l_eta(2)]=slice(eta(2),logf{2}(eta(2)),logf{2});
%     elseif upthypr==2
%         [eta(2),nl_eta(2)]=fminunc(@(q)-logf{2}(q),eta(2),opts_unc);
%     end
%     % eta_u
%     logf{3}=@(q)logpost_eta(q,mgC,y,m(3),V(3),3,M,ker{3});
%     if upthypr==1
%         [eta(3),l_eta(3)]=slice(eta(3),logf{3}(eta(3)),logf{3});
%         nl_eta=-l_eta;
%     elseif upthypr==2
%         [eta(3),nl_eta(3)]=fminunc(@(q)-logf{3}(q),eta(3),opts_unc);
%     end
%     % joint optmize
%     if upthypr==3
%         logF=@(q)logpost_eta(q,mgC,y,m,V,1:3,M,ker{3});
%         [eta,nl_eta]=fminunc(@(q)-sum(logF(q)),eta,opts_unc);
%     end
    
    if upthypr
        % update kernels
        ker{3}=ker{3}.update([],exp(eta(3)));
        mgC=mgC.update(mgC.stgp.update(mgC.stgp.C_x.update([],exp(eta(1))),mgC.stgp.C_t.update([],exp(eta(2)))));
    end
    
    if 0 && mgC.stgp.opt
        % sample Lambda with ESS
        logLik_Lambda=@(q)loglik_Lambda(q,mgC,y,M);
        prirnd_Lambda=@()mgC.stgp.scale_Lambda(ker{3}.rnd([],L),'dn');
        [Lambda,l_Lambda] = ESS(Lambda,logLik_Lambda(Lambda),prirnd_Lambda,logLik_Lambda);
        % update marginal kernel
        if sampleM && isgpuarray(Lambda)
            mgC=mgC.update(mgC.stgp.update([],[],gather(Lambda)));
        else
            mgC=mgC.update(mgC.stgp.update([],[],Lambda));
        end
    else
        l_Lambda=0;
    end
    
    % sample M directly
    if sampleM
        M=mgC.sample_postM(y);
    end
    
    % display sampling progress
    if ismember(iter,floor(Niter.*prog))
        fprintf('%.0f%% iterations completed.\n',100*iter/Niter);
    end
    
    % burn-in complete
    if(iter==NBurnIn)
        fprintf('Burn in completed!\n');
    end
    
    % save samples after burn-in
    engy(iter,:)=[nl_sigma2,nl_eta,-l_Lambda];
    if(iter>NBurnIn) && mod(iter-NBurnIn,thin) == 0
        NO_sav=ceil((iter-NBurnIn)/thin);
        if upthypr
            samp_sigma2(NO_sav,:)=sigma2;
            samp_eta(NO_sav,:)=eta;
        end
        if mgC.stgp.opt
            samp_Lambda(NO_sav,:,:)=Lambda;
        end
        if sampleM
            if max([I,J])>Nsamp
                samp_M(:,:,1)=samp_M(:,:,1)+M;
                samp_M(:,:,2)=samp_M(:,:,2)+M.^2;
            else
                samp_M(NO_sav,:,:)=M;
            end
        end
    end

end
if sampleM
    if max([I,J])>Nsamp
        samp_M=samp_M./Nsamp; samp_M(:,:,2)=sqrt(samp_M(:,:,2)-samp_M(:,:,1).^2);
    end
else
    samp_M=[];
end

% count time
time=toc;
% save
time_lbl=regexprep(num2str(fix(clock)),'    ','_');
keywd=[models{1},'_I',num2str(I),'_J',num2str(J),'_K',num2str(K),'_L',num2str(L),'_d',num2str(d),'_',repmat('non',1,~stationary),'stationary','_seedNO',num2str(seedNO),'_',time_lbl];
f_name=['simSTproc_',alg_name,repmat('_intM',1,intM),'_',keywd];
folder = './result/';
if exist(folder,'dir')~=7
    mkdir(folder);
end
save([folder,f_name,'.mat'],'seedNO','models','mdl_opt','upthypr','sampleM','intM','spdapx',...
                            'I','J','K','L','d','x','t','y','s','kappa','ker','mgC','a','b','m','V',...
                            'Nsamp','NBurnIn','thin','Niter','optini','samp_sigma2','samp_eta','samp_Lambda','samp_M','engy','time','-v7.3');
% summarize
fprintf('\nIt takes %.2f seconds to collect %e samples after thinning %d.\n', time,Nsamp,thin);
% shutdown parpool
if exist('poolobj','var') && ~isempty(poolobj)
    delete(poolobj);
end
end