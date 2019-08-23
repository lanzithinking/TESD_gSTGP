% inference of a simulated spatio-temporal process model using MCMC
% for multiple-trials
% optimize sigma2, eta or sample them using slice sampler,
% sample Lambda, (M) using ESS, (Gibbs) resp.
% hold out partial data for prediction

function []=sim_STproc_MCMC_holdout_f(mdl_opt,opthypr,jtupt,sampleM)
addpath('../util/','../sampler/');
% Random Numbers...
seedNO=2018;
seed=RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% settings
if ~exist('mdl_opt','var') || isempty(mdl_opt)
    mdl_opt=2;
end
if ~exist('opthypr','var') || isempty(opthypr)
    opthypr=false;
end
if ~exist('jtupt','var') || isempty(jtupt)
    jtupt=false;
end
if ~exist('sampleM','var') || isempty(sampleM)
    sampleM=true;
end

% model options
models={'kron_prod','kron_sum'};
% for mdl_opt=1:2;

% setting for simulation
N=[200,100]; % discretization sizes for space and time domains
K=100; % number of trials
d=1; % space dimension
% load or simulate data
[x,t,y]=generate_data(N,K,d,seedNO);

% thin the mesh
thin=[50,1];
x=x(1:thin(1):end,:); t=t(1:thin(2):end); y=y(1:thin(1):end,1:thin(2):end,:);
% N=[size(x,1),size(t,1)];
I=size(x,1); J=size(t,1); L=I;
% hold out some data (15% in time direction)
t(end-floor(J*.2)+1:2:end)=[]; y(:,end-floor(J*.2)+1:2:end,:)=[];
t(end-floor(J*.05)+1:end)=[]; y(:,end-floor(J*.05)+1:end,:)=[];
I=size(x,1); J=size(t,1); L=I;

% parameters of kernel
s=2; % smoothness
kappa=1.2; % decaying rate for dynamic eigenvalues
% spatial kernel
if d==1
    dist_x=pdist2(x,x,@(XI,XJ)abs(bsxfun(@minus,XI,XJ)).^s);
else
    dist_x=sum(abs(reshape(x,I,1,[])-reshape(x,1,I,[])).^s,3);
end
jit_x=1e-6.*speye(I);
ker{1}.s=s;ker{1}.dist=dist_x;ker{1}.jit=jit_x;
% temporal kernels
dist_t=pdist2(t,t,@(XI,XJ)abs(bsxfun(@minus,XI,XJ)).^s);
jit_t=1e-6.*speye(J);
ker{2}.s=s;ker{2}.dist=dist_t;ker{2}.jit=jit_t;
% for hyper-GP
ker{3}=ker{2}; ker{3}.kappa=kappa;
% specify (hyper)-priors
% (a,b) in inv-gamma priors for sigma2_*, * = eps, t, tilt
% (m,V) in (log) normal priors for eta_*, (eta=log-rho), * = x, t, tilt
switch mdl_opt
    case 1
        a=ones(1,3); b=[5e0,1e1,1e1];
        m=zeros(1,3); V=[1e-1,1e-1,1e-2];
    case 2
        a=ones(1,3); b=[1e-1,1e0,1e1];
        m=zeros(1,3); V=[1,1,3]; % K=100: V=[1,1,3]; K=1000: V=[1,1,4];
end

% allocation to save
Nsamp=1e4; burnrate=0.2; thin=2;
Niter=Nsamp*thin; NBurnIn=floor(Niter*burnrate); Niter=Niter+ NBurnIn;
samp_sigma2=zeros(Nsamp,3);
samp_eta=zeros(Nsamp,3);
samp_M=zeros(Nsamp,I,J);
samp_Lambda=zeros(Nsamp,J,L);
if opthypr && jtupt
    engy=zeros(Niter,3);
else
    engy=zeros(Niter,7);
end

% initializatioin
sigma2=1./gamrnd(a,1./b);
eta=normrnd(m,sqrt(V));
for k=1:length(ker)
    ker{k}.C=sigma2(k)^(k~=1).*(exp(-.5.*ker{k}.dist.*exp(-ker{k}.s.*eta(k)))+ker{k}.jit);
end
Lambda=matnrnd(zeros(J,L),ker{3}.C).*(1:L).^(-kappa/2);
switch mdl_opt
    case 1
        C_z=STGP(ker{1}.C,ker{2}.C,Lambda,mdl_opt).get_jtker();
        M=reshape(mvnrnd(zeros(1,I*J),C_z),I,J);
    case 2
        M=matnrnd(zeros(I,J),[],ker{2}.C); % (I,J)
end
% constant updates
dlta=[I*J*K,I*J,J*L]./2;
alpha=a+dlta;

% optimize initial location
[sigma2,eta,M,Lambda,objf]=opt4ini(sigma2,eta,M,Lambda,y,ker,a,b,m,V,models{mdl_opt},[1,1,1,0],jtupt,20);
% update kernels
for k=1:length(ker)
    ker{k}.C=sigma2(k)^(k~=1).*(exp(-.5.*ker{k}.dist.*exp(-ker{k}.s.*eta(k)))+ker{k}.jit);
end
stgp=STGP(ker{1}.C,ker{2}.C,Lambda,mdl_opt);
% record optimized initialization
optini=struct('sigma2',sigma2, 'eta',eta, 'M',M, 'Lambda',Lambda);
% optimization options
if opthypr
    opts_unc=optimoptions('fminunc','Algorithm','quasi-newton','display','off','MaxIterations',100);
    opts_con=optimoptions('fmincon','Algorithm','sqp','display','off','MaxIterations',100);
end
if mdl_opt==1
    stgp.trtdeg=false;
end

% MCMC
fprintf('Running MCMC with model %s ...\n',repmat('I',1,mdl_opt));
prog=0.05:0.05:1;
tic;
for iter=1:Niter
    
    % display sampling progress and online acceptance rate
    if ismember(iter,floor(Niter.*prog))
        fprintf('%.0f%% iterations completed.\n',100*iter/Niter);
    end
    
    % normalize C_t in stgp before updating sigma2
    stgp.C_t=stgp.C_t./sigma2(2);
    [C_z,C_xt]=stgp.get_jtker();
    
    % update sigma2
    switch mdl_opt
        case 1
            C0_pri=C_z; C0_lik=speye(stgp.I*stgp.J);
            % sigma2_eps
            logf{1}=@(q)logpost_sigma2([q,sigma2(2)],C0_pri,C0_lik,y,a(1),b(1),1);
            if ~opthypr
                [sigma2(1),l_sigma2(1)]=slice(sigma2(1),logf{1}(sigma2(1)),logf{1},[],[],[0,+Inf]);
            elseif ~jtupt
                [sigma2(1),nl_sigma2(1)]=fmincon(@(q)-logf{1}(q),sigma2(1),[],[],[],[],0,[],[],opts_con);
            end
        case 2
            C0_pri=sparse(kron(stgp.C_t,speye(stgp.I))); C0_lik=C_xt;
    end
    % sigma2_t
    logf{2}=@(q)logpost_sigma2([sigma2(1),q],C0_pri,C0_lik,y,a(2),b(2),2);
    if ~opthypr
        [sigma2(2),l_sigma2(2)]=slice(sigma2(2),logf{2}(sigma2(2)),logf{2},[],[],[0,+Inf]);
%         nl_sigma2=-l_sigma2;
    elseif ~jtupt
        [sigma2(2),nl_sigma2(2)]=fmincon(@(q)-logf{2}(q),sigma2(2),[],[],[],[],0,[],[],opts_con);
    end
    % joint optmize
    if opthypr && jtupt
        logF=@(q)logpost_sigma2(q,C0_pri,C0_lik,y,a(1:2),b(1:2),1:2);
        [sigma2(1:2),nl_sigma2]=fmincon(@(q)-sum(logF(q)),sigma2(1:2),[],[],[],[],zeros(2,1),[],[],opts_con);
    end
    % sigma2_tilt
    Lambda_til=Lambda.*(1:L).^(ker{3}.kappa/2);
    quad=Lambda_til.*(ker{3}.C\Lambda_til);
    dltb(3)=0.5.*sum(quad(:)).*sigma2(3);
    beta=b+dltb;
    if ~opthypr
        sigma2(3)=1./gamrnd(alpha(3),1./beta(3)); % sample
        l_sigma2(3)=log(gampdf(1./sigma2(3),alpha(3),1./beta(3)))-2*log(sigma2(3));
    else
        sigma2(3)=beta(3)./(alpha(3)+1); % optimize
        nl_sigma2(3)=-(log(gampdf(1./sigma2(3),alpha(3),1./beta(3)))-2*log(sigma2(3)));
    end
    if ~opthypr
        nl_sigma2=-l_sigma2;
    elseif jtupt
        nl_sigma2=sum(nl_sigma2);
    end
    
    % update eta
    % eta_x
    logf{1}=@(q)logpost_eta([q,eta(2)],sigma2(1:2),ker(1:2),stgp,y,m(1),V(1),1);
    if ~opthypr
        [eta(1),l_eta(1)]=slice(eta(1),logf{1}(eta(1)),logf{1});
    elseif ~jtupt
        [eta(1),nl_eta(1)]=fminunc(@(q)-logf{1}(q),eta(1),opts_unc);
    end
    % eta_t
    logf{2}=@(q)logpost_eta([eta(1),q],sigma2(1:2),ker(1:2),stgp,y,m(2),V(2),2);
    if ~opthypr
        [eta(2),l_eta(2)]=slice(eta(2),logf{2}(eta(2)),logf{2});
    elseif ~jtupt
        [eta(2),nl_eta(2)]=fminunc(@(q)-logf{2}(q),eta(2),opts_unc);
    end
    % eta_tilt
    logf{3}=@(q)logpost_eta(q,sigma2(3),ker(3),stgp,y,m(3),V(3),3);
    if ~opthypr
        [eta(3),l_eta(3)]=slice(eta(3),logf{3}(eta(3)),logf{3});
        nl_eta=-l_eta;
    elseif ~jtupt
        [eta(3),nl_eta(3)]=fminunc(@(q)-logf{3}(q),eta(3),opts_unc);
    end
    % joint optmize
    if opthypr && jtupt
        logF=@(q)logpost_eta(q,sigma2,ker,stgp,y,m,V,1:3);
        [eta,nl_eta]=fminunc(@(q)-sum(logF(q)),eta,opts_unc);
    end
    
    % update kernels
    for k=1:length(ker)
        ker{k}.C=sigma2(k)^(k~=1).*(exp(-.5.*ker{k}.dist.*exp(-ker{k}.s.*eta(k)))+ker{k}.jit);
    end
    stgp.C_x=ker{1}.C; stgp=stgp.get_spat(); stgp.C_t=ker{2}.C;
    
    % sample Lambda with ESS
    logLik_Lambda=@(q)loglik_Lambda(q,sigma2(1),stgp,y);
    prirnd_Lambda=@()matnrnd(zeros(J,L),ker{3}.C).*(1:L).^(-ker{3}.kappa/2);
    [Lambda,l_Lambda] = ESS(Lambda,logLik_Lambda(Lambda),prirnd_Lambda,logLik_Lambda);
    % update Lambda
    stgp.Lambda=Lambda;
    
    % sample M directly
    if sampleM
        [mu,chol_Sigma]=stgp.post_mean(y,sigma2(1));
        M=mu+chol_Sigma*randn(I*J,1);
        M=reshape(M,I,J);
    end
    
    % burn-in complete
    if(iter==NBurnIn)
        fprintf('Burn in completed!\n');
    end
    
    % save samples after burn-in
    engy(iter,:)=[nl_sigma2,nl_eta,-l_Lambda];
    if(iter>NBurnIn) && mod(iter-NBurnIn,thin) == 0
        samp_sigma2(ceil((iter-NBurnIn)/thin),:)=sigma2;
        samp_eta(ceil((iter-NBurnIn)/thin),:)=eta;
        samp_Lambda(ceil((iter-NBurnIn)/thin),:,:)=Lambda;
        if sampleM
            samp_M(ceil((iter-NBurnIn)/thin),:,:)=M;
        end
    end

end
if ~sampleM
    samp_M=[];
end

% count time
time=toc;
% save
time_lbl=regexprep(num2str(fix(clock)),'    ','_');
alg_name='MCMC';
if opthypr
    alg_name=['opt',alg_name];
    if jtupt
        alg_name=['jt',alg_name];
    end
end
f_name=['sim_STproc_',alg_name,'_',models{mdl_opt},'_I',num2str(I),'_J',num2str(J),'_K',num2str(K),'_L',num2str(L),'_d',num2str(d),'_',time_lbl];
folder = './result/';
if exist(folder,'dir')~=7
    mkdir(folder);
end
save([folder,f_name,'.mat'],'seedNO','opthypr','jtupt','sampleM','models','mdl_opt','I','J','K','L','d','x','t','y','ker','a','b','m','V',...
                            'Nsamp','NBurnIn','thin','Niter','optini','samp_sigma2','samp_eta','samp_M','samp_Lambda','engy','time','-v7.3');
% summarize
fprintf('\nIt takes %.2f seconds to collect %e samples after thinning %d.\n', time,Nsamp,thin);
end