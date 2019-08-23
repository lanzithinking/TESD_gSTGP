% inference of generalized STGP model for analyzing dynamic brain images of
% Alzheimer's disease patients
% for multiple-trials
% optimize sigma2, eta or sample them using slice sampler,
% sample Lambda, (M) using ESS, (Gibbs) resp.

function []=dynBrainIM_MCMC_fclst(grp_opt,mdl_opt,opthypr,jtupt,sampleM,intM,use_para)
% sufx={'','_mtimesx','_gpu'};
% stgp_ver=['STGP',sufx{2}];
% % addpath('../util/','../sampler/');
% % addpath(['../util/+',stgp_ver,'/']);
% if contains(stgp_ver,'mtimesx')
%     addpath('../util/mtimesx/');
% end
% addpath('../util/Image Graphs/');
if ~isdeployed
    addpath ~/STGP/code/util/;
    addpath ~/STGP/code/sampler/;
    addpath ~/STGP/code/util/STGP_mtimesx/;
    addpath ~/STGP/code/util/mtimesx/;
    addpath ~/STGP/code/util/Image_Graphs/;
end
if isdeployed
    grp_opt=str2num(grp_opt);
    mdl_opt=str2num(mdl_opt);
%     if contains('true',intM)
%         intM=true;
%     end
%     if contains('false',intM)
%         intM=false;
%     end
    opthypr=str2num(opthypr);
    jtupt=str2num(jtupt);
    sampleM=str2num(sampleM);
    intM=str2num(intM);
end
% Random Numbers...
seedNO=2018;
seed=RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% settings
if ~exist('grp_opt','var') || isempty(grp_opt)
    grp_opt=2;
end
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
if ~exist('intM','var') || isempty(intM)
    intM=(mdl_opt==1);
end
if ~exist('use_para','var') || isempty(use_para)
    use_para=false;
end

% large data storage path prefix
store_prefix='~/project-stat/STGP/code/AD/';

% settings
% for the data
types={'PET','MRI'};
typ=types{1};
groups={'CN','MCI','AD'};
grp=groups{grp_opt};
dur=[5,6,4];
J=dur(grp_opt);
stdtimes={[0:.5:1,2:3]',[0:.5:1.5,2:3]',[0:.5:1,2]'};
d=2;
sec=48;

% model options
models={'kron_prod','kron_sum'};
% for mdl_opt=2;
% compatible setting for M
if ~intM
    sampleM=true;
end
% obtain AD-PET data set
loc=[store_prefix,'data/'];
[t,y]=read_data(typ,grp,J,loc);
% normalize time
tt=datetime(t);
tt=datenum(tt-repmat(tt(1,:),size(tt,1),1));
t=tt./365;
% remove irregular observations
rmind=sum(abs(t-stdtimes{grp_opt})>.55)>0;
if any(rmind)
    fprintf('%d subject(s) removed!\n',sum(rmind));
end
% convert it to common time-frame % todo: extend the model to handle
% different times
t=mean(t(:,~rmind),2);
% select one section and scale image intensity
yy=cell2mat(shiftdim(y,-3));
if d==2
    yy=squeeze(yy(:,:,sec,:,~rmind));
end
y=double(yy)./32767; yy=[];
sz_y=size(y); imsz=sz_y(1:2);
% reshape y to be (I,J,K)
y=reshape(y,[],sz_y(3),sz_y(4));
N=prod(imsz);

% obtain dimensions
I=N;
K=size(y,3);
L=min([I,100]);
% if I*J>1e4
%     use_para=true;
% end
% obtain ROI
roi=get_roipoi(.75,'stackt',[],loc);

% parameters of kernel
s=2; % smoothness
kappa=.2; % decaying rate for dynamic eigenvalues
% graph kernel
g.w=1; g.size=imsz; g.mask=roi{grp_opt};
jit_g=1e-6;
% ker{1}=feval([stgp_ver,'.GL'],g,[],[],s,L,jit_g,true);
ker{1}=GL(g,[],[],s,L,jit_g,true);
% temporal kernels
jit_t=1e-6;
% ker{2}=feval([stgp_ver,'.GP'],t,[],[],s,L,jit_t,true);
ker{2}=GP(t,[],[],s,L,jit_t,true);
ker{3}=ker{2}; % for hyper-GP
% specify (hyper)-priors
% (a,b) in inv-gamma priors for sigma2_*, * = eps, t, u
% (m,V) in (log) normal priors for eta_*, (eta=log-rho), * = x, t, u
switch mdl_opt
    case 1
        a=ones(1,3); b=[5e0,1e1,1e1];
        m=zeros(1,3); V=[1e-1,1e-1,1e-2];
    case 2
        a=ones(1,3); b=[1e-1,1e0,1e-1];
        m=zeros(1,3); V=[1e-1,1,1];
end

% allocation to save
Nsamp=1e4; burnrate=0.2; thin=2;
Niter=Nsamp*thin; NBurnIn=floor(Niter*burnrate); Niter=Niter+ NBurnIn;
samp_sigma2=zeros(Nsamp,3);
samp_eta=zeros(Nsamp,3);
if max([I,J])>Nsamp
    samp_M=zeros(I,J,2);
else
    samp_M=zeros(Nsamp,I,J);
end
samp_Lambda=zeros(Nsamp,J,L);
if opthypr && jtupt
    engy=zeros(Niter,3);
else
    engy=zeros(Niter,7);
end

% initializatioin
% sigma2=1./gamrnd(a,1./b); 
sigma2=invgamrnd(a,b);
sigma2(1)=sigma2(1).^(mdl_opt==1);
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
% stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},Lambda,mdl_opt); [ker{1:2}]=deal([]);
stgp=hd(ker{1},ker{2},Lambda,kappa,mdl_opt); [ker{1:2}]=deal([]);
M=stgp.sample_priM; % (I,J)
% set up marginal stgp kernel
% mgC=feval([stgp_ver,'.mg'],stgp,K,sigma2(1),L); stgp=[];
mgC=mg(stgp,K,sigma2(1),L); stgp=[];
% constant updates
dlta=[I*J*K,I*J,J*L]./2;
alpha=a+dlta;

% setup parallel pool
if use_para
    clst = parcluster('local');
    max_wkr= clst.NumWorkers;
    poolobj=gcp('nocreate');
    if isempty(poolobj)
        poolobj=parpool('local',min([10,max_wkr]));
    end
end

% optimize initial location
[sigma2,eta,M,Lambda,mgC,ker{3},optimf]=opt4ini(sigma2,eta,M,Lambda,mgC,ker{3},y,a,b,m,V,[1,1,1,0],intM,jtupt,20);
% record optimized initialization
optini=struct('sigma2',sigma2, 'eta',eta, 'M',M, 'Lambda',Lambda,'optimf',optimf);
% optimization options
if opthypr
    opts_unc=optimoptions('fminunc','Algorithm','quasi-newton','display','off','MaxIterations',100);
    opts_con=optimoptions('fmincon','Algorithm','sqp','display','off','MaxIterations',100);
end

% MCMC
fprintf('Running MCMC for model %s with%s M integrated on %s data...\n',repmat('I',1,mdl_opt), repmat('out',1-intM), grp);
prog=0.05:0.05:1;
tic;
for iter=1:Niter
    
    % update sigma2
    if intM
        if mdl_opt==1
            % sigma2_eps
            logf{1}=@(q)logpost_sigma2([q,sigma2(2)],mgC,y,a(1),b(1),1);
            if ~opthypr
                [sigma2(1),l_sigma2(1)]=slice(sigma2(1),logf{1}(sigma2(1)),logf{1},[],[],[0,+Inf]);
            elseif ~jtupt
                [sigma2(1),nl_sigma2(1)]=fmincon(@(q)-logf{1}(q),sigma2(1),[],[],[],[],0,[],[],opts_con);
            end
        end
        % sigma2_t
        logf{2}=@(q)logpost_sigma2([sigma2(1),q],mgC,y,a(2),b(2),2);
        if ~opthypr
            [sigma2(2),l_sigma2(2)]=slice(sigma2(2),logf{2}(sigma2(2)),logf{2},[],[],[jit_t,+Inf]);
%             nl_sigma2=-l_sigma2;
        elseif ~jtupt
            [sigma2(2),nl_sigma2(2)]=fmincon(@(q)-logf{2}(q),sigma2(2),[],[],[],[],jit_t,[],[],opts_con);
        end
        % joint optmize
        if opthypr && jtupt
            logF=@(q)logpost_sigma2(q,mgC,y,a(1:2),b(1:2),1:2);
            [sigma2(1:2),nl_sigma2]=fmincon(@(q)-sum(logF(q)),sigma2(1:2),[],[],[],[],zeros(2,1),[],[],opts_con);
        end
    else
        switch mdl_opt
            case 1
                y_ctr=y-M;
                dltb(1)=0.5.*sum(y_ctr(:).^2);
                dltb(2)=0.5.*(M(:)'*mgC.stgp.solve(M(:))).*sigma2(2);
            case 2
                dltb(1)=0;
                quad=M.*mgC.stgp.C_t.solve(M')';
                dltb(2)=0.5.*sum(quad(:)).*sigma2(2);
        end
    end
    % sigma2_u
    U=mgC.stgp.scale_Lambda(Lambda);
    quad=U.*ker{3}.solve(U);
    dltb(3)=0.5.*sum(quad(:)).*sigma2(3);
    beta=b+dltb;
    idx2upd=(intM*3+(1-intM)*mdl_opt):3;
    if ~opthypr
%         sigma2_=1./gamrnd(alpha,1./beta); % sample
        sigma2_=invgamrnd(alpha,beta);
        l_sigma2_=log(gampdf(1./sigma2_,alpha,1./beta))-2*log(sigma2_);
        sigma2(idx2upd)=sigma2_(idx2upd); l_sigma2(idx2upd)=l_sigma2_(idx2upd);
    else
        sigma2_=beta./(alpha+1); % optimize
        nl_sigma2_=-(log(gampdf(1./sigma2_,alpha,1./beta))-2*log(sigma2_));
        sigma2(idx2upd)=sigma2_(idx2upd); nl_sigma2(idx2upd)=nl_sigma2_(idx2upd);
    end
    if ~opthypr
        nl_sigma2=-l_sigma2;
    elseif jtupt
        nl_sigma2=sum(nl_sigma2);
    end
    
    % update kernels
    ker{3}=ker{3}.update(sigma2(3));
    mgC=mgC.update(mgC.stgp.update([],mgC.stgp.C_t.update(sigma2(2))),sigma2(1));
    
    % setting for eta and Lambda
    if intM
        M=[];
    end
    
    % update eta
    % eta_x
    logf{1}=@(q)logpost_eta([q,eta(2)],mgC,y,m(1),V(1),1,M);
    if ~opthypr
        [eta(1),l_eta(1)]=slice(eta(1),logf{1}(eta(1)),logf{1});
    elseif ~jtupt
        [eta(1),nl_eta(1)]=fminunc(@(q)-logf{1}(q),eta(1),opts_unc);
    end
    % eta_t
    logf{2}=@(q)logpost_eta([eta(1),q],mgC,y,m(2),V(2),2,M);
    if ~opthypr
        [eta(2),l_eta(2)]=slice(eta(2),logf{2}(eta(2)),logf{2});
    elseif ~jtupt
        [eta(2),nl_eta(2)]=fminunc(@(q)-logf{2}(q),eta(2),opts_unc);
    end
    % eta_u
    logf{3}=@(q)logpost_eta(q,mgC,y,m(3),V(3),3,M,ker{3});
    if ~opthypr
        [eta(3),l_eta(3)]=slice(eta(3),logf{3}(eta(3)),logf{3});
        nl_eta=-l_eta;
    elseif ~jtupt
        [eta(3),nl_eta(3)]=fminunc(@(q)-logf{3}(q),eta(3),opts_unc);
    end
    % joint optmize
    if opthypr && jtupt
        logF=@(q)logpost_eta(q,mgC,y,m,V,1:3,M,ker{3});
        [eta,nl_eta]=fminunc(@(q)-sum(logF(q)),eta,opts_unc);
    end
    
    % update kernels
    ker{3}=ker{3}.update([],exp(eta(3)));
    mgC=mgC.update(mgC.stgp.update(mgC.stgp.C_x.update([],exp(eta(1))),mgC.stgp.C_t.update([],exp(eta(2)))));
    
    % sample Lambda with ESS
    logLik_Lambda=@(q)loglik_Lambda(q,mgC,y,M);
    prirnd_Lambda=@()mgC.stgp.scale_Lambda(ker{3}.rnd([],L),'dn');
    [Lambda,l_Lambda] = ESS(Lambda,logLik_Lambda(Lambda),prirnd_Lambda,logLik_Lambda);
    % update marginal kernel
    mgC=mgC.update(mgC.stgp.update([],[],Lambda));
    
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
        samp_sigma2(NO_sav,:)=sigma2;
        samp_eta(NO_sav,:)=eta;
        samp_Lambda(NO_sav,:,:)=Lambda;
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
alg_name='MCMC';
if opthypr
    alg_name=['opt',alg_name];
    if jtupt
        alg_name=['jt',alg_name];
    end
end
keywd=[models{mdl_opt},'_I',num2str(I),'_J',num2str(J),'_K',num2str(K),'_L',num2str(L),'_d',num2str(d),'_',time_lbl];
f_name=['dynBrainIM_',typ,'_',grp,'_',alg_name];
if intM
    f_name=[f_name,'_intM_',keywd];
else
    f_name=[f_name,'_',keywd];
end
% folder = './result/';
folder = [store_prefix,'summary/'];
if exist(folder,'dir')~=7
    mkdir(folder);
end
save([folder,f_name,'.mat'],'seedNO','opthypr','jtupt','sampleM','intM','models','mdl_opt','grp_opt','stdtimes',...
                            'I','J','K','L','d','t','y','imsz','kappa','g','ker','mgC','a','b','m','V',...
                            'Nsamp','NBurnIn','thin','Niter','optini','samp_sigma2','samp_eta','samp_M','samp_Lambda','engy','time','-v7.3');
% summarize
fprintf('\nIt takes %.2f seconds to collect %e samples after thinning %d.\n', time,Nsamp,thin);
end