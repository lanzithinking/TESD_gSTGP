% This is to obtain prediction of mean, covariance functions

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

% TESD prediction options
predcovs=strcat('TESD to',{' future',' neighbor'});
L_pred=length(predcovs);

% model options
models={'sep','kron_prod','kron_sum'};
L_mdl=length(models);
upthypr=1;
intM=false;
alg_name='MCMC';
if upthypr>=2
    alg_name=['opt',alg_name];
    if upthypr==3
        alg_name=['jt',alg_name];
    end
end
stationary=false;
seeds=2019:2028;
L_seed=length(seeds);

%% data

% parameters setting
N=[200,100]; % discretization sizes for space and time domains
% N=[20,500]; % discretization sizes for space and time domains
trials=[100,1000]; % number of trials
% trials=[1]; % number of trials
L_trial=length(trials);
d=1; % space dimension
% load or simulate data
[x,t,~]=generate_data(N,trials(end),d,stationary,seedNO);
l_x=0.5; l_t=0.3; l_xt=sqrt(l_x*l_t);
sigma2_n=1e-2; % noise variance

% thin the mesh
thin=[50,1];
% thin=[5,1];
x=x(1:thin(1):end,:); t=t(1:thin(2):end);
I=size(x,1); J=size(t,1); L=I;
% TESD to future: record holdout time index
tr_j=1:length(t);
tr_j(end-floor(J*.2)+1:2:end)=[]; tr_j(end-floor(J*.05)+1:end)=[];
te_j=setdiff(1:length(t),tr_j);
J_tr=length(tr_j); J_te=length(te_j);
% TESD to neighbors
x_te=[0.1];
I_te=size(x_te,1);
Times=[J_tr,J];

% parameters of kernel
s=2; % smoothness
% kappa=1.2; % decaying rate for dynamic eigenvalues
% spatial kernel
jit_x=1e-6;
% temporal kernels
jit_t=1e-6;

% distance to neighbors
% dist_xte=pdist2(x_te,x_te,'minkowski',s).^s;
dist_xtex=pdist2(x_te,x,'minkowski',s).^s;

%% truth

% % true mean field
% M=sum(cos(pi.*x),2).*sin(pi.*t');
% % true marginal covariance field
% C_x=exp(-(x-x').^2./(2*l_x)); C_t=exp(-(t-t').^2./(2*l_t));
% xt=reshape(x,I,1,[]).*t'; xt=reshape(xt,I*J,[]); C_xt=exp(-pdist2(xt,xt,'minkowski',1)./(2*l_xt));
% C_y=kron(C_t,C_x).*C_xt+sigma2_n.*eye(I*J);

% ground truth functions
x_mf_t=@(x,t)sum(cos(pi.*x),2).*sin(pi.*t');
% C_x_t=@(t)C_x.*exp(-abs(x-x').*t./(2*l_xt))+sigma2_n.*eye(I);
% C_t_x=@(x)C_t.*exp(-abs(x.*(t-t'))./(2*l_xt))+sigma2_n.*eye(J);
x_covf_t=@(x,t)exp(-(diff(x)).^2./(2*l_x)).*exp(-abs(diff(x)).*t./(2*l_xt))+sigma2_n.*(abs(diff(x))<1e-10);

t_all=t; J_all=J;
%% estimation

% points for evaluating covariances
% locations=[-0.5,0,1];
locations=x'; L_locations=length(locations);
x_idx=zeros(1,L_locations);
for i=1:L_locations
    [~,x_idx(i)]=min(abs(x-locations(i)));
end
cov_sub=nchoosek(1:L_locations,2);
L_cov=size(cov_sub,1);
est_row_idx{1}=x_idx(cov_sub(:,1))'+I*(0:J_tr-1); est_col_idx{1}=x_idx(cov_sub(:,2))'+I*(0:J_tr-1);
est_row_idx{2}=x_idx(cov_sub(:,1))'+I*(0:J-1); est_col_idx{2}=x_idx(cov_sub(:,2))'+I*(0:J-1);
est_idx{1}=sub2ind([I*J_tr,I*J_tr],est_row_idx{1},est_col_idx{1});
est_idx{2}=sub2ind([I*J,I*J],est_row_idx{2},est_col_idx{2});

pred_row_idx{1}=x_idx(cov_sub(:,1))'+I*(0:J_te-1); pred_col_idx{1}=x_idx(cov_sub(:,2))'+I*(0:J_te-1);
[gd1,gd2]=meshgrid(x_idx,1:I_te);
pred_row_idx{2}=gd1(:)+I*(0:J-1); pred_col_idx{2}=gd2(:)+I_te*(0:J-1);
pred_idx{1}=sub2ind([I*J_te,I*J_te],pred_row_idx{1},pred_col_idx{1});
pred_idx{2}=sub2ind([I*J,I_te*J],pred_row_idx{2},pred_col_idx{2});
x_pts{1}=x(x_idx(cov_sub));
[gd1,gd2]=meshgrid(locations,x_te); x_pts{2}=[gd1(:),gd2(:)];


% allocation to save results
M_estm=cell(L_mdl,L_trial,L_seed); [M_estm{:}]=deal(zeros(I,J)); %[M_estm{:}]=deal(zeros(I,J_tr)); 
M_estd = M_estm;
M_predm=cell(L_mdl,L_trial,L_seed); [M_predm{:}]=deal(zeros(I,J_te)); M_prestd = M_predm;
C_estm=cell(L_mdl,L_pred,L_trial,L_seed); C_predm=cell(L_mdl,L_pred,L_trial,L_seed); 
[C_estm{:,1,:,:}]=deal(sparse(I*J_tr,I*J_tr)); [C_estm{:,2,:,:}]=deal(sparse(I*J,I*J));
[C_predm{:,1,:,:}]=deal(sparse(I*J_te,I*J_te)); [C_predm{:,2,:,:}]=deal(sparse(I*J,I_te*J));
C_estd = C_estm; C_prestd = C_predm;
MSE=zeros(L_mdl,3,L_trial,L_seed);
MSPE=zeros(L_mdl,4,L_trial,L_seed);

% obtain prediction of the mean and joint covariance kernel from MCMC samples
% load data
folder = './prediction/';
files = dir(folder);
nfiles = length(files) - 2;
for sd=1:L_seed
    sprintf('Processing data for seed number %d ...\n',seeds(sd));
    for mdl=0:L_mdl-1
        keywd = {[alg_name,'_',repmat('intM_',1,intM),models{mdl+1},'_I',num2str(I)],['_L',num2str(L),'_d',num2str(d),'_',repmat('non',1,~stationary),'stationary','_seedNO',num2str(seeds(sd))]};
        f_name = ['predxft_',keywd{:}];
        for tr=1:L_trial
            for l=1:L_pred
                found=false;
                for k=1:nfiles
                    if contains(files(k+2).name,join(keywd,['_J',num2str(Times(l)),'_K',num2str(trials(tr))]))
                        load(strcat(folder, files(k+2).name));
                        fprintf('%s loaded.\n',files(k+2).name);
                        found=true; break;
                    end
                end
                if found
                    if l==2
                        [i_dgix,j_dgix]=find(kron(speye(J),ones(I,I_te)));
                        i_dgix=permute(reshape(i_dgix,I,I_te,J),[1,3,2]);
                        j_dgix=permute(reshape(j_dgix,I,I_te,J),[1,3,2]);
                        M_estm{mdl+1,tr,sd}=shiftdim(mean(samp_M,1)); M_estd{mdl+1,tr,sd}=shiftdim(std(samp_M,0,1));
                    end
    %                 if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
    %                     stgp=mgC.stgp;
    %                 else
                    ker{1}=feval([stgp_ver,'.GP'],x,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_x,true);
                    ker{2}=feval([stgp_ver,'.GP'],t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                    stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},optini.Lambda,kappa,mdl); %[ker{1:2}]=deal([]);
                    mgC=feval([stgp_ver,'.mg'],stgp,K,optini.sigma2(1),L);
    %                 end
                    ker{3}=feval([stgp_ver,'.GP'],t_all,[],[],stgp.C_t.s,stgp.C_t.L,stgp.C_t.jit);
                    if l==1
                        ker2_full=ker{3}; stgp_full=stgp; stgp_full.N=I*J_all;
%                         M_estm{mdl+1,tr,sd}=shiftdim(mean(samp_M,1)); M_estd{mdl+1,tr,sd}=shiftdim(std(samp_M,0,1));
                        M_predvar=zeros(I,J_te);
                    end
                    N_samp=size(samp_sigma2,1);
                    for n=1:N_samp
                        for k=1:length(ker)
                            ker{k}=ker{k}.update(samp_sigma2(n,k)^(k~=1),exp(samp_eta(n,k)));
                        end
                        if l==1
                            ker2_full=ker2_full.update(samp_sigma2(n,2),exp(samp_eta(n,2)));
                        end
                        if mdl
                            Lambda_n=shiftdim(samp_Lambda(n,:,:)); % (J_,L)
                        else
                            Lambda_n=[];
                        end
                        stgp=stgp.update(ker{1},ker{2},Lambda_n);
    %                     stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),Lambda_n);
                        [~,C_z_tr]=stgp.tomat();
                        % prediction of covariance
                        switch l
                            case 1
                                if mdl
                                    C_tilt=ker{3}.tomat;
                                    Lambda=zeros(J_all,L); Lambda(tr_j,:)=Lambda_n;
                                    Lambda(te_j,:)=C_tilt(te_j,tr_j)/C_tilt(tr_j,tr_j)*Lambda_n;
                                else
                                    Lambda=[];
                                end
                                % prediction of mean
                                switch mdl
                                    case {0,1}
                                        stgp_full=stgp_full.update(stgp_full.C_x.update([],exp(samp_eta(n,1))),ker2_full,Lambda);
                                        [~,C_M_n]=stgp_full.tomat();
                                        C_M_n=reshape(full(C_M_n),[I,J_all,I,J_all]);
                                        C_E=reshape(C_M_n(:,te_j,:,te_j),I*J_te,I*J_te); C_ED=reshape(C_M_n(:,te_j,:,tr_j),I*J_te,I*J_tr);
                                    case 2
                                        C_t=ker2_full.tomat;
                                        C_E=kron(C_t(te_j,te_j),speye(I));
                                        C_ED=kron(C_t(te_j,tr_j),speye(I));
                                end
                                mgC=mgC.update(stgp_full.update(stgp_full.C_x.update([],exp(samp_eta(n,1))),mgC.stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),Lambda_n),samp_sigma2(n,1));
                                mgC.stgp.N=I*J;
                                [mu,Sigma]=mgC.predM(y,C_E,C_ED);
                                M_predm{mdl+1,tr,sd}=M_predm{mdl+1,tr,sd}+reshape(mu,I,J_te); M_predvar=M_predvar+reshape(mu.^2,I,J_te);
    %                             M_prestd{mdl+1,tr,sd}=M_prestd{mdl+1,tr,sd}+reshape(diag(Sigma),I,J_te);
                                % TESD to future
                                if mdl
                                    Lambda_te=Lambda(te_j,:);
                                else
                                    Lambda_te=[];
                                end
                                ker_te=feval([stgp_ver,'.GP'],t_all(te_j),ker{2}.sigma2,ker{2}.l,stgp.C_t.s,1,stgp.C_t.jit,false);
                                stgp_te=stgp.update([],ker_te,Lambda_te); stgp_te.N=stgp_te.I*stgp_te.J;
                                [~,C_z_te]=stgp_te.tomat();
                            case 2
                                % TESD to neighbors
                                C_xteX=exp(-.5.*dist_xtex.*exp(-ker{1}.s.*samp_eta(n,1))); % (I_te,I)
                                if mdl
                                    Phi_xteX=C_xteX*(stgp.C_x.eigf./stgp.C_x.eigv'); % (I_te,L)
                                    PhiLambda2=reshape(stgp.C_x.eigf,stgp.I,1,[]).*reshape(Lambda_n.^2,1,stgp.J,[]); % (I,J,L)
                                    PhiLambda2=reshape(PhiLambda2,stgp.I*stgp.J,[]);
                                    PhiLambda2=PhiLambda2*Phi_xteX'; %(IJ, I_te)
                                else
                                    PhiLambda2=repmat(stgp.C_x.eigf*(C_xteX*stgp.C_x.eigf)',stgp.J,1);
                                end
                                if mdl~=2
                                    PhiLambda2=PhiLambda2.*stgp.C_t.sigma2;
                                end
                                C_z_te=sparse(i_dgix(:),j_dgix(:),PhiLambda2(:));
                        end
                        C_estm{mdl+1,l,tr,sd}=C_estm{mdl+1,l,tr,sd}+C_z_tr; C_estd{mdl+1,l,tr,sd}=C_estd{mdl+1,l,tr,sd}+C_z_tr.^2;
                        C_predm{mdl+1,l,tr,sd}=C_predm{mdl+1,l,tr,sd}+C_z_te; C_prestd{mdl+1,l,tr,sd}=C_prestd{mdl+1,l,tr,sd}+C_z_te.^2;
                    end
                    if l==1
                        M_predm{mdl+1,tr,sd}=M_predm{mdl+1,tr,sd}./N_samp; M_predvar=M_predvar./N_samp - M_predm{mdl+1,tr,sd}.^2;
                        M_prestd{mdl+1,tr,sd}=sqrt(M_prestd{mdl+1,tr,sd}./N_samp + M_predvar);
                    end
                    C_estm{mdl+1,l,tr,sd}=C_estm{mdl+1,l,tr,sd}./N_samp; C_estd{mdl+1,l,tr,sd}=sqrt(C_estd{mdl+1,l,tr,sd}./N_samp - C_estm{mdl+1,l,tr,sd}.^2);
                    C_predm{mdl+1,l,tr,sd}=C_predm{mdl+1,l,tr,sd}./N_samp; C_prestd{mdl+1,l,tr,sd}=sqrt(C_prestd{mdl+1,l,tr,sd}./N_samp - C_predm{mdl+1,l,tr,sd}.^2);
                end
                t_used(l)=time;
            end
            % calculate mean squared errors
            MSE(mdl+1,1,tr,sd)= mean( (M_estm{mdl+1,tr,sd}(x_idx,:) - x_mf_t(locations',t_all)).^2, 'all');
%             MSE(mdl+1,2,tr,sd)= mean( (C_estm{mdl+1,1,tr,sd}(est_idx{1})' - x_covf_t(x_pts{1}',t_all(tr_j))).^2, 'all');
            MSE(mdl+1,2,tr,sd)= mean( (C_estm{mdl+1,2,tr,sd}(est_idx{2})' - x_covf_t(x_pts{1}',t_all)).^2, 'all');
            MSE(mdl+1,3,tr,sd)= t_used(end);
            % calculate mean squared prediction errors
            MSPE(mdl+1,1,tr,sd)= mean( (M_predm{mdl+1,tr,sd}(x_idx,:) - x_mf_t(locations',t_all(te_j))).^2, 'all');
            MSPE(mdl+1,2,tr,sd)= mean( (C_predm{mdl+1,1,tr,sd}(pred_idx{1})' - x_covf_t(x_pts{1}',t_all(te_j))).^2, 'all');
            MSPE(mdl+1,3,tr,sd)= mean( (C_predm{mdl+1,2,tr,sd}(pred_idx{2})' - x_covf_t(x_pts{2}',t_all)).^2, 'all');
            MSPE(mdl+1,4,tr,sd)= mean(t_used);
        end
    end
end
% save the estimation results
save([folder,'pred_',[alg_name,repmat('_intM_',1,intM),'_I',num2str(I),'_J',num2str(J),'_L',num2str(L),'_d',num2str(d),'_',repmat('non',1,~stationary),'stationary'],'.mat'],...
     'M_estm','M_estd','M_predm','M_prestd','C_estm','C_estd','C_predm','C_prestd','MSE','MSPE');

%% post-processing

% remove outlier
MSE(:,:,:,3)=[]; MSPE(:,:,:,3)=[];

% MSE
varnames={'mean\\$K=100$','covariance\\$K=100$','time\\$K=100$','mean\\$K=1000$','covariance\\$K=1000$','time\\$K=1000$'};
mse{1}=median(MSE,4); mse{1}=array2table(mse{1}(:,:),'RowNames',models,'VariableNames',varnames);
writetable(mse{1},[folder,'MSE-mean.csv']);
mse{2}=std(MSE,0,4); mse{2}=array2table(mse{2}(:,:),'RowNames',models,'VariableNames',varnames);
writetable(mse{2},[folder,'MSE-std.csv']);

% MSPE
varnames={'mean\\$K=100$','TESD to future\\$K=100$','TESD to neighbor\\$K=100$','time\\$K=100$','mean\\$K=1000$','TESD to future\\$K=1000$','TESD to neighbor\\$K=1000$','time\\$K=1000$'};
mspe{1}=median(MSPE,4); mspe{1}=array2table(mspe{1}(:,:),'RowNames',models,'VariableNames',varnames);
writetable(mspe{1},[folder,'MSPE-mean.csv']);
mspe{2}=std(MSPE,0,4); mspe{2}=array2table(mspe{2}(:,:),'RowNames',models,'VariableNames',varnames);
writetable(mspe{2},[folder,'MSPE-std.csv']);
