% This is to plot predictive covariance functions of time for two cases
% 1. Temporal evolution of spatial dependence (TESD) to future time;
% 2. TESD to neighbors.

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

% prediction options
predcovs=strcat('TESD to',{' future',' neighbor'});
L_pred=length(predcovs);

% model
models={'sep','kron_prod','kron_sum'};
mdl=models{3};
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

%% data

% parameters setting
N=[200,100]; % discretization sizes for space and time domains
K=100; % number of trials
d=1; % space dimension
% load or simulate data
[x,t,y]=generate_data(N,K,d,stationary,seedNO);
l_x=0.5; l_t=0.3; l_xt=sqrt(l_x*l_t);
sigma2_n=1e-2; % noise variance

% thin the mesh
thin=[50,1];
x=x(1:thin(1):end,:); t=t(1:thin(2):end); y=y(1:thin(1):end,1:thin(2):end,:);
I=size(x,1); J=size(t,1); L=I;
% record holdout time index
tr_j=1:length(t);
tr_j(end-floor(J*.2)+1:2:end)=[]; tr_j(end-floor(J*.05)+1:end)=[];
te_j=setdiff(1:length(t),tr_j);
J_tr=length(tr_j); J_te=length(te_j);

% neighbors
x_te=[0.1];
I_te=size(x_te,1);

% training time points
Times=[J_tr,J];

% parameters of kernel
s=2; % smoothness
kappa=1.2; % decaying rate for dynamic eigenvalues
% spatial kernel
jit_x=1e-6;
% temporal kernels
jit_t=1e-6;

% distance to neighbors
% dist_xte=pdist2(x_te,x_te,'minkowski',s).^s;
dist_xtex=pdist2(x_te,x,'minkowski',s).^s;

%% truth

% % true marginal covariance field
% C_x=exp(-(x-x').^2./(2*l_x)); C_t=exp(-(t-t').^2./(2*l_t));
% xt=reshape(x,I,1,[]).*t'; xt=reshape(xt,I*J,[]); C_xt=exp(-pdist2(xt,xt,'minkowski',1)./(2*l_xt));
% C_y=kron(C_t,C_x).*C_xt+sigma2_n.*eye(I*J);

% ground truth functions
% C_x_t=@(t)C_x.*exp(-abs(x-x').*t./(2*l_xt))+sigma2_n.*eye(I);
% C_t_x=@(x)C_t.*exp(-abs(x.*(t-t'))./(2*l_xt))+sigma2_n.*eye(J);
x_covf_t=@(x,t)exp(-(diff(x)).^2./(2*l_x)).*exp(-abs(diff(x)).*t./(2*l_xt))+sigma2_n.*(abs(diff(x))<1e-10);

t_all=t; J_all=J;
%% estimation

% obtain prediction of joint covariance kernel from MCMC samples
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
keywd = {[alg_name,'_',repmat('intM_',1,intM),mdl,'_I',num2str(I)],['_K',num2str(K),'_L',num2str(L),'_d',num2str(d)]};
f_name = ['predcovft_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
else
    C_estm=cell(1,L_pred); [C_estm{:}]=deal(sparse(I*J_tr,I*J_tr),sparse(I*J,I*J)); C_estd = C_estm;
    C_predm=cell(1,L_pred); [C_predm{:}]=deal(sparse(I*J_te,I*J_te),sparse(I*J,I_te*J)); C_prestd = C_predm;
    for l=1:L_pred
        found=false;
        for k=1:nfiles
            if contains(files(k+2).name,join(keywd,['_J',num2str(Times(l))]))
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
            end
            N_samp=size(samp_sigma2,1);
            if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
                stgp=mgC.stgp;
            else
                ker{1}=feval([stgp_ver,'.GP'],x,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_x,true);
                ker{2}=feval([stgp_ver,'.GP'],t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},optini.Lambda,kappa,mdl_opt); %[ker{1:2}]=deal([]);
                mgC=feval([stgp_ver,'.mg'],stgp,K,optini.sigma2(1),L);
            end
            ker{3}=feval([stgp_ver,'.GP'],t_all,[],[],stgp.C_t.s,stgp.C_t.L,stgp.C_t.jit);
            for n=1:N_samp
                for k=1:length(ker)
                    ker{k}=ker{k}.update(samp_sigma2(n,k)^(k~=1),exp(samp_eta(n,k)));
                end
                Lambda_n=shiftdim(samp_Lambda(n,:,:)); % (J_,L)
                stgp=stgp.update(ker{1},ker{2},Lambda_n);
%                 stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),Lambda_n);
                [~,C_z_tr]=stgp.tomat();
                switch l
                    case 1
                        C_tilt=ker{3}.tomat;
                        Lambda_te=C_tilt(te_j,tr_j)/C_tilt(tr_j,tr_j)*Lambda_n;
                        ker_te=feval([stgp_ver,'.GP'],t_all(te_j),ker{2}.sigma2,ker{2}.l,stgp.C_t.s,1,stgp.C_t.jit,false);
                        stgp_te=stgp.update([],ker_te,Lambda_te); stgp_te.N=stgp_te.I*stgp_te.J;
                        [~,C_z_te]=stgp_te.tomat();
                    case 2
                        C_xteX=exp(-.5.*dist_xtex.*exp(-ker{1}.s.*samp_eta(n,1)));
                        Phi_xteX=C_xteX*(stgp.C_x.eigf./stgp.C_x.eigv');
                        PhiLambda2=reshape(stgp.C_x.eigf,stgp.I,1,[]).*reshape(Lambda_n.^2,1,stgp.J,[]);
                        PhiLambda2=reshape(PhiLambda2,stgp.I*stgp.J,[]);
                        PhiLambda2=PhiLambda2*Phi_xteX';
                        C_z_te=sparse(i_dgix(:),j_dgix(:),PhiLambda2(:));
                end
                C_estm{l}=C_estm{l}+C_z_tr; C_estd{l}=C_estd{l}+C_z_tr.^2;
                C_predm{l}=C_predm{l}+C_z_te; C_prestd{l}=C_prestd{l}+C_z_te.^2;
            end
            C_estm{l}=C_estm{l}./N_samp; C_estd{l}=sqrt(C_estd{l}./N_samp - C_estm{l}.^2);
            C_predm{l}=C_predm{l}./N_samp; C_prestd{l}=sqrt(C_prestd{l}./N_samp - C_predm{l}.^2);
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'C_estm','C_estd','C_predm','C_prestd');
end


%% plot

% plot conditional covariances
fig=figure(1); clf(fig);
set(fig,'pos',[0 800 800 400]);
ha=tight_subplot(1,2,[.1,.09],[.12,.1],[.07,.04]);
lty={'-','--','-.',':'};
lgd_loc={'southwest','northeast'};

% plot options
% times=[0,0.5,1];
locations=[-0.5,0,1]; L_locations=length(locations);
x_idx=zeros(1,L_locations);
for i=1:L_locations
    [~,x_idx(i)]=min(abs(x-locations(i)));
end
cov_sub=nchoosek(1:L_locations,2);
L_cov=size(cov_sub,1);
est_row_idx{1}=x_idx(cov_sub(:,1))'+I*(0:J_tr-1); est_col_idx{1}=x_idx(cov_sub(:,2))'+I*(0:J_tr-1);
pred_row_idx{1}=x_idx(cov_sub(:,1))'+I*(0:J_te-1); pred_col_idx{1}=x_idx(cov_sub(:,2))'+I*(0:J_te-1);
est_row_idx{2}=x_idx(cov_sub(:,1))'+I*(0:J-1); est_col_idx{2}=x_idx(cov_sub(:,2))'+I*(0:J-1);
[gd1,gd2]=meshgrid(x_idx,1:I_te);
pred_row_idx{2}=gd1(:)+I*(0:J-1); pred_col_idx{2}=gd2(:)+I_te*(0:J-1);

% plot predicted covariance functions
for l=1:L_pred
    subplot(ha(l));
    est_idx=sub2ind([I*Times(l),I*Times(l)],est_row_idx{l},est_col_idx{l});
    C_estm_=C_estm{l}(est_idx); C_estd_=full(C_estd{l}(est_idx));
    switch l
        case 1
            pred_idx=sub2ind([I*J_te,I*J_te],pred_row_idx{l},pred_col_idx{l});
        case 2
            pred_idx=sub2ind([I*J,I_te*J],pred_row_idx{l},pred_col_idx{l});
            tr_j=1:J; te_j=1:J;
    end
    C_predm_=C_predm{l}(pred_idx); C_prestd_=full(C_prestd{l}(pred_idx));
    % plot estimated covariance
    plot(0,0); hold on;
    set(gca,'colororderindex',1);
    if l==1
        [cband_l,cband_p]=boundedline(t(tr_j),C_estm_',reshape(1.96.*C_estd_',[size(C_estd_',1),1,size(C_estd_',2)]),'alpha'); hold on;
    %     h1=plot(t(tr_j),C_estm_','linewidth',1.5); hold on;
        plot(ones(2,1)*t(tr_j)',[0;.1]*ones(1,J_tr)-.5,'color',zeros(1,3)+.5,'linewidth',1); hold on;
    end
    % plot predicted covariance
    plot(0,0); hold on;
    set(gca,'colororderindex',1);
    [cband_l,cband_p]=boundedline(t(te_j),C_predm_',reshape(1.96.*C_prestd_',[size(C_prestd_',1),1,size(C_prestd_',2)]),'alpha','transparency',0.5); hold on;
    h2=plot(t(te_j),C_predm_','linestyle',lty{3},'linewidth',4); hold on;
    if l==1
        plot(ones(2,1)*t(te_j)',[0;.15]*ones(1,J_te)-.5,'color','k','linewidth',1.5,'linestyle',lty{3}); hold on;
    end
    % add truth
    set(gca,'colororderindex',1);
    switch l
        case 1
            x_pts=x(x_idx(cov_sub));
        case 2
            [gd1,gd2]=meshgrid(locations,x_te);
            x_pts=[gd1(:),gd2(:)];
    end
    plot(t,x_covf_t(x_pts',t),'linestyle',lty{2},'linewidth',2);
    xlim([min(t),max(t)]);
    set(gca,'box','on','fontsize',15);
    xlabel('t','fontsize',18);
    switch l
        case 1
            ylim([-.5,1]); ylabel('C_t(x, x'')','fontsize',18);
            lgd=legend(h2,strcat('(x=',sprintfc('%g',x_pts(:,1)),', x''=',sprintfc('%g',x_pts(:,2)),')'),'location',lgd_loc{2});
        case 2
            ylim([0,1.5]); ylabel('C_t(x, x_*)','fontsize',18);
            lgd=legend(h2,strcat('(x=',sprintfc('%g',x_pts(:,1)),', x_*=',sprintfc('%g',x_pts(:,2)),')'),'location',lgd_loc{l});
    end
    ylabh=get(gca,'YLabel'); set(ylabh,'Position',get(ylabh,'Position') + [0.02 0 0]);
    set(lgd,'fontsize',18,'box','off');
    title([predcovs{l},repmat(' ',1,4),'K = ',num2str(K)],'fontsize',18);
end
% save plot
fig.PaperPositionMode = 'auto';
fld_sav = './summary/figures/';
if exist(fld_sav,'dir')~=7
    mkdir(fld_sav);
end
print(fig,[fld_sav,f_name],'-dpng','-r0');
