% This is to plot predictive mean function of time with fixed locations

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
models={'sep','kron_prod','kron_sum'};
L_mdl=length(models);
opthypr=false;
jtupt=false;
intM=false;
alg_name='MCMC';
if opthypr
    alg_name=['opt',alg_name];
    if jtupt
        alg_name=['jt',alg_name];
    end
end

%% data

% parameters setting
N=[200,100]; % discretization sizes for space and time domains
K=1000; % number of trials
d=1; % space dimension
% load or simulate data
[x,t,y]=generate_data(N,K,d,seedNO);
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

% parameters of kernel
s=2; % smoothness
kappa=1.2; % decaying rate for dynamic eigenvalues
% spatial kernel
jit_x=1e-6;
% temporal kernels
jit_t=1e-6;

%% truth

% % true mean field
% M=sum(cos(pi.*x),2).*sin(pi.*t');

% ground truth functions
x_mf_t=@(x,t)sum(cos(pi.*x),2).*sin(pi.*t');

t_all=t; J_all=J;
%% estimation

% obtain prediction of mean function from MCMC samples
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
keywd = {[alg_name,'_',repmat('intM_',1,intM)],['I',num2str(I),'_J',num2str(J_tr),'_K',num2str(K),'_L',num2str(L),'_d',num2str(d)]};
f_name = ['predxmft_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
else
    M_estm=cell(1,L_mdl); [M_estm{:}]=deal(zeros(I,J_te)); M_estd = M_estm;
    M_predm=cell(1,L_mdl); [M_predm{:}]=deal(zeros(I,J_te)); M_prestd = M_predm;
    for mdl_opt=1:L_mdl-1
        found=false;
        for k=1:nfiles
            if contains(files(k+2).name,join(keywd,[models{mdl_opt+1},'_']))
                load(strcat(folder, files(k+2).name));
                fprintf('%s loaded.\n',files(k+2).name);
                found=true; break;
            end
        end
        if found
            M_estm{mdl_opt}=shiftdim(mean(samp_M,1)); M_estd{mdl_opt}=shiftdim(std(samp_M,0,1));
            N_samp=size(samp_sigma2,1); M_predvar=zeros(I,J_te);
            if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
                stgp=mgC.stgp;
            else
                ker{1}=feval([stgp_ver,'.GP'],x,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_x,true);
                ker{2}=feval([stgp_ver,'.GP'],t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},optini.Lambda,kappa,mdl_opt); [ker{1:2}]=deal([]);
                mgC=feval([stgp_ver,'.mg'],stgp,K,optini.sigma2(1),L);
            end
            ker{2}=feval([stgp_ver,'.GP'],t_all,[],[],stgp.C_t.s,stgp.C_t.L,stgp.C_t.jit);
            ker{3}=ker{2};
            for n=1:N_samp
                for k=2:length(ker)
                    ker{k}=ker{k}.update(samp_sigma2(n,k),exp(samp_eta(n,k)));
                end
                Lambda_n=shiftdim(samp_Lambda(n,:,:)); % (J_tr,L)
                switch mdl_opt
                    case 1
                        C_tilt=ker{3}.tomat;
                        Lambda=zeros(J_all,L); Lambda(tr_j,:)=Lambda_n;
                        Lambda(te_j,:)=C_tilt(te_j,tr_j)/C_tilt(tr_j,tr_j)*Lambda_n;
                        stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),ker{2},Lambda); stgp.N=I*J_all;
                        [~,C_M_n]=stgp.tomat();
                        C_M_n=reshape(full(C_M_n),[I,J_all,I,J_all]);
                        C_E=reshape(C_M_n(:,te_j,:,te_j),I*J_te,I*J_te); C_ED=reshape(C_M_n(:,te_j,:,tr_j),I*J_te,I*J_tr);
                    case 2
                        C_t=ker{2}.tomat;
                        C_E=kron(C_t(te_j,te_j),speye(I));
                        C_ED=kron(C_t(te_j,tr_j),speye(I));
                end
                mgC=mgC.update(stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),mgC.stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),Lambda_n),samp_sigma2(n,1));
                mgC.stgp.N=I*J;
                [mu,Sigma]=mgC.predM(y,C_E,C_ED);
                M_predm{mdl_opt}=M_predm{mdl_opt}+reshape(mu,I,J_te); M_predvar=M_predvar+reshape(mu.^2,I,J_te);
%                 M_prestd{mdl_opt}=M_prestd{mdl_opt}+reshape(diag(Sigma),I,J_te);
            end
            M_predm{mdl_opt}=M_predm{mdl_opt}./N_samp; M_predvar=M_predvar./N_samp - M_predm{mdl_opt}.^2;
            M_prestd{mdl_opt}=sqrt(M_prestd{mdl_opt}./N_samp + M_predvar);
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'M_estm','M_estd','M_predm','M_prestd');
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

t=t_all;
% plot predicted mean functions
for mdl_opt=1:L_mdl-1
    subplot(ha(mdl_opt));
    M_estm_=M_estm{mdl_opt}(x_idx,:); M_estd_=M_estd{mdl_opt}(x_idx,:);
    M_predm_=M_predm{mdl_opt}(x_idx,:); M_prestd_=M_prestd{mdl_opt}(x_idx,:);
    % plot estimated mean
    plot(0,0); hold on;
    set(gca,'colororderindex',1);
    [cband_l,cband_p]=boundedline(t(tr_j),M_estm_',reshape(1.96.*M_estd_',[size(M_estd_',1),1,size(M_estd_',2)]),'alpha'); hold on;
%     h1=plot(t(tr_j),M_estm_','linewidth',1.5); hold on;
    plot(ones(2,1)*t(tr_j)',[0;.1]*ones(1,J_tr)-1.5,'color',zeros(1,3)+.5,'linewidth',1); hold on; % add rug
    % plot predicted mean
    plot(0,0); hold on;
    set(gca,'colororderindex',1);
    [cband_l,cband_p]=boundedline(t(te_j),M_predm_',reshape(1.96.*M_prestd_',[size(M_prestd_',1),1,size(M_prestd_',2)]),'alpha','transparency',0.5); hold on;
    h2=plot(t(te_j),M_predm_','linestyle',lty{3},'linewidth',4); hold on;
    plot(ones(2,1)*t(te_j)',[0;.15]*ones(1,J_te)-1.5,'color','k','linewidth',1.5,'linestyle',lty{3}); hold on; % add rug
    % add truth
    set(gca,'colororderindex',1);
    plot(t,x_mf_t(locations',t)','linestyle',lty{2},'linewidth',2);
    xlim([min(t),max(t)]); ylim([-1.5,1]);
    set(gca,'box','on','fontsize',15);
    xlabel('t','fontsize',18); ylabel('m(x, t_*)','fontsize',18);
    ylabh=get(gca,'YLabel'); set(ylabh,'Position',get(ylabh,'Position') + [0.02 0 0]);
    lgd=legend(h2,strcat('x=',sprintfc('%g',locations)),'location','southwest');
    set(lgd,'orientation','horizontal','fontsize',18,'box','off');
    title(['Model ',repmat('I',1,mdl_opt),repmat(' ',1,4),'K = ',num2str(K)],'fontsize',18);
end
% save plot
fig.PaperPositionMode = 'auto';
fld_sav = './summary/figures/';
if exist(fld_sav,'dir')~=7
    mkdir(fld_sav);
end
print(fig,[fld_sav,f_name],'-dpng','-r0');
