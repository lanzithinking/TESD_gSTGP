% This is to plot predictive mean function of time with fixed locations

clear;
addpath('../util/');
addpath('~/Documents/MATLAB/tight_subplot/');
% addpath('~/Documents/MATLAB/boundedline/');
addpath(genpath('~/Documents/MATLAB/boundedline-pkg/'));
% addpath('~/Documents/MATLAB/supertitle/');
% Random Numbers...
seedNO = 2018;
seed = RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% model options
models={'kron_prod','kron_sum'};
L_mdl=length(models);

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
ker{3}=ker{2}; % for hyper-GP

%% truth

% % true mean field
% M=sum(cos(pi.*x),2).*sin(pi.*t');

% ground truth functions
x_mf_t=@(x,t)sum(cos(pi.*x),2).*sin(pi.*t');

%% estimation

% obtain prediction of mean function from MCMC samples
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
keywd = ['_I',num2str(I),'_J',num2str(J_tr),'_K',num2str(K),'_L',num2str(L),'_d',num2str(d)];
f_name = ['predxmft',keywd];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
else
    M_estm=cell(1,L_mdl); [M_estm{:}]=deal(zeros(I,J_te)); M_estd = M_estm;
    M_predm=cell(1,L_mdl); [M_predm{:}]=deal(zeros(I,J_te)); M_prestd = M_predm;
    for l=1:L_mdl
        found=false;
        for k=1:nfiles
            if contains(files(k+2).name,[models{l},keywd])
                load(strcat(folder, files(k+2).name),'samp_*');
                fprintf('%s loaded.\n',files(k+2).name);
                found=true; break;
            end
        end
        if found
            M_estm{l}=shiftdim(mean(samp_M,1)); M_estd{l}=shiftdim(std(samp_M,0,1));
            N_samp=size(samp_sigma2,1); M_predvar=zeros(I,J_te);
            for n=1:N_samp
                for k=1:length(ker)
                    ker{k}.C=samp_sigma2(n,k)^(k~=1).*(exp(-.5.*ker{k}.dist.*exp(-ker{k}.s.*samp_eta(n,k)))+ker{k}.jit);
                end
                Lambda_n=squeeze(samp_Lambda(n,:,:)); % (J_tr,L)
                switch l
                    case 1
                        Lambda=zeros(J,L); Lambda(tr_j,:)=Lambda_n;
                        Lambda(te_j,:)=ker{3}.C(te_j,tr_j)/ker{3}.C(tr_j,tr_j)*Lambda_n;
                        stgp=STGP(ker{1}.C,ker{2}.C,Lambda,models{l});
                        C_M_n=stgp.get_jtker();
                        C_M_n=reshape(C_M_n,[I,J,I,J]);
                        C_E=reshape(C_M_n(:,te_j,:,te_j),I*J_te,I*J_te); C_ED=reshape(C_M_n(:,te_j,:,tr_j),I*J_te,I*J_tr);
                        stgp.C_t=stgp.C_t(tr_j,tr_j); stgp.J=J_tr; stgp=stgp.get_bkdgix('IJI'); stgp.Lambda=Lambda_n;
                    case 2
                        stgp=STGP(ker{1}.C,ker{2}.C(tr_j,tr_j),Lambda_n,models{l});
                        C_E=kron(ker{2}.C(te_j,te_j),speye(I));
                        C_ED=kron(ker{2}.C(te_j,tr_j),speye(I));
                end
                [mu,Sigma]=stgp.pred_mean(y(:,tr_j,:),samp_sigma2(n,1),C_E,C_ED);
                M_predm{l}=M_predm{l}+reshape(mu,I,J_te); M_predvar=M_predvar+reshape(mu.^2,I,J_te);
                M_prestd{l}=M_prestd{l}+reshape(diag(Sigma),I,J_te);
            end
            M_predm{l}=M_predm{l}./N_samp; M_predvar=M_predvar./N_samp - M_predm{l}.^2;
            M_prestd{l}=sqrt(M_prestd{l}./N_samp + M_predvar);
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

% plot predicted mean functions
for l=1:L_mdl
    subplot(ha(l));
    M_estm_=M_estm{l}(x_idx,:); M_estd_=M_estd{l}(x_idx,:);
    M_predm_=M_predm{l}(x_idx,:); M_prestd_=M_prestd{l}(x_idx,:);
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
    title(['Model ',repmat('I',1,l),repmat(' ',1,4),'K = ',num2str(K)],'fontsize',18);
end
% save plot
fig.PaperPositionMode = 'auto';
fld_sav = './summary/figures/';
if exist(fld_sav,'dir')~=7
    mkdir(fld_sav);
end
print(fig,[fld_sav,f_name],'-dpng','-r0');
