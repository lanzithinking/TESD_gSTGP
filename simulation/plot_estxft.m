% This is to plot estimated mean function of time with fixed locations and
% selected spatial covariances as functions of time

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
models={'sep','kron_prod','kron_sum','nonsepstat'};
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
I=size(x,1); J=size(t,1);
L=I;

% parameters of kernel
s=2; % smoothness
% kappa=1.2; % decaying rate for dynamic eigenvalues
% spatial kernel
jit_x=1e-6;
% temporal kernels
jit_t=1e-6;

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
if stationary
    x_covf_t=@(x,t)exp(-(diff(x)).^2./(2*l_x)).*exp(-abs(diff(x)).*ones(length(t),1)./(2*l_xt))+sigma2_n.*(abs(diff(x))<1e-10);
else
    x_covf_t=@(x,t)exp(-(diff(x)).^2./(2*l_x)).*exp(-abs(diff(x)).*t./(2*l_xt))+sigma2_n.*(abs(diff(x))<1e-10);
end

%% estimation

% estimate the mean and joint covariance kernel from MCMC samples
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
for mdl=0:L_mdl-1
keywd = {[alg_name,'_',repmat('intM_',1,intM),models{mdl+1},'_I',num2str(I),'_J',num2str(J)],['_L',num2str(L),'_d',num2str(d),'_',repmat('non',1,~stationary),'stationary']};
f_name = ['estxft_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
else
    M_estm=cell(1,L_trial); [M_estm{:}]=deal(zeros(I,J)); M_estd = M_estm;
    C_estm=cell(1,L_trial); [C_estm{:}]=deal(sparse(I*J,I*J)); C_estd = C_estm;
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
%             if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
%                 stgp=mgC.stgp;
%             else
                ker{1}=feval([stgp_ver,'.GP'],x,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_x,true);
                ker{2}=feval([stgp_ver,'.GP'],t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                if mdl==3
                    stgp=nonsepstat(x,t,optini.sigma2(2),exp(optini.eta(1)),1,L,jit_x,mdl-1,true);
%                     if K>1
%                         var_hat=median(var(y,0,3),'all');
%                     else
%                         var_hat=var(y,0,'all');
%                     end
%                     stgp=nonsepstat(x,t,var_hat,exp(optini.eta(1)),1,L,jit_x,mdl-1,true);
                else
                    stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},optini.Lambda,kappa,mdl);
                end
                [ker{1:2}]=deal([]);
%             end
            N_samp=size(samp_sigma2,1);
            if ~isempty(samp_M)
                M_estm{tr}=shiftdim(mean(samp_M,1)); M_estd{tr}=shiftdim(std(samp_M,0,1));
            else
                M_estm{tr}=optini.M; M_estd{tr}=std(y,0,3)./sqrt(K);
            end
            for n=1:N_samp
                stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),shiftdim(samp_Lambda(n,:,:)));
%                 if mdl==3
%                     stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))));
%                 else
%                     stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),shiftdim(samp_Lambda(n,:,:)));
%                 end
                [~,C_z_n]=stgp.tomat();
                C_estm{tr}=C_estm{tr}+C_z_n; C_estd{tr}=C_estd{tr}+C_z_n.^2;
            end
            C_estm{tr}=C_estm{tr}./N_samp; C_estd{tr}=sqrt(C_estd{tr}./N_samp - C_estm{tr}.^2);
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'M_estm','M_estd','C_estm','C_estd');
end


%% plot

% plot conditional covariances
fig=figure(1); clf(fig);
set(fig,'pos',[0 800 800 600]);
ha=tight_subplot(L_trial,2,[.125,.09],[.08,.06],[.07,.04]);
lty={'-','--','-.',':'};
% lgd_loc={'southwest','southwest','northeast','northeast'};
lgd_loc={'northeast','northeast','northeast','northeast'};

% plot options
% times=[0,0.5,1];
locations=[-0.5,0,1]; L_locations=length(locations);
x_idx=zeros(1,L_locations);
for i=1:L_locations
    [~,x_idx(i)]=min(abs(x-locations(i))); % to-do: prediction
end
cov_sub=nchoosek(1:L_locations,2);
L_cov=size(cov_sub,1);
row_idx=x_idx(cov_sub(:,1))'+I*(0:J-1); col_idx=x_idx(cov_sub(:,2))'+I*(0:J-1);

% plot estimated mean functions and spatial covariance functions of time t
for tr=1:L_trial
    % plot mean
    subplot(ha((tr-1)*2+1));
    M_estm_=M_estm{tr}(x_idx,:); M_estd_=M_estd{tr}(x_idx,:);
    plot(0,0); hold on;
    set(gca,'colororderindex',1);
    [cband_l,cband_p]=boundedline(t,M_estm_',reshape(1.96.*M_estd_',[size(M_estd_',1),1,size(M_estd_',2)]),'alpha'); hold on;
    h1=plot(t,M_estm_','linewidth',1.5); hold on;
    xlim([min(t),max(t)]); ylim([-1.5,1.1]);
    set(gca,'box','on','fontsize',15);
    xlabel('t','fontsize',18); ylabel('m(x, t)','fontsize',18);
    set(gca,'colororderindex',1);
    plot(t,x_mf_t(locations',t)','linestyle','--','linewidth',2.5); % add truth
    lgd=legend(h1,strcat('x=',sprintfc('%g',locations)),'location','southwest');
    set(lgd,'orientation','horizontal','fontsize',18,'box','off');
    title(['Model ',join([repmat('0',1,mdl==0),repmat('I',1,mdl)]),repmat(' ',1,4),'K = ',num2str(trials(tr))],'fontsize',20);
    % plot covariance
    subplot(ha(2*tr));
    C_estm_=C_estm{tr}(sub2ind([I*J,I*J],row_idx,col_idx));
    C_estd_=full(C_estd{tr}(sub2ind([I*J,I*J],row_idx,col_idx)));
    plot(0,0); hold on;
    set(gca,'colororderindex',1);
    [cband_l,cband_p]=boundedline(t,C_estm_',reshape(1.96.*C_estd_',[size(C_estd_',1),1,size(C_estd_',2)]),'alpha'); hold on;
    h2=plot(t,C_estm_','linewidth',1.5); hold on;
    xlim([min(t),max(t)]);
    switch mdl
        case {0,2,3}
            ylim([-.1,1]);
        case 1
%             ylim([-3,2]);
            ylim([-1,1.5]);
    end
    set(gca,'box','on','fontsize',15);
    xlabel('t','fontsize',18); ylabel('C_t(x, x'')','fontsize',18);
    set(gca,'colororderindex',1);
    x_pts=x(x_idx(cov_sub));
    plot(t,x_covf_t(x_pts',t),'linestyle','--','linewidth',2.5); % add truth
    lgd=legend(h2,strcat('(x=',sprintfc('%g',x_pts(:,1)),', x''=',sprintfc('%g',x_pts(:,2)),')'),'location',lgd_loc{mdl+1});
    set(lgd,'fontsize',18,'box','off');
    title(['Model ',join([repmat('0',1,mdl==0),repmat('I',1,mdl)]),repmat(' ',1,4),'K = ',num2str(trials(tr))],'fontsize',20);
end
% save plot
fig.PaperPositionMode = 'auto';
fld_sav = './summary/figures/';
if exist(fld_sav,'dir')~=7
    mkdir(fld_sav);
end
print(fig,[fld_sav,f_name],'-dpng','-r0');

end