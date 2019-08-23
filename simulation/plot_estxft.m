% This is to plot estimated mean function of time with fixed locations and
% selected spatial covariances as functions of time

clear;
addpath('../util/');
addpath('~/Documents/MATLAB/tight_subplot/');
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
trials=[100,1000]; % number of trials
L_trial=length(trials);
d=1; % space dimension
% load or simulate data
[x,t,~]=generate_data(N,trials(end),d,seedNO);
l_x=0.5; l_t=0.3; l_xt=sqrt(l_x*l_t);
sigma2_n=1e-2; % noise variance

% thin the mesh
thin=[50,1];
x=x(1:thin(1):end,:); t=t(1:thin(2):end);
I=size(x,1); J=size(t,1);
L=I;

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

%% estimation

% estimate the mean and joint covariance kernel from MCMC samples
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
for l=1:L_mdl
keywd = {[models{l},'_I',num2str(I),'_J',num2str(J)],['_L',num2str(L),'_d',num2str(d)]};
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
            N_samp=size(samp_sigma2,1);
            M_estm{tr}=shiftdim(mean(samp_M,1)); M_estd{tr}=shiftdim(std(samp_M,0,1));
            for n=1:N_samp
                for k=1:2
                    ker{k}.C=samp_sigma2(n,k)^(k~=1).*(exp(-.5.*ker{k}.dist.*exp(-ker{k}.s.*samp_eta(n,k)))+ker{k}.jit);
                end
                C_z_n=STGP(ker{1}.C,ker{2}.C,squeeze(samp_Lambda(n,:,:)),models{l}).get_jtker();
                if l==1
                    C_z_n=C_z_n+samp_sigma2(n,1).*speye(I*J);
                end
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
lgd_loc={'southwest','northeast'};

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
    xlim([min(t),max(t)]); %ylim([-1.2,1.2]);
    set(gca,'box','on','fontsize',15);
    xlabel('t','fontsize',18); ylabel('m(x, t)','fontsize',18);
    set(gca,'colororderindex',1);
    plot(t,x_mf_t(locations',t)','linestyle','--','linewidth',2.5); % add truth
    lgd=legend(h1,strcat('x=',sprintfc('%g',locations)),'location','southwest');
    set(lgd,'orientation','horizontal','fontsize',18,'box','off');
    title(['Model ',repmat('I',1,l),repmat(' ',1,4),'K = ',num2str(trials(tr))],'fontsize',20);
    % plot covariance
    subplot(ha(2*tr));
    C_estm_=C_estm{tr}(sub2ind([I*J,I*J],row_idx,col_idx));
    C_estd_=full(C_estd{tr}(sub2ind([I*J,I*J],row_idx,col_idx)));
    plot(0,0); hold on;
    set(gca,'colororderindex',1);
    [cband_l,cband_p]=boundedline(t,C_estm_',reshape(1.96.*C_estd_',[size(C_estd_',1),1,size(C_estd_',2)]),'alpha'); hold on;
    h2=plot(t,C_estm_','linewidth',1.5); hold on;
    xlim([min(t),max(t)]);
    switch l
        case 1
            ylim([-3,2]);
        case 2
            ylim([0,1]);
    end
    set(gca,'box','on','fontsize',15);
    xlabel('t','fontsize',18); ylabel('C_t(x, x'')','fontsize',18);
    set(gca,'colororderindex',1);
    x_pts=x(x_idx(cov_sub));
    plot(t,x_covf_t(x_pts',t),'linestyle','--','linewidth',2.5); % add truth
    lgd=legend(h2,strcat('(x=',sprintfc('%g',x_pts(:,1)),', x''=',sprintfc('%g',x_pts(:,2)),')'),'location',lgd_loc{l});
    set(lgd,'fontsize',18,'box','off');
            
    title(['Model ',repmat('I',1,l),repmat(' ',1,4),'K = ',num2str(trials(tr))],'fontsize',20);
end
% save plot
fig.PaperPositionMode = 'auto';
fld_sav = './summary/figures/';
if exist(fld_sav,'dir')~=7
    mkdir(fld_sav);
end
print(fig,[fld_sav,f_name],'-dpng','-r0');

end