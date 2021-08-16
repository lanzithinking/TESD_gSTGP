% This is to plot predictive mean function of time with fixed locations and
% selected spatial covariances as functions of time for two cases
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

% obtain prediction of the mean and joint covariance kernel from MCMC samples
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
for mdl_opt=0:L_mdl-1
keywd = {[alg_name,'_',repmat('intM_',1,intM),models{mdl_opt+1},'_I',num2str(I)],['_L',num2str(L),'_d',num2str(d)]};
f_name = ['predxft_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
else
    M_estm=cell(1,L_trial); [M_estm{:}]=deal(zeros(I,J_tr)); M_estd = M_estm;
    M_predm=cell(1,L_trial); [M_predm{:}]=deal(zeros(I,J_te)); M_prestd = M_predm;
    C_estm=cell(L_pred,L_trial); C_predm=cell(L_pred,L_trial); 
    for tr=1:L_trial
        [C_estm{:,tr}]=deal(sparse(I*J_tr,I*J_tr),sparse(I*J,I*J));
        [C_predm{:,tr}]=deal(sparse(I*J_te,I*J_te),sparse(I*J,I_te*J));
    end
    C_estd = C_estm; C_prestd = C_predm;
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
                end
%                 if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
%                     stgp=mgC.stgp;
%                 else
                ker{1}=feval([stgp_ver,'.GP'],x,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_x,true);
                ker{2}=feval([stgp_ver,'.GP'],t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},optini.Lambda,kappa,mdl_opt); %[ker{1:2}]=deal([]);
                mgC=feval([stgp_ver,'.mg'],stgp,K,optini.sigma2(1),L);
%                 end
                ker{3}=feval([stgp_ver,'.GP'],t_all,[],[],stgp.C_t.s,stgp.C_t.L,stgp.C_t.jit);
                if l==1
                    ker2_full=ker{3}; stgp_full=stgp; stgp_full.N=I*J_all;
                    M_estm{tr}=shiftdim(mean(samp_M,1)); M_estd{tr}=shiftdim(std(samp_M,0,1));
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
                    if mdl_opt
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
                            if mdl_opt
                                C_tilt=ker{3}.tomat;
                                Lambda=zeros(J_all,L); Lambda(tr_j,:)=Lambda_n;
                                Lambda(te_j,:)=C_tilt(te_j,tr_j)/C_tilt(tr_j,tr_j)*Lambda_n;
                            else
                                Lambda=[];
                            end
                            % prediction of mean
                            switch mdl_opt
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
                            M_predm{tr}=M_predm{tr}+reshape(mu,I,J_te); M_predvar=M_predvar+reshape(mu.^2,I,J_te);
%                             M_prestd{tr}=M_prestd{tr}+reshape(diag(Sigma),I,J_te);
                            % TESD to future
                            if mdl_opt
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
                            if mdl_opt
                                Phi_xteX=C_xteX*(stgp.C_x.eigf./stgp.C_x.eigv'); % (I_te,L)
                                PhiLambda2=reshape(stgp.C_x.eigf,stgp.I,1,[]).*reshape(Lambda_n.^2,1,stgp.J,[]); % (I,J,L)
                                PhiLambda2=reshape(PhiLambda2,stgp.I*stgp.J,[]);
                                PhiLambda2=PhiLambda2*Phi_xteX'; %(IJ, I_te)
                            else
                                PhiLambda2=repmat(stgp.C_x.eigf*(C_xteX*stgp.C_x.eigf)',stgp.J,1);
                            end
                            if mdl_opt~=2
                                PhiLambda2=PhiLambda2.*stgp.C_t.sigma2;
                            end
                            C_z_te=sparse(i_dgix(:),j_dgix(:),PhiLambda2(:));
                    end
                    C_estm{l,tr}=C_estm{l,tr}+C_z_tr; C_estd{l,tr}=C_estd{l,tr}+C_z_tr.^2;
                    C_predm{l,tr}=C_predm{l,tr}+C_z_te; C_prestd{l,tr}=C_prestd{l,tr}+C_z_te.^2;
                end
                if l==1
                    M_predm{tr}=M_predm{tr}./N_samp; M_predvar=M_predvar./N_samp - M_predm{tr}.^2;
                    M_prestd{tr}=sqrt(M_prestd{tr}./N_samp + M_predvar);
                end
                C_estm{l,tr}=C_estm{l,tr}./N_samp; C_estd{l,tr}=sqrt(C_estd{l,tr}./N_samp - C_estm{l,tr}.^2);
                C_predm{l,tr}=C_predm{l,tr}./N_samp; C_prestd{l,tr}=sqrt(C_prestd{l,tr}./N_samp - C_predm{l,tr}.^2);
            end
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'M_estm','M_estd','M_predm','M_prestd','C_estm','C_estd','C_predm','C_prestd');
end


%% plot

% setting
fig=figure(1); clf(fig);
set(fig,'pos',[0 800 1200 600]);
ha=tight_subplot(L_trial,3,[.125,.06],[.08,.06],[.06,.04]);
lty={'-','--','-.',':'};
% lgd_loc={'southwest','southwest','northeast'};
lgd_loc={'northeast','northeast','northeast'};

% plot options
% times=[0,0.5,1];
locations=[-0.5,0,1]; L_locations=length(locations);
x_idx=zeros(1,L_locations);
for i=1:L_locations
    [~,x_idx(i)]=min(abs(x-locations(i))); % to-do: prediction
end
cov_sub=nchoosek(1:L_locations,2);
L_cov=size(cov_sub,1);
est_row_idx{1}=x_idx(cov_sub(:,1))'+I*(0:J_tr-1); est_col_idx{1}=x_idx(cov_sub(:,2))'+I*(0:J_tr-1);
pred_row_idx{1}=x_idx(cov_sub(:,1))'+I*(0:J_te-1); pred_col_idx{1}=x_idx(cov_sub(:,2))'+I*(0:J_te-1);
est_row_idx{2}=x_idx(cov_sub(:,1))'+I*(0:J-1); est_col_idx{2}=x_idx(cov_sub(:,2))'+I*(0:J-1);
[gd1,gd2]=meshgrid(x_idx,1:I_te);
pred_row_idx{2}=gd1(:)+I*(0:J-1); pred_col_idx{2}=gd2(:)+I_te*(0:J-1);

t=t_all;
% plot predicted mean functions and spatial covariance functions of time t
for tr=1:L_trial
    % plot mean
    M_estm_=M_estm{tr}(x_idx,:); M_estd_=M_estd{tr}(x_idx,:);
    M_predm_=M_predm{tr}(x_idx,:); M_prestd_=M_prestd{tr}(x_idx,:);
    subplot(ha((tr-1)*3+1));
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
    xlim([min(t),max(t)]); ylim([-1.5,1.1]);
    set(gca,'box','on','fontsize',15);
    xlabel('t','fontsize',18); ylabel('m(x, t_*)','fontsize',18);
    ylabh=get(gca,'YLabel'); set(ylabh,'Position',get(ylabh,'Position') + [0.02 0 0]);
    lgd=legend(h2,strcat('x=',sprintfc('%g',locations)),'location','southwest');
    set(lgd,'orientation','horizontal','fontsize',18,'box','off');
    title(['Model ',join([repmat('0',1,mdl_opt==0),repmat('I',1,mdl_opt)]),repmat(' ',1,4),'K = ',num2str(trials(tr))],'fontsize',20);
    
    % plot covariance
    tr__j=tr_j; te__j=te_j;
    for l=1:L_pred
        est_idx=sub2ind([I*Times(l),I*Times(l)],est_row_idx{l},est_col_idx{l});
        C_estm_=C_estm{l,tr}(est_idx); C_estd_=full(C_estd{l,tr}(est_idx));
        switch l
            case 1
                pred_idx=sub2ind([I*J_te,I*J_te],pred_row_idx{l},pred_col_idx{l});
            case 2
                pred_idx=sub2ind([I*J,I_te*J],pred_row_idx{l},pred_col_idx{l});
                tr__j=1:J; te__j=1:J;
        end
        C_predm_=C_predm{l,tr}(pred_idx); C_prestd_=full(C_prestd{l,tr}(pred_idx));
        subplot(ha((tr-1)*3+1+l));
        % plot estimated covariance
        plot(0,0); hold on;
        set(gca,'colororderindex',1);
        if l==1
            [cband_l,cband_p]=boundedline(t(tr__j),C_estm_',reshape(1.96.*C_estd_',[size(C_estd_',1),1,size(C_estd_',2)]),'alpha'); hold on;
        %     h1=plot(t(tr__j),C_estm_','linewidth',1.5); hold on;
            plot(ones(2,1)*t(tr__j)',[0;.1]*ones(1,J_tr)-1./mdl_opt,'color',zeros(1,3)+.5,'linewidth',1); hold on;
        end
        % plot predicted covariance
        plot(0,0); hold on;
        set(gca,'colororderindex',1);
        [cband_l,cband_p]=boundedline(t(te__j),C_predm_',reshape(1.96.*C_prestd_',[size(C_prestd_',1),1,size(C_prestd_',2)]),'alpha','transparency',0.4-.1*(l==2)); hold on;
        h2=plot(t(te__j),C_predm_','linestyle',lty{3},'linewidth',4); hold on;
        if l==1
            plot(ones(2,1)*t(te__j)',[0;.15]*ones(1,J_te)-1./mdl_opt,'color','k','linewidth',1.5,'linestyle',lty{3}); hold on;
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
                switch mdl_opt
                    case {0,2}
                        ylim([-.1,1]);
                    case 1
                        ylim([-1,1.5]);
                end
                ylabel('C_t(x, x'')','fontsize',18);
                lgd=legend(h2,strcat('(x=',sprintfc('%g',x_pts(:,1)),', x''=',sprintfc('%g',x_pts(:,2)),')'),'location',lgd_loc{2});
            case 2
                switch mdl_opt
                    case 0
                        ylim([-.1,1.5]);
                    case 1
                        ylim([-.5,2.5]);
                    case 2
                        ylim([0,1.7]);
                end 
                ylabel('C_t(x, x_*)','fontsize',18);
                lgd=legend(h2,strcat('(x=',sprintfc('%g',x_pts(:,1)),', x_*=',sprintfc('%g',x_pts(:,2)),')'),'location',lgd_loc{l});
                newht=lgd.Position(4)*.8;
                lgd.Position(2)=lgd.Position(2)+lgd.Position(4)*.1;
                lgd.Position(4)=newht;
        end
        ylabh=get(gca,'YLabel'); set(ylabh,'Position',get(ylabh,'Position') + [0.02 0 0]);
        set(lgd,'fontsize',16,'box','off');
%         title([predcovs{l},'Model ',join([repmat('0',1,mdl_opt==0),repmat('I',1,mdl_opt)]),repmat(' ',1,4),'K = ',num2str(trials(tr))],'fontsize',18);
        title([predcovs{l},repmat(' ',1,4),'K = ',num2str(trials(tr))],'fontsize',20);
    end
end
% save plot
fig.PaperPositionMode = 'auto';
fld_sav = './summary/figures/';
if exist(fld_sav,'dir')~=7
    mkdir(fld_sav);
end
print(fig,[fld_sav,f_name],'-dpng','-r0');

end