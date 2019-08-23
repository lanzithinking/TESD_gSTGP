% This is to plot predicative mean function of time with fixed locations as
% functions of time

clear;
sufx={'','_mtimesx','_gpu'};
stgp_ver=['STGP',sufx{2}];
addpath('../util/');
% addpath(['../util/+',stgp_ver,'/']);
if contains(stgp_ver,'mtimesx')
    addpath('../util/mtimesx/');
end
addpath('../util/Image Graphs/');
addpath('../util/tight_subplot/');
% addpath(genpath('../util/boundedline-pkg/'));
% Random Numbers...
seedNO = 2018;
seed = RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% data settings
types={'PET','MRI'};
typ=types{1};
groups={'CN','MCI','AD'};
L_grp=length(groups);
% grp=groups{2};
dur=[5,6,4]-1;
stdtimes={[0:.5:1,2:3]',[0:.5:1.5,2:3]',[0:.5:1,2]'};
L=100;
d=2;
sec=48;
% model options
models={'kron_prod','kron_sum'};
L_mdl=length(models);
opthypr=false;
jtupt=false;
% intM=false;
alg_name='MCMC';
if opthypr
    alg_name=['opt',alg_name];
    if jtupt
        alg_name=['jt',alg_name];
    end
end
% obtain ROI
roi=get_roipoi(.75,'stackt');

% parameters of kernel
s=2; % smoothness
kappa=.2; % decaying rate for dynamic eigenvalues
% graph kernel
g.w=1;% g.size=imsz; g.mask=roi{grp_opt};
jit_g=1e-6;
% temporal kernels
jit_t=1e-6;

%% estimation

% estimate the mean from MCMC samples
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
for l=2:L_mdl
intM=true;
keywd = {[alg_name,'_',repmat('intM_',intM),models{l}],['_L',num2str(L),'_d',num2str(d)]};
f_name = ['predmft_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
    fprintf('%s loaded.\n',[f_name,'.mat']);
else
    M_predm=cell(1,L_grp); M_prestd=M_predm; Times=M_predm;
    for gr=1:L_grp
        grp=groups{gr}; J=dur(gr); g.mask=roi{gr};
        % read the hold-out data
        [t_hld,y_hld{gr}]=read_data(typ,grp,J+1);
        tt=datetime(t_hld);
        tt=datenum(tt-repmat(tt(1,:),size(tt,1),1));
        t_hld=tt./365;
        rmind=sum(abs(t_hld-stdtimes{gr})>.55)>0;
        if any(rmind)
            fprintf('%d subject(s) removed!\n',sum(rmind));
        end
        t_hld=mean(t_hld(:,~rmind),2);
        yy=cell2mat(shiftdim(y_hld{gr},-3)); % MRI images need to standize the sizes
        if d==2
            yy=squeeze(yy(:,:,sec,:,~rmind));
        end
        y_hld{gr}=double(yy)./32767; yy=[];
        % read fitted data
        keywd{3}=['_J',num2str(J)];
        found=false;
        for k=1:nfiles
            fname_k=files(k+2).name;
            if contains(fname_k,['_',grp,'_']) && contains(fname_k,keywd{1}) && contains(fname_k,keywd{2}) && contains(fname_k,keywd{3})
                load(strcat(folder, fname_k));
                fprintf('%s loaded.\n',fname_k);
                found=true; break;
            end
        end
        if found
            M_predm{gr}=0; M_prestd{gr}=0;
            y_gr=reshape(y_hld{gr},[],size(y_hld{gr},3),size(y_hld{gr},4));
            if ~exist('imsz','var')
                imsz=sqrt(size(y,1)).*ones(1,2);
            end
            g.size=imsz;
            % prediction
            if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
                stgp=mgC.stgp;
            else
                ker{1}=feval([stgp_ver,'.GL'],g,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_g,true);
                ker{2}=feval([stgp_ver,'.GP'],t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},optini.Lambda,kappa,l); [ker{1:2}]=deal([]);
                mgC=feval([stgp_ver,'.mg'],stgp,K,optini.sigma2(1),L);
            end
            ker{2}=feval([stgp_ver,'.GP'],t_hld,[],[],stgp.C_t.s,stgp.C_t.L,stgp.C_t.jit);
            ker{3}=ker{2};
            mgC.K=size(y_hld{gr},4);
            N_samp=size(samp_sigma2,1); M_predvar=zeros(I,1);
            for n=1:N_samp
                for k=2:length(ker)
                    ker{k}=ker{k}.update(samp_sigma2(n,k),exp(samp_eta(n,k)));
                end
                Lambda_n=shiftdim(samp_Lambda(n,:,:));
                switch l
                    case 1
                        C_tilt=ker{3}.tomat;
                        Lambda=[Lambda_n;C_tilt(end,1:end-1)/C_tilt(1:end-1,1:end-1)*Lambda_n];
                        stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),ker{2},Lambda);
                        [~,C_z_n]=stgp.mult(full(sparse(J*I+(1:I),(1:I),1,I*(J+1),I)));
                        C_E=C_z_n(J*I+(1:I),:); C_ED=C_z_n(1:J*I,:)'; % oversized!
                    case 2
                        C_t=ker{2}.tomat;
                        C_E=kron(C_t(end,end),speye(I));
                        C_ED=kron(C_t(end,1:end-1),speye(I));
                end
                mgC=mgC.update(stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),mgC.stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),Lambda_n),samp_sigma2(n,1));
                mu=mgC.predM(y_gr(:,1:end-1,:),C_E,C_ED);
                M_predm{gr}=M_predm{gr}+mu; M_predvar=M_predvar+mu.^2;
%                 var_n=diag(C_E)-sum(C_ED.*mgC.solve(C_ED',true)',2);
%                 M_prestd{gr}=M_prestd{gr}+var_n;
            end
            M_predm{gr}=M_predm{gr}./N_samp; M_predvar=M_predvar./N_samp - M_predm{gr}.^2;
            M_prestd{gr}=sqrt(M_prestd{gr}./N_samp + M_predvar);
            Times{gr}=t;
            % reshape
            M_predm{gr}=reshape(M_predm{gr},imsz(1),imsz(2),[]); M_prestd{gr}=reshape(M_prestd{gr},imsz(1),imsz(2),[]);
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'y_hld','M_predm','M_prestd','Times','stdtimes');
end


%% plot

% figure settings
Tlab={'Initial','6 months','1 year','18 months','2 years','3 years','4 years'};
[~,imax_dur]=max(dur);

if exist([folder,f_name,'.mat'],'file')
    % figure setting
    fld_sav = './summary/figures/';
    if exist(fld_sav,'dir')~=7
        mkdir(fld_sav);
    end
    fig=figure(1); clf(fig);
    set(fig,'pos',[0 800 800 600]);
    ha=tight_subplot(2,L_grp,[.07,.05],[.06,.07],[.04,.05]);
    clim=[+Inf,-Inf];
    for gr=1:L_grp
        grp=groups{gr};
        idx_g=find(stdtimes{imax_dur}==stdtimes{gr}(end));
        
        % plot actual data
        y_gr=y_hld{gr};
        im_g=y_gr(:,:,end,1);
%         im_g=mean(y_gr(:,:,end,:),4);
        if ~isempty(im_g)
            h_sub=subplot(ha(gr));
            imagesc(im_g);
            set(gca,'xticklabel',[],'yticklabel',[]);
            title([grp, ' (',Tlab{idx_g},')'],'fontsize',20);
            cmin_j=min(im_g(:)); cmax_j=max(im_g(:));
            if cmin_j<clim(1)
                clim(1)=cmin_j;
            end
            if cmax_j>clim(2)
                clim(2)=cmax_j;
            end
        end
        
        % plot prediction
        im_g=M_predm{gr};%.*32767;
        if ~isempty(im_g)
            h_sub=subplot(ha(L_grp+gr));
%             imshow(im_g);
            imagesc(im_g);
            set(gca,'xticklabel',[],'yticklabel',[]);
%             title([grp, ' (',Tlab{idx_g},')'],'fontsize',20);
            cmin_j=min(im_g(:)); cmax_j=max(im_g(:));
            if cmin_j<clim(1)
                clim(1)=cmin_j;
            end
            if cmax_j>clim(2)
                clim(2)=cmax_j;
            end
        end
    end
%     % adjust color scale afterwards
%     if all(isfinite(clim))
%         for j=1:length(ha)
%             h_sub=subplot(ha(j));
%             caxis(clim);
%         end
%     end
%     % add common colorbar
%     h_pos=h_sub.Position;
%     colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4).*2.18],'fontsize',14);
%     caxis(clim);
    % save plot
    fig.PaperPositionMode = 'auto';
    print(fig,[fld_sav,f_name],'-dpng','-r0');
end

end