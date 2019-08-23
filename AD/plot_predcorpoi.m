% This is to plot predicative correlation to the point of interest (poi) at
% new time point(s)

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

% set the poi (index in the 160x160 grid mesh)
poi=[];
% correlation threshold
% cor_thld=0.1;
sps_den=1;

% data settings
types={'PET','MRI'};
typ=types{1};
groups={'CN','MCI','AD'};
L_grp=length(groups);
% grp=groups{2};
dur=[5,6,4]-1; [~,imax_dur]=max(dur);
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
f_name = ['predcorpoi_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
    fprintf('%s loaded.\n',[f_name,'.mat']);
else
    Var_predm=cell(L_grp,2); Var_prestd=cell(1,L_grp); Times=cell(1,L_grp);
    Corpoi_predm=cell(L_grp,2); Corpoi_prestd=cell(1,L_grp); Thlds=cell(1,L_grp);
    [roi_msk,poi_idx]=get_roipoi; poi=median(cell2mat(reshape(poi_idx,[],1)));
    for gr=1:L_grp
        grp=groups{gr}; J=dur(gr); g.mask=roi{gr};
        fprintf('Processing %s group...\n',grp);
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
            Var_predm{gr,2}=sparse(0); Var_prestd{gr}=sparse(0);
            Corpoi_predm{gr,2}=sparse(0); Corpoi_prestd{gr}=sparse(0); Thlds{gr}=0;
            y_gr=reshape(y_hld{gr},[],size(y_hld{gr},3),size(y_hld{gr},4));
            if ~exist('imsz','var')
                imsz=sqrt(size(y,1)).*ones(1,2);
            end
            g.size=imsz;
            % set roi and poi
            mask=roi_msk(ismember(stdtimes{imax_dur},stdtimes{gr}),gr); %poi=poi_idx(:,gr);
            poidx=sub2ind(imsz,poi(1),poi(2));
            % estimate from data
            Var_predm{gr,1}=reshape(var(y_gr(:,end,:),0,3),imsz); 
            Corpoi_predm{gr,1}=reshape(corr(squeeze(y_gr(:,end,:))',squeeze(y_gr(poidx,end,:))),imsz);
            Var_predm{gr,1}=sparse(Var_predm{gr,1}.*mask{J+1});
            Corpoi_predm{gr,1}=sparse(Corpoi_predm{gr,1}.*mask{J+1});
            % prediction
            if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
                stgp=mgC.stgp;
            else
                ker{1}=feval([stgp_ver,'.GL'],g,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_g,true);
                ker{2}=feval([stgp_ver,'.GP'],t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},optini.Lambda,kappa,l); [ker{1:2}]=deal([]);
            end
            ker{2}=feval([stgp_ver,'.GP'],t_hld,[],[],stgp.C_t.s,stgp.C_t.L,stgp.C_t.jit);
            ker{3}=ker{2};
            N_samp=size(samp_sigma2,1); thin=1;
            prog=0.05:0.05:1; tic;
            for n=1:thin:N_samp
                for k=2:length(ker)
                    ker{k}=ker{k}.update(samp_sigma2(n,k),exp(samp_eta(n,k)));
                end
                Lambda_n=shiftdim(samp_Lambda(n,:,:));
                C_tilt=ker{3}.tomat;
                Lambda_te=C_tilt(end,1:end-1)/C_tilt(1:end-1,1:end-1)*Lambda_n;
                switch l
                    case 1
                        ker_te=feval([stgp_ver,'.GP'],t_hld(end),ker{2}.sigma2,ker{2}.l,stgp.C_t.s,1,stgp.C_t.jit,false);
                        stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),ker_te,Lambda_te);
                        [~,C_z_te]=stgp.tomat; % maybe oversized!
                        var_te=C_z_te(sub2ind([I,I],1:I,1:I))';
                        corpoi_te=C_z_te(:,poidx)./sqrt(var_te(poidx))'./sqrt(var_te);
                    case 2
                        PhiLambda2_te=stgp.C_x.eigf.*Lambda_te.^2;
                        var_te=sum(PhiLambda2_te.*stgp.C_x.eigf,2)+ker{2}.sigma2;
                        corpoi_te=PhiLambda2_te*stgp.C_x.eigf(poidx,:)'; corpoi_te(poidx)=corpoi_te(poidx)+ker{2}.sigma2;
                        corpoi_te=corpoi_te./sqrt(var_te(poidx))'./sqrt(var_te);
                end
                % impose ROI
                var_te=reshape(var_te,imsz).*mask{J+1};
                corpoi_te=reshape(corpoi_te,imsz).*mask{J+1};
                % sparsify the correlation
%                 corpoi_thld=quantile(abs(nonzeros(corpoi_te)),1-sps_den);
% %                 corpoi_thld=maxk(abs(nonzeros(corpoi_te)),min([nnz(corpoi_te),ceil(sps_den*numel(corpoi_te)/2)])); corpoi_thld=corpoi_thld(end);
%                 sps_ind=abs(corpoi_te)>corpoi_thld;
%                 corpoi_te=corpoi_te.*sps_ind;
                Var_predm{gr,2}=Var_predm{gr,2}+var_te; Var_prestd{gr}=Var_prestd{gr}+var_te.^2;
                Corpoi_predm{gr,2}=Corpoi_predm{gr,2}+corpoi_te; Corpoi_prestd{gr}=Corpoi_prestd{gr}+corpoi_te.^2;
%                 Thlds{j}=Thlds{j}+cor_thld;
                % display the progress
                iter=1+floor(n/thin);
                if ismember(iter,floor(N_samp/thin.*prog))
                    fprintf('%.0f%% completed.\n',100*iter/(N_samp/thin));
                end
            end
            time_gr=toc; fprintf('\n %.2f seconds used.\n', time_gr);
            Var_predm{gr,2}=Var_predm{gr,2}./(N_samp/thin); Var_prestd{gr}=sqrt(Var_prestd{gr}./(N_samp/thin) - Var_predm{gr,2}.^2);
%             Var_predm{gr,2}=reshape(Var_predm{gr,2},imsz); Var_prestd{gr}=reshape(Var_prestd{gr},imsz);
            Corpoi_predm{gr,2}=Corpoi_predm{gr,2}./(N_samp/thin); Corpoi_prestd{gr}=sqrt(Corpoi_prestd{gr}./(N_samp/thin) - Corpoi_predm{gr,2}.^2);
%             Corpoi_predm{gr,2}=reshape(Corpoi_predm{gr,2},imsz); Corpoi_prestd{gr}=reshape(Corpoi_prestd{gr},imsz);
%             Thlds{gr}=Thlds{gr}./(N_samp/thin);
            Times{gr}=t;
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'Var_predm','Var_prestd','Corpoi_predm','Corpoi_prestd','Times','Thlds','poi_idx','stdtimes','-v7.3');
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
%     set(fig,'pos',[0 800 800 600]);
    set(fig,'pos',[0 800 1000 400]);
%     ha=tight_subplot(2,L_grp,[.07,.05],[.06,.07],[.04,.05]);
    ha=tight_subplot(1,L_grp,[.07,.05],[.06,.07],[.05,.06]);
    clim=[+Inf,-Inf];
%     clim=[0,1];
    
%     [~,poi_idx]=get_roipoi; 
    poi=median(cell2mat(reshape(poi_idx,[],1)));
    for gr=1:L_grp
        grp=groups{gr};
        idx_g=find(stdtimes{imax_dur}==stdtimes{gr}(end));
        
%         % plot correlation to poi estimated from data
%         im_g=Corpoi_predm{gr,1};
%         if ~isempty(im_g)
%             h_sub=subplot(ha(gr));
%             imagesc(im_g,clim); hold on;
%             plot(poi(2),poi(1),'rx','markersize',18,'linewidth',4);
%             set(gca,'xticklabel',[],'yticklabel',[]);
%             title([grp, ' (',Tlab{idx_g},')'],'fontsize',20);
% %             cmin_j=min(im_g(:)); cmax_j=max(im_g(:));
% %             if cmin_j<clim(1)
% %                 clim(1)=cmin_j;
% %             end
% %             if cmax_j>clim(2)
% %                 clim(2)=cmax_j;
% %             end
%         end
        
        % plot prediction
        im_g=Corpoi_predm{gr,2};%.*32767;
        if ~isempty(im_g)
%             h_sub=subplot(ha(L_grp+gr));
            h_sub=subplot(ha(gr));
%             imshow(im_g);
            imagesc(im_g); hold on;
            plot(poi(2),poi(1),'rx','markersize',18,'linewidth',4);
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
    end
    % adjust color scale afterwards
    if all(isfinite(clim))
        for j=1:length(ha)
            h_sub=subplot(ha(j));
            caxis(clim);
        end
    end
    % add common colorbar
    h_pos=h_sub.Position;
%     colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4).*2.18],'fontsize',14);
    colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4)],'fontsize',14);
    caxis(clim);
    % save plot
    fig.PaperPositionMode = 'auto';
    print(fig,[fld_sav,f_name],'-dpng','-r0');
end

end