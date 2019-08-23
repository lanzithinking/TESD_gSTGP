% This is to plot extended correlation to the point of interest (poi) at
% new location(s)

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

% correlation threshold
% cor_thld=0.1;
% sps_den=.1;
% coarsen the image for refining
imcrsn=4;

% data settings
types={'PET','MRI'};
typ=types{1};
groups={'CN','MCI','AD'};
L_grp=length(groups);
% grp=groups{2};
dur=[5,6,4]; [max_dur,imax_dur]=max(dur);
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
% obtain full ROI for inference
roi_full=get_roipoi(.75,'stackt');

% parameters of kernel
s=2; % smoothness
kappa=.2; % decaying rate for dynamic eigenvalues
% full graph kernel
imsz_full=[160,160]; I_full=prod(imsz_full);
g_full.w=1; g_full.size=imsz_full; %g_full.mask=roi_full{grp_opt};
imsz=imsz_full./imcrsn; I=prod(imsz);
% selecting matrix for coarse mesh
[gdx,gdy]=meshgrid(1:imcrsn:imsz_full(1),1:imcrsn:imsz_full(2));
S=sparse(gdx,gdy,1,imsz_full(1),imsz_full(2));
nzid=find(S(:));
jit_g=1e-6;
% temporal kernels
jit_t=1e-6;

%% estimation

% estimate the mean from MCMC samples
% load data
folder = './summary/extX/';
files = dir(folder);
nfiles = length(files) - 2;
for l=2:L_mdl
intM=true;
keywd = {[alg_name,'_',repmat('intM_',intM),models{l}],['_L',num2str(L),'_d',num2str(d)]};
f_name = ['extdcorpoi_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
    fprintf('%s loaded.\n',[f_name,'.mat']);
else
    Var_estm=cell(max_dur,L_grp); Var_estd=Var_estm;
    Corpoi_estm=cell(max_dur,L_grp); Corpoi_estd=Corpoi_estm;
    Var_extm=cell(max_dur,L_grp); Var_extstd=Var_extm; 
    Corpoi_extm=cell(max_dur,L_grp); Corpoi_extstd=Corpoi_extm; 
    Times=cell(1,L_grp);
    % roi and poi for displaying
    [roi_full_msk,poi_full_idx]=get_roipoi; poi_full=median(cell2mat(reshape(poi_full_idx,[],1)));
    roi_msk=cell(max_dur,L_grp); poi_idx=roi_msk;
    for gr=1:L_grp
        for j=1:dur(gr)
            jdx=find(stdtimes{imax_dur}==stdtimes{gr}(j));
            roi_msk{jdx,gr}=roi_full_msk{jdx,gr}(1:imcrsn:imsz_full(1),1:imcrsn:imsz_full(2));
            poi_idx{jdx,gr}=ceil(poi_full_idx{jdx,gr}./imcrsn);
        end
    end
    poi=median(cell2mat(reshape(poi_idx,[],1)));
    for gr=1:L_grp
        grp=groups{gr}; J=dur(gr); g_full.mask=roi_full{gr};
        fprintf('Processing %s group...\n',grp);
        % read fitted data
        keywd{3}=['_I',num2str(I),'_J',num2str(J)];
        found=false;
        for k=1:nfiles
            fname_k=files(k+2).name;
            if contains(fname_k,['_',grp,'_']) && contains(fname_k,keywd{1}) && contains(fname_k,keywd{2}) && contains(fname_k,keywd{3})
                load([folder, fname_k]);
                fprintf('%s loaded.\n',fname_k);
                found=true; break;
            end
        end
        if found
            [Var_estm{:,gr}]=deal(sparse(0)); [Var_estd{:,gr}]=deal(sparse(0));
            [Corpoi_estm{:,gr}]=deal(sparse(0)); [Corpoi_estd{:,gr}]=deal(sparse(0));
            [Var_extm{:,gr}]=deal(sparse(0)); [Var_extstd{:,gr}]=deal(sparse(0));
            [Corpoi_extm{:,gr}]=deal(sparse(0)); [Corpoi_extstd{:,gr}]=deal(sparse(0));
            if ~exist('imsz','var')
                imsz=sqrt(size(y,1)).*ones(1,2);
            end
            N=prod(imsz); %g.size=imsz;
            % set roi and poi
            mask_full=roi_full_msk(:,gr);
            poidx_full=sub2ind(imsz_full,poi_full(1),poi_full(2));
            mask=roi_msk(:,gr); %poi=median(cell2mat(poi_idx(:,gr)));
            poidx=sub2ind(imsz,poi(1),poi(2));
            % estimate and prediction
            C_x=feval([stgp_ver,'.GL'],g_full,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_g,true); % full spatial kernel
            if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
                stgp=mgC.stgp;
            else
                ker{1}=feval([stgp_ver,'.GL'],g,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_g,true);
                ker{2}=feval([stgp_ver,'.GP'],t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},optini.Lambda,kappa,l); [ker{1:2}]=deal([]);
            end
            N_samp=size(samp_sigma2,1); thin=1;
            prog=0.05:0.05:1; tic;
            for n=1:thin:N_samp
                C_x=C_x.update([],exp(samp_eta(n,1)));
                stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),shiftdim(samp_Lambda(n,:,:)));
                for j=1:J
                    jdx=find(stdtimes{imax_dur}==stdtimes{gr}(j));
                    [gdy,gdx]=meshgrid(1:L,nzid);
                    switch l
                        case 1
                            % estimation on coarse mesh
                            [~,sliceC]=stgp.mult(ndSparse(sparse((j-1)*I+(1:I),1:I,1,I*J,I)));
                            sliceC=sliceC((j-1)*I+(1:I),:);
                            var_j_est=sliceC(sub2ind([I,I],1:I,1:I));
                            corpoi_j_est=sliceC(:,poidx);
                            % extension on fine mesh
                            Phi_xteXcholC_j=sparse(gdx,gdy,stgp.C_x.solve(chol(sliceC,'lower')),I_full,L);
                            Phi_xteXcholC_j=C_x.mult(Phi_xteXcholC_j);
                            var_j_ext=sum(Phi_xteXcholC_j.^2,2);
                            Phi_xteXcovpoi_j=sparse(nzid,ones(I,1),stgp.C_x.solve(sliceC(:,poidx)),I_full,1);
                            corpoi_j_ext=C_x.mult(Phi_xteXcovpoi_j);
                        case 2
                            % estimation on coarse mesh
                            PhiLambda_j=stgp.C_x.eigf.*stgp.Lambda(j,:);
                            var_j_est=sum(PhiLambda_j.^2,2)+stgp.C_t.sigma2;
                            corpoi_j_est=PhiLambda_j*PhiLambda_j(poidx,:)'; corpoi_j_est(poidx)=corpoi_j_est(poidx)+stgp.C_t.sigma2;
                            % extension on fine mesh
%                             Phi_xteXLambda_j=sparse(I_full,L);
%                             Phi_xteXLambda_j(nzid,:)=PhiLambda_j./C_x.eigv';
                            Phi_xteXLambda_j=sparse(gdx,gdy,PhiLambda_j./C_x.eigv',I_full,L);
                            Phi_xteXLambda_j=C_x.mult(Phi_xteXLambda_j);
                            var_j_ext=sum(Phi_xteXLambda_j.^2,2);
                            corpoi_j_ext=Phi_xteXLambda_j*PhiLambda_j(poidx,:)';
                    end
                    corpoi_j_est=corpoi_j_est./sqrt(var_j_est(poidx))'./sqrt(var_j_est);
                    corpoi_j_ext=corpoi_j_ext./sqrt(var_j_ext(poidx))'./sqrt(var_j_ext);
                    % impose ROI
                    % estimation
                    var_j_est=reshape(var_j_est,imsz).*mask{jdx};
                    corpoi_j_est=reshape(corpoi_j_est,imsz).*mask{jdx};
                    Var_estm{jdx,gr}=Var_estm{jdx,gr}+var_j_est; Var_estd{jdx,gr}=Var_estd{jdx,gr}+var_j_est.^2;
                    Corpoi_estm{jdx,gr}=Corpoi_estm{jdx,gr}+corpoi_j_est; Corpoi_estd{jdx,gr}=Corpoi_estd{jdx,gr}+corpoi_j_est.^2;
                    %  extension
                    var_j_ext=reshape(var_j_ext,imsz_full).*mask_full{jdx};
                    corpoi_j_ext=reshape(corpoi_j_ext,imsz_full).*mask_full{jdx};
                    Var_extm{jdx,gr}=Var_extm{jdx,gr}+var_j_ext; Var_extstd{jdx,gr}=Var_extstd{jdx,gr}+var_j_ext.^2;
                    Corpoi_extm{jdx,gr}=Corpoi_extm{jdx,gr}+corpoi_j_ext; Corpoi_extstd{jdx,gr}=Corpoi_extstd{jdx,gr}+corpoi_j_ext.^2;
                end
                % display the progress
                iter=1+floor(n/thin);
                if ismember(iter,floor(N_samp/thin.*prog))
                    fprintf('%.0f%% completed.\n',100*iter/(N_samp/thin));
                end
            end
            time_gr=toc; fprintf('\n %.2f seconds used.\n', time_gr);
            % postprocess
            for j=1:J
                jdx=find(stdtimes{imax_dur}==stdtimes{gr}(j));
                % estimation
                Var_estm{jdx,gr}=Var_estm{jdx,gr}./(N_samp/thin); Var_estd{jdx,gr}=sqrt(Var_estd{jdx,gr}./(N_samp/thin) - Var_estm{jdx,gr}.^2);
                Corpoi_estm{jdx,gr}=Corpoi_estm{jdx,gr}./(N_samp/thin); Corpoi_estd{jdx,gr}=sqrt(Corpoi_estd{jdx,gr}./(N_samp/thin) - Corpoi_estm{jdx,gr}.^2);
                % extension
                Var_extm{jdx,gr}=Var_extm{jdx,gr}./(N_samp/thin); Var_extstd{jdx,gr}=sqrt(Var_extstd{jdx,gr}./(N_samp/thin) - Var_extm{jdx,gr}.^2);
                Corpoi_extm{jdx,gr}=Corpoi_extm{jdx,gr}./(N_samp/thin); Corpoi_extstd{jdx,gr}=sqrt(Corpoi_extstd{jdx,gr}./(N_samp/thin) - Corpoi_extm{jdx,gr}.^2);
            end
            Times{gr}=t;
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'Var_estm','Var_estd','Corpoi_estm','Corpoi_estd',...
        'Var_extm','Var_extstd','Corpoi_extm','Corpoi_extstd','Times','poi_idx','poi_full_idx','stdtimes','-v7.3');
end


%% plot

% figure settings
Tlab={'Initial','6 months','1 year','18 months','2 years','3 years','4 years'};

if exist([folder,f_name,'.mat'],'file')
    % saving figures
    fld_sav = [folder,'figures/'];
    if exist(fld_sav,'dir')~=7
        mkdir(fld_sav);
    end
    
%     [~,poi_idx]=get_roipoi; 
    poi=ceil(mean(cell2mat(reshape(poi_idx(1:4,:),[],1))));
    poi_full=ceil(mean(cell2mat(reshape(poi_full_idx(1:4,:),[],1))));
%     poi_full=median(cell2mat(reshape(poi_full_idx,[],1)));
%     poi=median(cell2mat(reshape(poi_idx,[],1)));
    for gr=1:L_grp
        grp=groups{gr};
        
%         % plot variance
%         Var_est_gr=Var_estm(:,gr); Var_ext_gr=Var_extm(:,gr);
%         if ~isempty(Var_est_gr{1})
%             fig=figure((l-1)*L_grp+gr); clf(fig);
%             set(fig,'pos',[0 800 1200 500]);
% %             ha=tight_subplot(2,J,[.1,.02],[.08,.125],[.02,.02]);
%             ha=tight_subplot(2,max_dur,[.07,.02],[.07,.09],[.02,.06]);
%             cmin=cellfun(@(x)min(nonzeros(x)),Var_est_gr,'UniformOutput',false);
%             cmax=cellfun(@(x)max(nonzeros(x)),Var_est_gr,'UniformOutput',false);
%             clim(1,:)=[min(cell2mat(cmin)),max(cell2mat(cmax))];
%             cmin=cellfun(@(x)min(nonzeros(x)),Var_ext_gr,'UniformOutput',false);
%             cmax=cellfun(@(x)max(nonzeros(x)),Var_ext_gr,'UniformOutput',false);
%             clim(2,:)=[min(cell2mat(cmin)),max(cell2mat(cmax))];
%             
%             % estimation
%             for j=1:max_dur
%                 h_sub=subplot(ha(j));
%                 if any(stdtimes{imax_dur}(j)==stdtimes{gr})
%                     im_j=Var_est_gr{j};
%     %                 imshow(im_j);
%                     imagesc(im_j,clim(1,:));
%                     set(gca,'xticklabel',[],'yticklabel',[]);
%                     title([grp, ' (',Tlab{j},')'],'fontsize',20);
%                 else
%                     axis off;
%                 end
%             end
%             % add common colorbar
%             h_pos=h_sub.Position;
%             colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4)],'fontsize',14);
%             caxis(clim(1,:));
%             
%             % extension
%             for j=1:max_dur
%                 h_sub=subplot(ha(max_dur+j));
%                 if any(stdtimes{imax_dur}(j)==stdtimes{gr})
%                     im_j=Var_ext_gr{j};
%     %                 imshow(im_j);
%                     imagesc(im_j,clim(2,:));
%                     set(gca,'xticklabel',[],'yticklabel',[]);
% %                     title([grp, ' (',Tlab{j},')'],'fontsize',20);
%                 else
%                     axis off;
%                 end
%             end
%             % add common colorbar
%             h_pos=h_sub.Position;
%             colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4)],'fontsize',14);
%             caxis(clim(2,:));
%             % save plot
%             fig.PaperPositionMode = 'auto';
%             print(fig,[fld_sav,grp,'_extvarft_',keywd{:}],'-dpng','-r0');
%         end
        
        % plot correlation of poi
        Corpoi_est_gr=Corpoi_estm(:,gr); Corpoi_ext_gr=Corpoi_extm(:,gr);
        if ~isempty(Corpoi_est_gr{1})
            fig=figure((L_mdl+l-1)*L_grp+gr); clf(fig);
            set(fig,'pos',[0 800 1200 500]);
            ha=tight_subplot(2,max_dur,[.07,.02],[.07,.09],[.02,.06]);
            clim=[0,1];
            
            % estimation
%             poi_gr=poi_idx(:,gr);
            for j=1:max_dur
                h_sub=subplot(ha(j));
                if any(stdtimes{imax_dur}(j)==stdtimes{gr})
                    im_j=Corpoi_est_gr{j};
%                     imshow(im_j);
                    imagesc(im_j,clim); hold on;
%                     plot(poi_gr{j}(2),poi_gr{j}(1),'rx','markersize',18,'linewidth',4);
                    plot(poi(2),poi(1),'rx','markersize',18,'linewidth',4);
                    set(gca,'xticklabel',[],'yticklabel',[]);
                    title([grp, ' (',Tlab{j},')'],'fontsize',20);
                else
                    axis off;
                end
            end
            % add common colorbar
            h_pos=h_sub.Position;
            colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4)],'fontsize',14);
            caxis(clim);
            
            % extension
            for j=1:max_dur
                h_sub=subplot(ha(max_dur+j));
                if any(stdtimes{imax_dur}(j)==stdtimes{gr})
                    im_j=Corpoi_ext_gr{j};
%                     imshow(im_j);
                    imagesc(im_j,clim); hold on;
                    plot(poi_full(2),poi_full(1),'rx','markersize',18,'linewidth',4);
                    set(gca,'xticklabel',[],'yticklabel',[]);
%                     title([grp, ' (',Tlab{j},')'],'fontsize',20);
                else
                    axis off;
                end
            end
            % add common colorbar
            h_pos=h_sub.Position;
            colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4)],'fontsize',14);
            caxis(clim);
            
            % save plot
            fig.PaperPositionMode = 'auto';
            print(fig,[fld_sav,grp,'_',f_name],'-dpng','-r0');
        end
    end
end

end
