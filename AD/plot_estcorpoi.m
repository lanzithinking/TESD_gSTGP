% This is to plot estimated correlation to the point of interest (poi) as a 
% function of time.
% The result is a series of dynamic images showing the correlation between
% the pixels of whole image and the poi.

clear;
sufx={'','_mtimesx','_gpu'};
stgp_ver=['STGP',sufx{1}];
addpath('../util/');
% addpath(['../util/+',stgp_ver,'/']);
if contains(stgp_ver,'mtimesx')
    addpath('../util/mtimesx/');
end
addpath('../util/Image Graphs/');
addpath('../util/ndSparse/');
addpath('../util/tight_subplot/');
% addpath(genpath('../util/boundedline-pkg/'));
% if ~isdeployed
%     addpath ~/STGP/code/util/;
%     addpath ~/STGP/code/util/STGP/;
%     addpath ~/STGP/code/util/ndSparse/;
%     addpath ~/STGP/code/util/Image_Graphs/;
%     addpath ~/STGP/code/util/tight_subplot/;
% end
% Random Numbers...
seedNO = 2018;
seed = RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% set the poi (index in the 160x160 grid mesh)
poi=[];
% correlation threshold
% cor_thld=0.1;
% sps_den=1;

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
f_name = ['estcorpoi_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
    fprintf('%s loaded.\n',[f_name,'.mat']);
else
    Var_estm=cell(max_dur,L_grp); Var_estd=Var_estm; Times=cell(1,L_grp);
    Corpoi_estm=cell(max_dur,L_grp); Corpoi_estd=Corpoi_estm; Thlds=Corpoi_estm;
    [roi_msk,poi_idx]=get_roipoi; poi=ceil(mean(cell2mat(reshape(poi_idx(1:4,:),[],1))));
    for gr=1:L_grp
        grp=groups{gr}; J=dur(gr); g.mask=roi{gr};
        fprintf('Processing %s group...\n',grp);
        keywd{3}=['_J',num2str(J)];
%         % set the POI
%         if isempty(poi)
%             if ~exist([folder,grp,'_estcovft_',keywd{:},'.mat'],'file')
%                 plot_estcovft_bigblk_f(gr,l,[],[],intM);
%             end
%             load([folder,grp,'_estcovft_',keywd{:},'.mat'],'Var_estm');
% %             [~,poi]=max(sum(cat(3,Var_estm{:}),3),[],[1,2]);
%             Var_sum=sum(cat(3,Var_estm{:}),3);
%             [~,poi]=max(Var_sum(:));
%         end
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
            [Var_estm{:,gr}]=deal(sparse(0)); [Var_estd{:,gr}]=deal(sparse(0));
            [Corpoi_estm{:,gr}]=deal(sparse(0)); [Corpoi_estd{:,gr}]=deal(sparse(0)); [Thlds{:,gr}]=deal(0);
            if ~exist('imsz','var')
                imsz=sqrt(size(y,1)).*ones(1,2);
            end
            g.size=imsz;
            N=prod(imsz); I=N; J=length(t);
            % get mask
%             mask=cell(1,J);
%             for j=1:J
%                 mask{j}=roi(reshape(samp_M(:,j,1),imsz));
% %                 figure(j); imshow(full(mask{j}));
%             end
%             % get poi
%             M_sum=sum(samp_M(:,:,1).*reshape(cell2mat(mask),I,J),2);
%             [~,poi]=max(M_sum(:));
            mask=roi_msk(:,gr); %poi=poi_idx(:,gr);
            poidx=sub2ind(imsz,poi(1),poi(2));
            if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
                stgp=mgC.stgp;
            else
                ker{1}=STGP.GL(g,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_g,true);
                ker{2}=STGP.GP(t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                stgp=STGP.hd(ker{1},ker{2},optini.Lambda,kappa,l); [ker{1:2}]=deal([]);
            end
            % poi index in the correlation matrix
%             poidx=poi;
            N_samp=size(samp_sigma2,1); thin=1;
            prog=0.05:0.05:1; tic;
            for n=1:thin:N_samp
                stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),shiftdim(samp_Lambda(n,:,:)));
                for j=1:J
                    jdx=find(stdtimes{imax_dur}==stdtimes{gr}(j));
%                     poidx=sub2ind(imsz,poi{j}(1),poi{j}(2));
                    if l==2
%                         Phi_x=mask{jdx}(:).*stgp.C_x.eigf;
%                         PhiLambda2_j=Phi_x.*stgp.Lambda(j,:).^2;
%                         var_j=sum(PhiLambda2_j.*Phi_x,2)+stgp.C_t.sigma2;
                        PhiLambda2_j=stgp.C_x.eigf.*stgp.Lambda(j,:).^2;
                        var_j=sum(PhiLambda2_j.*stgp.C_x.eigf,2)+stgp.C_t.sigma2;
                        corpoi_j=PhiLambda2_j*stgp.C_x.eigf(poidx,:)'; corpoi_j(poidx)=corpoi_j(poidx)+stgp.C_t.sigma2;
                        corpoi_j=corpoi_j./sqrt(var_j(poidx))'./sqrt(var_j);
                    else
                        [~,sliceC]=stgp.mult(ndSparse(sparse((j-1)*I+(1:I),1:I,1,I*J,I)));
                        sliceC=sliceC((j-1)*I+(1:I),:);
                        var_j=sliceC(sub2ind([I,I],1:I,1:I));
                        corpoi_j=sliceC(:,poidx)./sqrt(var_j(poidx))'./sqrt(var_j);
                    end
                    % impose ROI
                    var_j=reshape(var_j,imsz).*mask{jdx};
                    corpoi_j=reshape(corpoi_j,imsz).*mask{jdx};
%                     % sparsify the correlation
%                     corpoi_thld=quantile(abs(nonzeros(corpoi_j)),1-sps_den);
% %                     corpoi_thld=maxk(abs(nonzeros(corpoi_j)),min([nnz(corpoi_j),ceil(sps_den*numel(corpoi_j)/2)])); corpoi_thld=corpoi_thld(end);
%                     sps_ind=abs(corpoi_j)>corpoi_thld;
%                     corpoi_j=corpoi_j.*sps_ind;
                    Var_estm{jdx,gr}=Var_estm{jdx,gr}+var_j; Var_estd{jdx,gr}=Var_estd{jdx,gr}+var_j.^2;
                    Corpoi_estm{jdx,gr}=Corpoi_estm{jdx,gr}+corpoi_j; Corpoi_estd{jdx,gr}=Corpoi_estd{jdx,gr}+corpoi_j.^2;
%                     Thlds{jdx,gr}=Thlds{jdx,gr}+cor_thld;
                end
                % display the progress
                iter=1+floor(n/thin);
                if ismember(iter,floor(N_samp/thin.*prog))
                    fprintf('%.0f%% completed.\n',100*iter/(N_samp/thin));
                end
            end
            time_gr=toc; fprintf('\n %.2f seconds used.\n', time_gr);
            for j=1:J
                jdx=find(stdtimes{imax_dur}==stdtimes{gr}(j));
                Var_estm{jdx,gr}=Var_estm{jdx,gr}./(N_samp/thin); Var_estd{jdx,gr}=sqrt(Var_estd{jdx,gr}./(N_samp/thin) - Var_estm{jdx,gr}.^2);
%                 Var_estm{jdx,gr}=reshape(Var_estm{jdx,gr},imsz); Var_estd{jdx,gr}=reshape(Var_estd{jdx,gr},imsz);
                Corpoi_estm{jdx,gr}=Corpoi_estm{jdx,gr}./(N_samp/thin); Corpoi_estd{jdx,gr}=sqrt(Corpoi_estd{jdx,gr}./(N_samp/thin) - Corpoi_estm{jdx,gr}.^2);
%                 Corpoi_estm{jdx,gr}=reshape(Corpoi_estm{jdx,gr},imsz); Corpoi_estd{jdx,gr}=reshape(Corpoi_estd{jdx,gr},imsz);
%                 if nnz(Corpoi_estm{jdx,gr})/numel(Corpoi_estm{jdx,gr})>sps_den
% %                     corpoi_thld=maxk(abs(nonzeros(Corpoi_estm{jdx,gr})),min([nnz(Corpoi_estm{jdx,gr}),ceil(sps_den*numel(Corpoi_estm{jdx,gr})/2)])); corpoi_thld=corpoi_thld(end);
%                     corpoi_thld=maxk(abs(nonzeros(Corpoi_estm{jdx,gr})),ceil(sps_den*numel(Corpoi_estm{jdx,gr})/2)); corpoi_thld=corpoi_thld(end);
%                     sps_ind=abs(Corpoi_estm{jdx,gr})>=corpoi_thld;
%                     Corpoi_estm{jdx,gr}=Corpoi_estm{jdx,gr}.*sps_ind; Corpoi_estd{jdx,gr}=Corpoi_estd{jdx,gr}.*sps_ind; 
%                 end
%                 Thlds{jdx,gr}=Thlds{jdx,gr}./(N_samp/thin);
            end
            Times{gr}=t;
%             % temporary saving
%             Var_estm_g=Var_estm(:,gr); Var_estd_g=Var_estd(:,gr); Corpoi_estm_g=Corpoi_estm(:,gr); Corpoi_estd_g=Corpoi_estd(:,gr);
%             Times_g=Times(gr); Thlds_g=Thlds(:,gr);
%             save([folder,grp,'_',f_name,'.mat'],'Var_estm_g','Var_estd_g','Corpoi_estm_g','Corpoi_estd_g','Times_g','Thlds_g','poi','stdtimes','-v7.3');
%             Var_estm_g=[]; Var_estd_g=[]; Corpoi_estm_g=[]; Corpoi_estd_g=[];
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'Var_estm','Var_estd','Corpoi_estm','Corpoi_estd','Times','Thlds','poi_idx','stdtimes','-v7.3');
end

%% plot

% figure settings
Tlab={'Initial','6 months','1 year','18 months','2 years','3 years','4 years'};

if exist([folder,f_name,'.mat'],'file')
    % saving figures
    fld_sav = './summary/figures/';
    if exist(fld_sav,'dir')~=7
        mkdir(fld_sav);
    end
    
%     [~,poi_idx]=get_roipoi; 
    poi=ceil(mean(cell2mat(reshape(poi_idx(1:4,:),[],1))));
    for gr=1:L_grp
        grp=groups{gr};
        
%         % plot variance
%         Var_gr=Var_estm(:,gr);
%         if ~isempty(Var_gr{1})
%             fig=figure((l-1)*L_grp+gr); clf(fig);
% %             set(fig,'pos',[0 800 800 600]);
% %             ha=tight_subplot(ceil(J/3),3,[.1,.07],[.08,.05],[.06,.04]);
%             set(fig,'pos',[0 800 1200 300]);
% %             ha=tight_subplot(1,J,[.1,.02],[.08,.125],[.02,.02]);
%             ha=tight_subplot(1,max_dur,[.1,.02],[.08,.125],[.02,.06]);
%             cmin=cellfun(@(x)min(nonzeros(x)),Var_gr,'UniformOutput',false);
%             cmax=cellfun(@(x)max(nonzeros(x)),Var_gr,'UniformOutput',false);
%             clim=[min(cell2mat(cmin)),max(cell2mat(cmax))];
%             
%             for j=1:max_dur
%                 h_sub=subplot(ha(j));
%                 if any(stdtimes{imax_dur}(j)==stdtimes{gr})
%                     im_j=Var_gr{j};
%     %                 imshow(im_j);
%                     imagesc(im_j,clim);
%                     set(gca,'xticklabel',[],'yticklabel',[]);
%     %                 title([grp, ' (t=', num2str(Times{gr}(j)),')'],'fontsize',20);
%                     title([grp, ' (',Tlab{j},')'],'fontsize',20);
%     %                 if gr==1
%     %                     title(Tlab{j},'fontsize',20);
%     %                 end
%                 else
%                     axis off;
%                 end
%             end
%             % add common colorbar
%             h_pos=h_sub.Position;
%             colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4)],'fontsize',14);
%             caxis(clim);
%             % save plot
%             fig.PaperPositionMode = 'auto';
%             print(fig,[fld_sav,grp,'_estvarft_',keywd{:}],'-dpng','-r0');
%         end
        
        % plot correlation of poi
        Corpoi_gr=Corpoi_estm(:,gr);
        if ~isempty(Corpoi_gr{1})
            fig=figure((L_mdl+l-1)*L_grp+gr); clf(fig);
            set(fig,'pos',[0 800 1200 300]);
            ha=tight_subplot(1,max_dur,[.1,.02],[.08,.125],[.02,.06]);
%             cmin=cellfun(@(x)min(nonzeros(x)),Corpoi_gr,'UniformOutput',false);
%             cmax=cellfun(@(x)max(nonzeros(x)),Corpoi_gr,'UniformOutput',false);
%             clim=[min(cell2mat(cmin)),max(cell2mat(cmax))];
            clim=[0,1];
            
%             poi_gr=poi_idx(:,gr);
            for j=1:max_dur
                h_sub=subplot(ha(j));
                if any(stdtimes{imax_dur}(j)==stdtimes{gr})
                    im_j=Corpoi_gr{j};
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
            % save plot
            fig.PaperPositionMode = 'auto';
            print(fig,[fld_sav,grp,'_',f_name],'-dpng','-r0');
        end
    end
end



end

% close parallel pools
delete(gcp('nocreate'));