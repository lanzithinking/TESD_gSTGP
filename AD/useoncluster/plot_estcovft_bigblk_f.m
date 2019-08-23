% This is to plot estimated variance/correlation function of time.
% The result is a series of dynamic connected images with correlation
% truncated above some threshold.

function []=plot_estcovft_bigblk_f(grp_opt,mdl_opt,opthypr,jtupt,intM,use_para)
% sufx={'','_mtimesx','_gpu'};
% stgp_ver=['STGP',sufx{1}];
% addpath('../util/');
% % addpath(['../util/+',stgp_ver,'/']);
% if contains(stgp_ver,'mtimesx')
%     addpath('../util/mtimesx/');
% end
% addpath('../util/Image Graphs/');
% addpath('../util/ndSparse/');
% addpath('../util/tight_subplot/');
% addpath(genpath('../util/boundedline-pkg/'));
if ~isdeployed
    addpath ~/STGP/code/util/;
    addpath ~/STGP/code/util/STGP/;
    addpath ~/STGP/code/util/ndSparse/;
    addpath ~/STGP/code/util/Image_Graphs/;
    addpath ~/STGP/code/util/tight_subplot/;
end
if isdeployed
    grp_opt=str2num(grp_opt);
    mdl_opt=str2num(mdl_opt);
    opthypr=str2num(opthypr);
    jtupt=str2num(jtupt);
    intM=str2num(intM);
end
% Random Numbers...
seedNO = 2018;
seed = RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% settings
if ~exist('grp_opt','var') || isempty(grp_opt)
    grp_opt=2;
end
if ~exist('mdl_opt','var') || isempty(mdl_opt)
    mdl_opt=2;
end
if ~exist('opthypr','var') || isempty(opthypr)
    opthypr=false;
end
if ~exist('jtupt','var') || isempty(jtupt)
    jtupt=false;
end
if ~exist('intM','var') || isempty(intM)
    intM=(mdl_opt==1);
end
if ~exist('use_para','var') || isempty(use_para)
    use_para=false;
end

% large data storage path prefix
store_prefix='~/project-stat/STGP/code/AD/';
datloc=[store_prefix,'data/']; 

% correlation threshold
% cor_thld=0.1;
sps_den=.1;
% sps_den=1/160;

% data settings
% types={'PET','MRI'};
% typ=types{1};
groups={'CN','MCI','AD'};
% L_grp=length(groups);
% grp=groups{2};
dur=[5,6,4]; [max_dur,imax_dur]=max(dur); J=dur(grp_opt);
stdtimes={[0:.5:1,2:3]',[0:.5:1.5,2:3]',[0:.5:1,2]'};
L=100;
d=2;
% sec=48;
% model options
models={'kron_prod','kron_sum'};
% L_mdl=length(models);
% intM=false;
alg_name='MCMC';
if opthypr
    alg_name=['opt',alg_name];
    if jtupt
        alg_name=['jt',alg_name];
    end
end
% obtain ROI
roi=get_roipoi(.75,'stackt',[],datloc);

% parameters of kernel
s=2; % smoothness
kappa=0; % decaying rate for dynamic eigenvalues
% graph kernel
g.w=1;
% g.size=imsz;
g.mask=roi{grp_opt};
jit_g=1e-6;
% temporal kernels
jit_t=1e-6;


%% estimation

% set up parallel environment
if use_para
    clst=parcluster('local');
    max_wkr=clst.NumWorkers;
    poolobj=gcp('nocreate');
    if isempty(poolobj)
        poolobj=parpool(clst,min([2,max_wkr]));
    end
end

% processing results

grp=groups{grp_opt};
fprintf('Processing %s group...\n',grp);
    
% estimate the mean from MCMC samplest
% load data
% folder = './summary/';
folder = [store_prefix,'summary/'];
files = dir(folder);
nfiles = length(files) - 2;
keywd = {[alg_name,'_',repmat('intM_',intM),models{mdl_opt}],['_J',num2str(J)],['_L',num2str(L),'_d',num2str(d)]};
f_name = [grp,'_estcovft_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
    fprintf('%s loaded.\n',[f_name,'.mat']);
else
    Var_estm=cell(J,1); Var_estd=Var_estm;
    Cor_estm=cell(J,1); Cor_estd=Cor_estm; Thlds=Cor_estm;
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
        [Var_estm{:}]=deal(0); [Var_estd{:}]=deal(0);
        [Cor_estm{:}]=deal(sparse(0)); [Cor_estd{:}]=deal(sparse(0)); [Thlds{:}]=deal(0);
        if ~exist('imsz','var')
            imsz=sqrt(size(y,1)).*ones(1,2);
        end
        g.size=imsz;
        N=prod(imsz); I=N; J=length(t);
        % get mask
%         mask=cell(1,J);
%         for j=1:J
%             mask{j}=roi(reshape(samp_M(:,j,1),imsz));
% %             figure(j); imshow(full(mask{j}));
%         end
%         roi_msk=get_roipoi;
        roi_msk=get_roipoi([],[],[],datloc);
        mask=roi_msk(ismember(stdtimes{imax_dur},stdtimes{grp_opt}),grp_opt);
        if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
            stgp=mgC.stgp;
        else
            ker{1}=GL(g,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_g,true);
            ker{2}=GP(t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
            stgp=hd(ker{1},ker{2},optini.Lambda,kappa,mdl_opt); [ker{1:2}]=deal([]);
        end
        N_samp=size(samp_sigma2,1); thin=10;
        prog=0.05:0.05:1; tic;
        for n=1:thin:N_samp
            stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),shiftdim(samp_Lambda(n,:,:)));
            for j=1:J
                if mdl_opt==2
                    % impose ROI
                    Phi_x=mask{j}(:).*stgp.C_x.eigf;
                    PhiLambda2_j=Phi_x.*stgp.Lambda(j,:).^2;
                    var_j=sum(PhiLambda2_j.*Phi_x,2)+stgp.C_t.sigma2;
                    cor_j=tril(PhiLambda2_j*Phi_x',-1);
                    cor_j=cor_j./sqrt(var_j)'./sqrt(var_j);
                    % impose ROI
                    var_j=reshape(var_j,imsz).*mask{j};
                else
                    [~,sliceC]=stgp.mult(ndSparse(sparse((j-1)*I+(1:I),1:I,1,I*J,I)));
                    sliceC=sliceC((j-1)*I+(1:I),:);
                    var_j=sliceC(sub2ind([I,I],1:I,1:I));
                    cor_j=tril(sliceC,-1);
                    cor_j=sparse(cor_j)./sqrt(var_j)'./sqrt(var_j);
                    % impose ROI
                    var_j=reshape(var_j,imsz).*mask{j};
                    cor_j=cor_j.*mask{j}(:)'.*mask{j}(:);
                end
                % sparsify the correlation
                cor_thld=quantile(abs(nonzeros(cor_j)),1-sps_den);
%                 cor_thld=maxk(abs(nonzeros(cor_j)),min([nnz(cor_j),ceil(sps_den*numel(cor_j)/2)])); cor_thld=cor_thld(end);
                sps_ind=abs(cor_j)>cor_thld;
                cor_j=cor_j.*sps_ind;
                Var_estm{j}=Var_estm{j}+var_j; Var_estd{j}=Var_estd{j}+var_j.^2;
                Cor_estm{j}=Cor_estm{j}+cor_j; Cor_estd{j}=Cor_estd{j}+cor_j.^2;
                Thlds{j}=Thlds{j}+cor_thld;
            end
            % display the progress
            iter=1+floor(n/thin);
            if ismember(iter,floor(N_samp/thin.*prog))
                fprintf('%.0f%% completed.\n',100*iter/(N_samp/thin));
            end
        end
        time_gr=toc; fprintf('\n %.2f seconds used.\n', time_gr);
        for j=1:J
            Var_estm{j}=Var_estm{j}./(N_samp/thin); Var_estd{j}=sqrt(Var_estd{j}./(N_samp/thin) - Var_estm{j}.^2);
%             Var_estm{j}=reshape(Var_estm{j},imsz(1),imsz(2)); Var_estd{j}=reshape(Var_estd{j},imsz(1),imsz(2));
            Cor_estm{j}=Cor_estm{j}./(N_samp/thin); Cor_estd{j}=sqrt(Cor_estd{j}./(N_samp/thin) - Cor_estm{j}.^2);
%             if nnz(Cor_estm{j})/numel(Cor_estm{j})>sps_den
% %                 cor_thld=maxk(abs(nonzeros(Cor_estm{j})),min([nnz(Cor_estm{j}),ceil(sps_den*numel(Cor_estm{j})/2)])); cor_thld=cor_thld(end);
%                 cor_thld=maxk(abs(nonzeros(Cor_estm{j})),ceil(sps_den*numel(Cor_estm{j})/2)); cor_thld=cor_thld(end);
%                 sps_ind=abs(Cor_estm{j})>=cor_thld;
%                 Cor_estm{j}=Cor_estm{j}.*sps_ind; Cor_estd{j}=Cor_estd{j}.*sps_ind; 
%             end
            Thlds{j}=Thlds{j}./(N_samp/thin);
        end
        Times=t;
        % save the estimation results
        save([folder,f_name,'.mat'],'Var_estm','Var_estd','Cor_estm','Cor_estd','Times','imsz','sps_den','Thlds','mask','stdtimes','-v7.3');
    end
end

%% plot

% figure settings
Tlab={'Initial','6 months','1 year','18 months','2 years','3 years','4 years'};

if exist([folder,f_name,'.mat'],'file')
    grp=groups{grp_opt};
    % saving figures
    fld_sav =[folder,'figures/'];
    if exist(fld_sav,'dir')~=7
        mkdir(fld_sav);
    end
    
%     Var_estm=Var_estm_g; Cor_estm=Cor_estm_g;
    if ~isempty(Var_estm{1})
%         fig1=figure(1); clf(fig1);
        fig1=figure('visible','off');
%         set(fig,'pos',[0 800 800 600]);
%         ha=tight_subplot(ceil(J/3),3,[.1,.07],[.08,.05],[.06,.04]);
        set(fig1,'pos',[0 800 1200 300]);
%         ha=tight_subplot(1,J,[.1,.02],[.08,.125],[.02,.02]);
        ha=tight_subplot(1,max_dur,[.1,.02],[.08,.125],[.02,.06]);
        cmin=cellfun(@(x)min(nonzeros(x)),Var_estm,'UniformOutput',false);
        cmax=cellfun(@(x)max(nonzeros(x)),Var_estm,'UniformOutput',false);
        clim=[min(cell2mat(cmin)),max(cell2mat(cmax))];
        
%         J=length(Var_estm);
        % plot variance
        for j=1:max_dur
            h_sub=subplot(ha(j));
            jdx=find(stdtimes{imax_dur}(j)==stdtimes{grp_opt});
            if ~isempty(jdx)
                im_j=Var_estm{jdx};
    %             imshow(im_j);
                imagesc(im_j,clim);
                set(gca,'xticklabel',[],'yticklabel',[]);
    %             title([grp, ' (t=', num2str(Times(j)),')'],'fontsize',20);
                title([grp, ' (',Tlab{j},')'],'fontsize',20);
    %             if gr==1
    %                 title(Tlab{j},'fontsize',20);
    %             end
            else
                axis off;
            end
        end
        % add common colorbar
        h_pos=h_sub.Position;
        colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4)],'fontsize',14);
        caxis(clim);
        % save plot
        fig1.PaperPositionMode = 'auto';
        print(fig1,[fld_sav,grp,'_estvarft_',keywd{:}],'-dpng','-r0');
        
%         fig2=figure(2); clf(fig2);
        fig2=figure('visible','off');
        set(fig2,'pos',[0 800 1200 300]);
        ha=tight_subplot(1,max_dur,[.1,.02],[.08,.125],[.02,.06]);
        
        % plot correlation (connected image)
        if ~exist('imsz','var')
            imsz=sqrt(size(Cor_estm{1},1)).*ones(1,2);
        end
        clim=[+Inf,-Inf]; %sps_den=2e-3;
        for j=1:max_dur
            h_sub=subplot(ha(j));
            jdx=find(stdtimes{imax_dur}(j)==stdtimes{grp_opt});
            if ~isempty(jdx)
                cor_j=Cor_estm{jdx};
                % adjust the sparsity
%                 if nnz(cor_j)/numel(cor_j)>sps_den
%     %                 cor_thld=maxk(abs(nonzeros(cor_j)),min([nnz(cor_j),ceil(sps_den*numel(cor_j)/2)])); cor_thld=cor_thld(end);
%                     cor_thld=maxk(abs(nonzeros(cor_j)),ceil(sps_den*numel(cor_j)/2)); cor_thld=cor_thld(end);
%                     sps_ind=abs(cor_j)>=cor_thld;
%                     cor_j=cor_j.*sps_ind;
%                 end
                % obtain the connected image
                [Idx,Idy]=find(cor_j);
                [gh_x,gh_y]=ind2sub(imsz,[Idx;Idy]);
                conn=sparse(gh_x,gh_y,1,imsz(1),imsz(2));
                
    %             imshow(conn);
                imagesc(conn); hold on;
                set(gca,'xticklabel',[],'yticklabel',[]);
                title([grp, ' (',Tlab{j},')'],'fontsize',20);
                cmin_j=min(conn(:)); cmax_j=max(conn(:));
                if cmin_j<clim(1)
                    clim(1)=cmin_j;
                end
                if cmax_j>clim(2)
                    clim(2)=cmax_j;
                end
            else
                axis off;
            end
        end
        % adjust color scale afterwards
        for j=1:max_dur
            h_sub=subplot(ha(j));
            jdx=find(stdtimes{imax_dur}(j)==stdtimes{grp_opt});
            if ~isempty(jdx)
                caxis(clim);
            end
        end
        % add common colorbar
        h_pos=h_sub.Position;
        colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4)],'fontsize',14);
        caxis(clim);
        % save plot
        fig2.PaperPositionMode = 'auto';
        print(fig2,[fld_sav,grp,'_estcorft_',keywd{:}],'-dpng','-r0');
    end
end


% close parallel pools
delete(gcp('nocreate'));
end