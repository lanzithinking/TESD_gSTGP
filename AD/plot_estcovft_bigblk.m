% This is to plot estimated variance/correlation function of time.
% The result is a series of dynamic connected images with correlation
% truncated above some threshold.

clear;
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
% % addpath(genpath('../util/boundedline-pkg/'));
if ~isdeployed
    addpath ~/STGP/code/util/;
    addpath ~/STGP/code/util/STGP/;
    addpath ~/STGP/code/util/ndSparse/;
    addpath ~/STGP/code/util/Image_Graphs/;
    addpath ~/STGP/code/util/tight_subplot/;
end
% Random Numbers...
seedNO = 2018;
seed = RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% correlation threshold
% cor_thld=0.1;
sps_den=1e-2;

% data settings
types={'PET','MRI'};
typ=types{1};
groups={'CN','MCI','AD'};
L_grp=length(groups);
% grp=groups{2};
dur=[5,6,4]; [max_dur,imax_dur]=max(dur);
stdtimes={[0:.5:1,2:3]',[0:.5:1.5,2:3]',[0:.5:1,2]'};
L=40;
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
kappa=2; % decaying rate for dynamic eigenvalues
% graph kernel
g.w=1;% g.size=imsz; g.mask=roi{mdl_opt};
jit_g=1e-6;
% temporal kernels
jit_t=1e-6;


%% estimation

% % set up parallel environment
% clst=parcluster('local');
% max_wkr=clst.NumWorkers;
% poolobj=gcp('nocreate');
% if isempty(poolobj)
%     poolobj=parpool(clst,min([2,max_wkr]));
% end

% estimate the mean from MCMC samplest
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
for l=2:L_mdl
intM=true; g.mask=roi{l};
keywd = {[alg_name,'_',repmat('intM_',intM),models{l}],['_L',num2str(L),'_d',num2str(d)]};
f_name = ['estcovft_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
else
    Var_estm=cell(max_dur,L_grp); Var_estd=Var_estm; Times=cell(1,L_grp);
    Cor_estm=cell(max_dur,L_grp); Cor_estd=Cor_estm; Thlds=Cor_estm;
    for gr=1:L_grp
        grp=groups{gr}; J=dur(gr);
        fprintf('Processing %s group...\n',grp);
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
            [Var_estm{:,gr}]=deal(0); [Var_estd{:,gr}]=deal(0);
            [Cor_estm{:,gr}]=deal(sparse(0)); [Cor_estd{:,gr}]=deal(sparse(0)); [Thlds{:,gr}]=deal(0);
            if ~exist('imsz','var')
                imsz=sqrt(size(y,1)).*ones(1,2);
            end
            g.size=imsz;
            N=prod(imsz); I=N; J=length(t);
            if exist('mgC','var')&&isa(mgC,[stgp_ver,'.mg'])
                stgp=mgC.stgp;
            else
                ker{1}=GL(g,optini.sigma2(1),exp(optini.eta(1)),s,L,jit_g,true);
                ker{2}=GP(t,optini.sigma2(2),exp(optini.eta(2)),s,L,jit_t,true);
                stgp=hd(ker{1},ker{2},optini.Lambda,kappa,l); [ker{1:2}]=deal([]);
            end
            N_samp=size(samp_sigma2,1); thin=10;
            prog=0.05:0.05:1; tic;
            for n=1:thin:N_samp
                stgp=stgp.update(stgp.C_x.update([],exp(samp_eta(n,1))),stgp.C_t.update(samp_sigma2(n,2),exp(samp_eta(n,2))),shiftdim(samp_Lambda(n,:,:)));
                for j=1:J
                    if l==2
                        PhiLambda2_j=stgp.C_x.eigf.*stgp.Lambda(j,:).^2;
                        var_j=sum(PhiLambda2_j.*stgp.C_x.eigf,2)+stgp.C_t.sigma2;
                        cor_j=tril(PhiLambda2_j*stgp.C_x.eigf',-1);
                        cor_j=sparse(cor_j)./sqrt(var_j)'./sqrt(var_j);
                    else
                        [~,sliceC]=stgp.mult(ndSparse(sparse((j-1)*I+(1:I),1:I,1,I*J,I)));
                        sliceC=sliceC((j-1)*I+(1:I),:);
                        var_j=sliceC(sub2ind([I,I],1:I,1:I));
                        cor_j=tril(sliceC,-1);
                        cor_j=sparse(cor_j)./sqrt(var_j)'./sqrt(var_j);
                    end
                    % sparsify the correlation
                    cor_thld=quantile(abs(nonzeros(cor_j)),1-sps_den);
%                     cor_thld=maxk(abs(nonzeros(cor_j)),min([nnz(cor_j),ceil(sps_den*numel(cor_j)/2)])); cor_thld=cor_thld(end);
                    sps_ind=abs(cor_j)>cor_thld;
                    cor_j=cor_j.*sps_ind;
                    Var_estm{j,gr}=Var_estm{j,gr}+var_j; Var_estd{j,gr}=Var_estd{j,gr}+var_j.^2;
                    Cor_estm{j,gr}=Cor_estm{j,gr}+cor_j; Cor_estd{j,gr}=Cor_estd{j,gr}+cor_j.^2;
                    Thlds{j,gr}=Thlds{j,gr}+cor_thld;
                end
                % display the progress
                iter=1+floor(n/thin);
                if ismember(iter,floor(N_samp/thin.*prog))
                    fprintf('%.0f%% completed.\n',100*iter/(N_samp/thin));
                end
            end
            time_gr=toc; fprintf('\n %.2f seconds used.\n', time_gr);
            for j=1:J
                Var_estm{j,gr}=Var_estm{j,gr}./(N_samp/thin); Var_estd{j,gr}=sqrt(Var_estd{j,gr}./(N_samp/thin) - Var_estm{j,gr}.^2);
                Var_estm{j,gr}=reshape(Var_estm{j,gr},imsz(1),imsz(2)); Var_estd{j,gr}=reshape(Var_estd{j,gr},imsz(1),imsz(2));
                Cor_estm{j,gr}=Cor_estm{j,gr}./(N_samp/thin); Cor_estd{j,gr}=sqrt(Cor_estd{j,gr}./(N_samp/thin) - Cor_estm{j,gr}.^2);
                if nnz(Cor_estm{j,gr})/numel(Cor_estm{j,gr})>sps_den
%                     cor_thld=maxk(abs(nonzeros(Cor_estm{j,gr})),min([nnz(Cor_estm{j,gr}),ceil(sps_den*numel(Cor_estm{j,gr})/2)])); cor_thld=cor_thld(end);
                    cor_thld=maxk(abs(nonzeros(Cor_estm{j,gr})),ceil(sps_den*numel(Cor_estm{j,gr})/2)); cor_thld=cor_thld(end);
                    sps_ind=abs(Cor_estm{j,gr})>=cor_thld;
                    Cor_estm{j,gr}=Cor_estm{j,gr}.*sps_ind; Cor_estd{j,gr}=Cor_estd{j,gr}.*sps_ind; 
                end
                Thlds{j,gr}=Thlds{j,gr}./(N_samp/thin);
            end
            Times{gr}=t;
            % temporary saving
            Var_estm_g=Var_estm(:,gr); Var_estd_g=Var_estd(:,gr); Cor_estm_g=Cor_estm(:,gr); Cor_estd_g=Cor_estd(:,gr);
            Times_g=Times(gr); Thlds_g=Thlds(:,gr);
            save([folder,grp,'_',f_name,'.mat'],'Var_estm_g','Var_estd_g','Cor_estm_g','Cor_estd_g','Times_g','Thlds_g','stdtimes','-v7.3');
            Var_estm_g=[]; Var_estd_g=[]; Cor_estm_g=[]; Cor_estd_g=[];
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'Var_estm','Var_estd','Cor_estm','Cor_estd','Times','Thlds','stdtimes','-v7.3');
end

%% plot

% figure settings
Tlab={'Initial','6 months','1 year','18 months','2 years','3 years','4 years'};

if exist([folder,f_name,'.mat'],'file')
    for gr=1:L_grp
        grp=groups{gr};
        % plot variance
%         Var_gr=Var_estm{gr};%.*32767;
%         clim=[min(Var_gr(:)),max(Var_gr(:))];
        Var_gr=Var_estm(:,gr);
        if ~isempty(Var_gr{1})
%             J=size(Var_gr,3);
%             J=length(Var_gr);
            fig=figure((l-1)*L_grp+gr); clf(fig);
%             set(fig,'pos',[0 800 800 600]);
%             ha=tight_subplot(ceil(J/3),3,[.1,.07],[.08,.05],[.06,.04]);
            set(fig,'pos',[0 800 1200 300]);
%             ha=tight_subplot(1,J,[.1,.02],[.08,.125],[.02,.02]);
            ha=tight_subplot(1,max_dur,[.1,.02],[.08,.125],[.02,.02]);
            
            for j=1:max_dur
                subplot(ha(j));
                jdx=find(stdtimes{imax_dur}(j)==stdtimes{gr});
                if ~isempty(jdx)
    %                 im_j=Var_gr(:,:,jdx);
                    im_j=Var_gr{jdx};
    %                 imshow(im_j);
                    imagesc(im_j);
                    set(gca,'xticklabel',[],'yticklabel',[]);
    %                 title([grp, ' (t=', num2str(Times{gr}(j)),')'],'fontsize',20);
                    title([grp, ' (',Tlab{j},')'],'fontsize',20);
    %                 if gr==1
    %                     title(Tlab{j},'fontsize',20);
    %                 end
                else
                    axis off;
                end
            end
            % save plot
            fig.PaperPositionMode = 'auto';
            fld_sav = './summary/figures/';
            if exist(fld_sav,'dir')~=7
                mkdir(fld_sav);
            end
            print(fig,[fld_sav,grp,'_estvarft_',keywd{:}],'-dpng','-r0');
        end
        % plot correlation (connected image)
        if ~isempty(Var_gr{1})
            fig=figure((L_mdl+l-1)*L_grp+gr); clf(fig);
            set(fig,'pos',[0 800 1200 300]);
            ha=tight_subplot(1,max_dur,[.1,.02],[.08,.125],[.02,.02]);
            
            for j=1:max_dur
                subplot(ha(j));
                jdx=find(stdtimes{imax_dur}(j)==stdtimes{gr});
                if ~isempty(jdx)
                    % obtain the connected image
                    [Idx,Idy]=find(Cor_estm{jdx,gr});
                    [gh_x,gh_y]=ind2sub(imsz,[Idx;Idy]);
                    conn=sparse(gh_x,gh_y,1,imsz(1),imsz(2));

                    subplot(ha(j));
    %                 imshow(conn);
                    imagesc(conn);
                    set(gca,'xticklabel',[],'yticklabel',[]);
                    title([grp, ' (',Tlab{j},')'],'fontsize',20);
                else
                    axis off;
                end
            end
            % save plot
            fig.PaperPositionMode = 'auto';
            fld_sav = './summary/figures/';
            if exist(fld_sav,'dir')~=7
                mkdir(fld_sav);
            end
            print(fig,[fld_sav,grp,'_estcorft_',keywd{:}],'-dpng','-r0');
        end
    end
end



end

% close parallel pools
delete(gcp('nocreate'));