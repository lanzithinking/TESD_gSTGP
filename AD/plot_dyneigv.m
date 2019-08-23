% This is to plot dynamic eigenvalues

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
dur=[5,6,4];
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


%% estimation

% estimate the mean from MCMC samples
% load data
folder = './summary/';
files = dir(folder);
nfiles = length(files) - 2;
for l=2:L_mdl
intM=true;
keywd = {[alg_name,'_',repmat('intM_',intM),models{l}],['_L',num2str(L),'_d',num2str(d)]};
f_name = ['dyneigv_',keywd{:}];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
    fprintf('%s loaded.\n',[f_name,'.mat']);
else
    dyneigv_m=cell(1,L_grp); dyneigv_std=dyneigv_m; Times=dyneigv_m;
    for gr=1:L_grp
        grp=groups{gr}; J=dur(gr);
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
%             dyneigv_m{gr}=shiftdim(mean(samp_Lambda,1)); dyneigv_std{gr}=shiftdim(std(samp_Lambda,0,1));
            dyneigv_m{gr}=shiftdim(mean(abs(samp_Lambda),1)); dyneigv_std{gr}=shiftdim(std(abs(samp_Lambda),0,1));
            Times{gr}=t;
        end
    end
    % save the estimation results
    save([folder,f_name,'.mat'],'dyneigv_m','dyneigv_std','Times','stdtimes');
end


%% plot

% figure settings
Tlab={'Initial','6 months','1 year','18 months','2 years','3 years','4 years'};
[max_dur,imax_dur]=max(dur);

if exist([folder,f_name,'.mat'],'file')
    % figure setting
    fld_sav = './summary/figures/';
    if exist(fld_sav,'dir')~=7
        mkdir(fld_sav);
    end
    
    if ~isempty(dyneigv_m{1})
        fig=figure(1); clf(fig);
        set(fig,'pos',[0 800 1000 400]);
%         ha=tight_subplot(1,L_grp,[.07,.075],[.08,.08],[.07,.06]);
        ha=tight_subplot(1,L_grp,[.07,.05],[.15,.08],[.05,.06]);
        cmin=cellfun(@(x)min(x(:)),dyneigv_m);
        cmax=cellfun(@(x)max(x(:)),dyneigv_m);
        clim=[min(cmin),max(cmax)];

        for gr=1:L_grp
            grp=groups{gr};

            % plot dynamic eigenvalues
            im_g=dyneigv_m{gr};
            if ~isempty(im_g)
                J=size(im_g,1);
                h_sub=subplot(ha(gr));
%                 imagesc(im_g,clim);
                imagesc(im_g',clim);
                set(gca,'fontsize',14);
%                 yticks((1:J)); yticklabels(Tlab(ismember(stdtimes{imax_dur},stdtimes{gr})));
                xticks((1:J)); xticklabels(Tlab(ismember(stdtimes{imax_dur},stdtimes{gr}))); xtickangle(45);
                title(grp,'fontsize',20);
            end
        end
        % add common colorbar
        h_pos=h_sub.Position;
        colorbar('position',[sum(h_pos([1,3]))+.015 h_pos(2) 0.015 h_pos(4)],'fontsize',14);
        caxis(clim);
        
        % save plot
        fig.PaperPositionMode = 'auto';
        print(fig,[fld_sav,f_name],'-dpng','-r0');
    end
end

end