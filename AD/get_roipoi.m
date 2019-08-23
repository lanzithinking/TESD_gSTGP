% get the region of interest (ROI) and the point(s) of interest (POI) of
% the brain images

function [roi_msk,poi_idx]=get_roipoi(pcnt,region,k,loc,PLOT)
if ~exist('pcnt','var') || isempty(pcnt)
    pcnt=.835;
end
if ~exist('region','var') || isempty(region)
    region='all';
end
if ~exist('k','var') || isempty(k)
    k=1;
end
if ~exist('loc','var') || isempty(loc)
    loc='./data/';
end
if ~exist('PLOT','var') || isempty(PLOT)
    PLOT=false;
end

% folder='./';
folder=loc;
keywd=strsplit(num2str(pcnt),'.'); keywd=['_p',keywd{end},region];
f_name=['roipoi',keywd];
if exist([folder,f_name,'.mat'],'file')
    load([folder,f_name,'.mat']);
    fprintf('%s loaded.\n',[f_name,'.mat']);
else
    % settings
    % for the data
    types={'PET','MRI'};
    typ=types{1};
    groups={'CN','MCI','AD'};
    L_grp=length(groups);
    % grp=groups{grp_opt};
    dur=[5,6,4];
    stdtimes={[0:.5:1,2:3]',[0:.5:1.5,2:3]',[0:.5:1,2]'};
    d=2;
    sec=48;

    % allocate saving
    [max_dur,imax_dur]=max(dur);
    roi_msk=cell(max_dur,L_grp); poi_idx=roi_msk;
    roi_stackt=cell(1,L_grp); poi_stackt=roi_stackt;

    for gr=1:L_grp
        grp=groups{gr};
        J=dur(gr);
        % obtain AD-PET data set
        [t,y]=read_data(typ,grp,J);
        % normalize time
        tt=datetime(t);
        tt=datenum(tt-repmat(tt(1,:),size(tt,1),1));
        t=tt./365;
        % remove irregular observations
        rmind=sum(abs(t-stdtimes{gr})>.55)>0;
        if any(rmind)
            fprintf('%d subject(s) removed!\n',sum(rmind));
        end
        % convert it to common time-frame % todo: extend the model to handle
        % different times
        t=mean(t(:,~rmind),2);
        % select one section and scale image intensity
        yy=cell2mat(shiftdim(y,-3));
        if d==2
            yy=squeeze(yy(:,:,sec,:,~rmind));
        end
        y=double(yy)./32767; yy=[];
        sz_y=size(y); imsz=sz_y(1:2);
        J=size(t,1);

        % average brain images over trials
        y_gr=mean(y,4);
        
        % plot if necessary
        if PLOT
            addpath('../util/tight_subplot/');
            fig=figure(gr); clf(fig);
            set(fig,'pos',[0 800 1200 300]);
            ha=tight_subplot(1,max_dur,[.1,.02],[.08,.125],[.02,.02]);
            Tlab={'Initial','6 months','1 year','18 months','2 years','3 years','4 years'};
        end

        for j=1:J
            jdx=find(stdtimes{imax_dur}==stdtimes{gr}(j));
            roi_msk{jdx,gr}=roi(y_gr(:,:,j),pcnt,region);
            y_j=y_gr(:,:,j);
            [maxk_val,maxk_id]=maxk(y_j(:),k);
            [id_i,id_j]=ind2sub(imsz,maxk_id);
            poi_idx{jdx,gr}=[id_i,id_j];
            if PLOT
                subplot(ha(jdx));
                imshow(full(roi_msk{jdx,gr})); hold on;
                plot(id_j,id_i,'rx','markersize',18,'linewidth',4);
                title([grp, ' (',Tlab{jdx},')'],'fontsize',20);
            end
            if contains(region,'stackt')
                if j==1
                    roi_stackt{gr}=roi_msk{jdx,gr};
                    poi_stackt{gr}=poi_idx{jdx,gr}; maxkval_stackt=maxk_val;
                else
                    roi_stackt{gr}=roi_stackt{gr}|roi_msk{jdx,gr};
                    [maxkval_stackt,maxkstk_id]=maxk([maxkval_stackt;maxk_val],k);
                    poi_stackt{gr}=[poi_stackt{gr};poi_idx{jdx,gr}];
                    poi_stackt{gr}=poi_stackt{gr}(maxkstk_id,:);
                end
            end
        end
    end
    if contains(region,'stackt')
        roi_msk=roi_stackt; poi_idx=poi_stackt;
    end

    % save
    if exist(folder,'dir')~=7
        mkdir(folder);
    end
    save([folder,f_name,'.mat'],'pcnt','k','roi_msk','poi_idx');
end

end