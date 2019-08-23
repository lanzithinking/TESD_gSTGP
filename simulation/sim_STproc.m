% This is to generate simulated data of multiple spatio-temporal processes

function [x,t,y,L_y]=sim_STproc(N,K,d,seedNO,SAVE,PLOT,d_plt)
if ~exist('N','var')
    N=[200,100]; % discretization sizes for space and time domains
end
if ~exist('K','var')
    K=1; % number of trials
end
if ~exist('d','var')
    d=1; % spatial dimensions
end
if ~exist('seed','var')
    seedNO=2018;
end
if ~exist('SAVE','var')
    SAVE=false;
end
if ~exist('PLOT','var')
    PLOT=false;
end
if ~exist('d_plt','var')
    d_plt=1; % spatial dimension to plot
end

% Random Numbers...
seed = RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% parameters setting
x=linspace(-1,1,N(1)+1)'; %x=x(2:end);
x=repmat(x,1,d); % (I,d)
t=linspace(0,1,N(2)+1)'; %t=t(2:end); % (J,1)
I=size(x,1); J=size(t,1);
l_x=0.5; l_t=0.3; l_xt=sqrt(l_x*l_t);
sigma2_n=1e-2; % noise variance

% form mean function
m=sum(cos(pi.*x),2).*sin(pi.*t');

% form covariance kernel
C_x=exp(-pdist2(x,x,'squaredeuclidean')./(2*l_x)); C_t=exp(-(t-t').^2./(2*l_t));
xt=reshape(x,I,1,[]).*t'; xt=reshape(xt,I*J,[]); C_xt=exp(-pdist2(xt,xt,'minkowski',1)./(2*l_xt));
C_y=kron(C_t,C_x).*C_xt+sigma2_n.*speye(I*J);

% generate data
% y=mvnrnd(zeros(1,I*J),C_z);
L_y=chol(C_y,'lower');
y=L_y*randn(I*J,K);
y=reshape(y,[I,J,K])+m; % (I,J,K)

% file name
f_name=['sim_STproc_I',num2str(I),'_J',num2str(J),'_K',num2str(K),'_d',num2str(d)];
% save data to file
if SAVE
    folder = './data/';
    if exist(folder,'dir')~=7
        mkdir(folder);
    end
    save([folder,f_name,'.mat'],'N','K','d','seedNO','x','t','y');
end

% plot data
if PLOT
    addpath('~/Documents/MATLAB/tight_subplot/');
    % setting
    fig=figure(1); clf(fig);
    set(fig,'pos',[0 800 700 350]);
    ha=tight_subplot(1,2,[0.1,.09],[.13,.08],[.07,.05]);
    % get meshgrid
    [x_gd,t_gd]=meshgrid(x(:,d_plt),t); % (N_t, N_x)
    y_gd=y(:,:,1)'; % plot the first trial
    % plot data points in 3d
    subplot(ha(1));
    scatter3(x_gd(:),t_gd(:),y_gd(:),10,y_gd(:),'filled');
    set(gca,'fontsize',14);
    x_lbl='x';
    if d>1
        x_lbl=[x_lbl,'_',num2str(d_plt)];
    end
    xlabel(x_lbl,'fontsize',16,'rot',18,'pos',[0,-0.2,-2]);
    ylabel('t','fontsize',16,'rot',-60,'pos',[-1.6,0.4,-2]);
    zlabel('y','fontsize',16,'rot',0,'pos',[-1.35,1.05,2]);
    % plot data points in 2d field
    subplot(ha(2));
%     surf(x_gd,t_gd,y_gd);
%     view(0,90);
    imagesc(y_gd);
    set(gca,'YDir','normal');
    set(gca,'XTick',linspace(1,I,5),'XTickLabel',linspace(-1,1,5)); 
    set(gca,'YTick',linspace(1,J,6),'YTickLabel',linspace(0,1,6));
    set(gca,'fontsize',14);
    xlabel(x_lbl,'fontsize',16);
%     ylabel('t','fontsize',16,'rot',0,'pos',[-1.25,0.5,-2]);
    ylabel('t','fontsize',16,'rot',0,'pos',[-25,50,0]);
    % save plot
    if SAVE
        fig.PaperPositionMode = 'auto';
%         folder = './figures/';
%         if exist(folder,'dir')~=7
%             mkdir(folder);
%         end
        print(fig,[folder,f_name],'-dpng','-r0');
    end
end

end