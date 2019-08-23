% This is to plot joint covariance kernels

addpath('~/Documents/MATLAB/tight_subplot/');
% Random Numbers...
seed = RandStream('mt19937ar','Seed',2018);
RandStream.setGlobalStream(seed);

% parameters setting
N=[200,100]; % discretization sizes for space and time domains
x=linspace(-1,1,N(1)+1)'; x=x(2:end);
t=linspace(0,1,N(2)+1)'; t=t(2:end);
l_x=0.5; l_t=0.3; l_xt=sqrt(l_x*l_t);
sigma2_n=1e-2; % noise variance

% thin the mesh
thin=[20,10];
x=x(1:thin(1):end,:); t=t(1:thin(2):end);
I=length(x); J=length(t);

% form spatial and temporal covariance kernels respectively
C_x=exp(-pdist2(x,x,'squaredeuclidean')./(2*l_x));
C_t=exp(-(t-t').^2./(2*l_t));
% sample from eigen-values from another GP
L=20; %5;
Lambda=mvnrnd(zeros(1,size(C_t,1)),C_t,L)'; %(J,L)

% joint kernel options
jtkerns={'separate','kron_prod','kron_sum'};
L_jtk=length(jtkerns);

% plot conditional covariances
fig=figure(1); clf(fig);
% set(fig,'pos',[0 800 800 400]);
% ha=tight_subplot(1,L_jtk,[0.13,.06],[.08,.1],[.05,.09]);
set(fig,'pos',[0 800 1000 350]);
ha=tight_subplot(1,L_jtk,[.05,.03],[.1,.1],[.03,.07]);
lty={'-','--','-.',':'};
xq=[0:I:(I-1)*J;I:I:I*J]+.5; %xq(1)=1; 
rep2=ones(2,1); blkdiag_lines=cell(1,4);
blkdiag_lines{1}=[kron(xq(1,:)',rep2),xq(:)];
blkdiag_lines{2}=[kron(xq(2,:)',rep2),xq(:)];
blkdiag_lines{3}=[xq(:),kron(xq(1,:)',rep2)];
blkdiag_lines{4}=[xq(:),kron(xq(2,:)',rep2)];

% plot joint covariance kernerl Cov(z)
for i=1:L_jtk
    subplot(ha(i));
    if i==1
        C_z=kron(C_t,C_x)+1e-6.*speye(I*J);
%         C_z=kron(C_t+1e-6.*speye(J),C_x+1e-6.*speye(I));
    else
        C_z=STkern(C_x,C_t,Lambda,jtkerns{i});
    end
    imagesc(C_z); hold on;
%     spy(C_z);
    if i==1
        c_lims=caxis;
    else
        cl=caxis;
        c_lims(1)=min(c_lims(1),cl(1)); c_lims(2)=max(c_lims(2),cl(2));
    end
    set(gca,'YDir','normal');
    set(gca,'XTick',[],'XTickLabel',[]); 
    set(gca,'YTick',[],'YTickLabel',[]);
    set(gca,'fontsize',14);
    xlabel('z','fontsize',16);
    y_lbl=ylabel('z''','fontsize',16,'rot',0);
    set(y_lbl,'position',get(y_lbl,'position') - [1 0 0]);
    if i==1
        title('Model 0','fontsize',18);
    else
        title(['Model ',repmat('I',1,i-1)],'fontsize',18);
    end
    % add time-direction blocks
    for j=1:4
        plot(blkdiag_lines{j}(:,1),blkdiag_lines{j}(:,2),'color','red','linewidth',2); hold on;
    end
end
% adjust the color climits
for i=1:L_jtk
    h_sub=subplot(ha(i));
    caxis(c_lims);
end
% add colorbar
h_pos=h_sub.Position;
colorbar('position',[sum(h_pos([1,3]))+.02 h_pos(2) 0.02 h_pos(4)],'linewidth',.1);
% save plot
fig.PaperPositionMode = 'auto';
print(fig,'./figures/sim_jtkern','-dpng','-r0');
