% This is to plot simulated process

% setting for simulation
N=[200,100]; % discretization sizes for space and time domains
K=1; % number of trials
d=1; % space dimension
seedNO=2018;
% load stationary data
[x,t,y{1}]=generate_data(N,K,d,true,seedNO);
% load non-stationary data
[x,t,y{2}]=generate_data(N,K,d,false,seedNO);
I=size(x,1); J=size(t,1);

% plot
addpath('../util/tight_subplot/');
% setting
fig=figure(1); clf(fig);
set(fig,'pos',[0 800 700 350]);
ha=tight_subplot(1,2,[0.1,.09],[.13,.08],[.07,.05]);
% get meshgrid
[x_gd,t_gd]=meshgrid(x,t); % (N_t, N_x)
% plot stationary case
subplot(ha(1));
imagesc(y{1}');
set(gca,'YDir','normal');
set(gca,'XTick',linspace(1,I,5),'XTickLabel',linspace(-1,1,5)); 
set(gca,'YTick',linspace(1,J,6),'YTickLabel',linspace(0,1,6));
set(gca,'fontsize',14);
xlabel('x','fontsize',16);
ylabel('t','fontsize',16,'rot',0,'pos',[-25,50,0]);
title('Stationary Process','fontsize',14);
% plot non-stationary case
subplot(ha(2));
imagesc(y{2}');
set(gca,'YDir','normal');
set(gca,'XTick',linspace(1,I,5),'XTickLabel',linspace(-1,1,5)); 
set(gca,'YTick',linspace(1,J,6),'YTickLabel',linspace(0,1,6));
set(gca,'fontsize',14);
xlabel('x','fontsize',16);
ylabel('t','fontsize',16,'rot',0,'pos',[-25,50,0]);
title('Non-Stationary Process','fontsize',14);
% save
fig.PaperPositionMode = 'auto';
print(fig,'./sim_STP','-dpng','-r0');