% This is to plot random covariances

clear
sufx={'','_mtimesx','_gpu'};
stgp_ver=['STGP',sufx{2}];
addpath('../util/','../sampler/');
% addpath(['../util/+',stgp_ver,'/']);
if contains(stgp_ver,'mtimesx')
    addpath('../util/mtimesx/');
end
% Random Numbers...
seedNO=2018;
seed=RandStream('mt19937ar','Seed',seedNO);
RandStream.setGlobalStream(seed);

% model options
models={'sep','kron_prod','kron_sum','nonsepstat'};
L_mdl=length(models);
hold_out=false;

% setting for simulation
N=[200,100]; % discretization sizes for space and time domains
K=1; % number of trials
d=1; % space dimension
% load data
[x,t,~]=generate_data(N,K,d,true,seedNO);
% thin the mesh
thin=[20,10];
x=x(1:thin(1):end,:); t=t(1:thin(2):end);
I=size(x,1); J=size(t,1);
L=min([I,J]);

% parameters of kernel
s=2; % smoothness
kappa=1.2; % decaying rate for dynamic eigenvalues
% kappa=.2; % decaying rate for dynamic eigenvalues
% spatial kernel
jit_x=1e-6;
ker{1}=feval([stgp_ver,'.GP'],x,[],[],s,L,jit_x,true);
% temporal kernels
jit_t=1e-6;
ker{2}=feval([stgp_ver,'.GP'],t,[],[],s,L,jit_t,true);
% for hyper-GP
ker{3}=ker{2};

% plot
addpath('../util/tight_subplot/');
fig=figure(1); clf(fig);
set(fig,'pos',[0 800 900 600]);
ha=tight_subplot(2,3,[.05,.03],[.05,.05],[.04,.05]);

for mdl_opt=1:2
% specify (hyper)-priors
% (a,b) in inv-gamma priors for sigma2_*, * = eps, t, u
% (m,V) in (log) normal priors for eta_*, (eta=log-rho), * = x, t, u
switch mdl_opt
    case {0,1}
        a=ones(1,3); b=[5e0,1e1,1e1];
        m=zeros(1,3); V=[1e-1,1e-1,1e-2];
    case 2
        if ~hold_out
            a=ones(1,3); b=[1e-1,1e0,5e0];
            m=zeros(1,3); V=[1,1,abs(log10(K)-1)]; % K=100: V=[1,1,1]; K=1000: V=[1,1,2];
        else
            a=ones(1,3); b=[1e-1,1e0,1e1];
            m=zeros(1,3); V=[1,1,log10(K)+1]; % K=100: V=[1,1,3]; K=1000: V=[1,1,4];
        end
end
for j=1:3
subplot(ha(j+(mdl_opt-1)*3));
% initializatioin
% sigma2=1./gamrnd(a,1./b); 
sigma2=invgamrnd(a,b);
sigma2(1)=sigma2(1).^(mdl_opt~=2);
eta=normrnd(m,sqrt(V));
for k=1:length(ker)
    ker{k}=ker{k}.update(sigma2(k)^(k~=1),exp(eta(k)));
end
if isnumeric(kappa)
    gamma=(1:L).^(-kappa/2); % induce Cauchy sequence only if kappa>1
elseif contains(kappa,'eigCx')
    [~,gamma]=ker{1}.eigs(L); % eigenvalues of C_x
    gamma=sqrt(gamma');
else
    gamma=ones(1,L);
end
Lambda=ker{3}.rnd([],L).*gamma;
stgp=feval([stgp_ver,'.hd'],ker{1},ker{2},Lambda,kappa,mdl_opt,[],[]);


[~,C_z]=stgp.tomat;
imagesc(C_z); hold on;
if mdl_opt==1 && j==1
    c_lims=caxis;
else
    cl=caxis;
    c_lims(1)=min(c_lims(1),cl(1)); c_lims(2)=max(c_lims(2),cl(2));
end
set(gca,'YDir','normal');
set(gca,'XTick',[],'XTickLabel',[]); 
set(gca,'YTick',[],'YTickLabel',[]);
set(gca,'fontsize',14);
if mdl_opt==2
    xlabel('z','fontsize',16);
end
if j==1
    y_lbl=ylabel('z''','fontsize',16,'rot',0);
    set(y_lbl,'position',get(y_lbl,'position') - [2 0 0]);
end
title(['Model ',join([repmat('0',1,mdl_opt==0),repmat('I',1,mdl_opt)])]);
end
end

% % adjust the color climits
% for mdl_opt=1:2
% for j=1:3
%     h_sub=subplot(ha(j+(mdl_opt-1)*3));
%     caxis(c_lims);
% end
% end
% % add colorbar
% h_pos=h_sub.Position;
% colorbar('position',[sum(h_pos([1,3]))+.02 h_pos(2) 0.02 h_pos(4)],'linewidth',.1);
% save plot
fig.PaperPositionMode = 'auto';
print(fig,'./randcov','-dpng','-r0');
