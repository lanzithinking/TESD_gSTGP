%%%% - nonseparable stationary STGP - %%%%
%%%%   Shiwei Lan                     %%%%
%%%%   SoMSS, ASU                     %%%%
%%%%   slan@asu.edu                   %%%% 
%%%%   lanzithinking@gmail.com        %%%%
%%%%   Copyright @ 2021               %%%%
%%%% - nonseparable stationary STGP - %%%%

classdef nonsepstat
    % nonseparable stationary covariance functions for space-time data
    % -- by Tilmann Gneiting, JASA, Jun 2002, 97(458), pp590-600
    % C(h;u) = sigma^2/(|u|+1) * exp(-.5(|h|/l)^s/(|u|+1)))
    properties
        C_x; C_t; % spatial/temporal kernels (left for code consistence)
        Lambda; % time-varying eigenvalues in K-L expansion for C_x|t (left for code consistence)
        kappa=2; % decaying rate for dynamic eigenvalues (left for code consistence)
        x; t; % spatial/temporal inputs
        sigma2=1; % magnitude of the kernel
        l=.5; % correlation length of the kernel
        s=1; % smoothness of the kernel
        I,J; % spatial/temporal dimensions
        d,N,L; % input space dimensions, number of inputs, Karhunen-Loeve truncation number
        jtkers={'sep','kron_prod','kron_sum'}; % joint kernel models (left for code consistence)
        opt=0; % joint kernel model choice (left for code consistence)
        ker_opt='sep'; % joint kernel (left for code consistence)
        jit=1e-6; % jitter added to joint kernel
        eigf,eigv; % partial (L) eigenpair of the joint kernel
        store_eig=false; % indicator whether to store eigenpair
        spdapx=false; % use speed up (e.g. parfor) or approximation; number of workers when it is numeric
    end
    methods
        function self=nonsepstat(x,t,sigma2,l,s,L,jit,opt,store_eig,spdapx)
            % constructor
            % initialization
           if exist('x','var') && ~isempty(x)
                self.x=x;
           end
            if exist('t','var') && ~isempty(t)
                self.t=t;
            end
            if exist('sigma2','var') && ~isempty(sigma2)
                self.sigma2=sigma2;
            end
            if exist('l','var') && ~isempty(l)
                self.l=l;
            end
            if exist('s','var') && ~isempty(s)
                self.s=s;
            end
            [self.I,self.d]=size(self.x);
            self.J=length(self.t);
            self.N=self.I*self.J;
            if exist('L','var') && ~isempty(L)
                self.L=L;
            else
                self.L=min([self.N,100]);
            end
            if self.L>self.N
                warning('Karhunen-Loeve truncation number cannot exceed the dimension of discrete basis!');
                self.L=self.N;
            end
            if exist('jit','var') && ~isempty(jit)
                self.jit=jit;
            end
            if exist('opt','var') && ~isempty(opt)
                if isnumeric(opt)
                    self.opt=opt;
                elseif ischar(opt)
                    self.opt=0:length(self.jtkers)-1;
                    self.opt=self.opt(contains(self.jtkers,opt));
                else
                    error('Wrong option!');
                end
            end
            self.ker_opt=self.jtkers{self.opt+1};
            if exist('store_eig','var') && ~isempty(store_eig)
                self.store_eig=store_eig;
            end
            if self.store_eig
                % obtain partial eigen-basis
                [self.eigf,self.eigv]=self.eigs;
            end
            if exist('spdapx','var') && ~isempty(spdapx)
                self.spdapx=spdapx;
            elseif self.N>1e3
                self.spdapx=true;
            end
            if isnumeric(self.spdapx) && self.spdapx>1
                clst = parcluster('local');
                max_wkr= clst.NumWorkers;
                poolobj=gcp('nocreate');
                if isempty(poolobj)
                    poolobj=parpool('local',min([self.spdapx,max_wkr]));
                end
            end
            % tweek for code consistence
            self.C_x.update=@(sigma2,l)l;
            self.C_t.update=@(sigma2,l)sigma2;
            self.C_t.matn0pdf=@(varargin)0;
            self.C_t.solve=@(varargin)0;
            self.Lambda=zeros(self.J,self.L);
        end
        
        function [C_xtv,C_zv]=mult(self,v,alpha,beta)
            % joint kernel apply (mulitply) a function (vector)
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1; % -1: inverse; 0.5 square-root
            end
            if ~exist('beta','var') || isempty(beta)
                beta=0; % coefficient before likelihood component
            end
            if size(v,1)~=self.N
                v=reshape(v,self.N,[]); % (IJ,K_)
            end
            if ~self.spdapx
                if nargout==1
                    C_xt=self.tomat(alpha,beta,self.opt==1);
                    C_xtv=C_xt*v;
                elseif nargout>1
                    [C_xt,C_z]=self.tomat(alpha,beta,self.opt==1);
                    C_xtv=C_xt*v; C_zv=C_z*v;
                end
            else
                if ~isempty(gcp('nocreate'))
                    C_xtv=cell(self.I,1);
                    parfor i=1:self.I
                        blk_i=exp(-.5.*kron(1./(abs(self.t-self.t')+1),(pdist2(self.x(i,:),self.x,'minkowski',self.s)./self.l).^self.s)).*kron(1./(abs(self.t-self.t')+1),ones(1,self.I)); % (J,IJ)
%                         blk_i(:,(i-1)*self.J+1:i*self.J)=blk_i(:,(i-1)*self.J+1:i*self.J)+self.jit.*speye(self.J);
                        C_xtv{i}=blk_i*v;
                    end
                    C_xtv=cell2mat(C_xtv);
                else
                    C_xtv=zeros(size(v));
                    for i=1:self.I
                        blk_i=exp(-.5.*kron(1./(abs(self.t-self.t')+1),(pdist2(self.x(i,:),self.x,'minkowski',self.s)./self.l).^self.s)).*kron(1./(abs(self.t-self.t')+1),ones(1,self.I));
%                         blk_i(:,(i-1)*self.J+1:i*self.J)=blk_i(:,(i-1)*self.J+1:i*self.J)+self.jit.*speye(self.J);
                        C_xtv((i-1)*self.J+1:i*self.J,:)=blk_i*v;
                    end
                end
                C_xtv=C_xtv.*self.sigma2; % (IJ,K_)
                if nargout>1
                    C_zv=C_xtv+((alpha>=0).*self.jit+beta).*v; % (IJ,K_)
                end
            end
        end
        
        function bkdgix=get_bkdgix(self,ord)
            % obtain index of block diagonal matrix (IxI)*J
            if ~exist('ord','var')
                ord='IIJ';
            end
            [i_dgix,j_dgix]=find(kron(speye(self.J),ones(self.I)));
            if ~strcmp(ord,'IIJ')
                perm=[strfind(ord,'I'),strfind(ord,'J')];
                i_dgix=permute(reshape(i_dgix,self.I,self.I,self.J),perm);
                j_dgix=permute(reshape(j_dgix,self.I,self.I,self.J),perm);
            end
            bkdgix=[i_dgix(:),j_dgix(:)];
        end
        
        function [C_xt,C_z]=tomat(self,alpha,beta,trtdeg)
            % return the joint kernel in matrix format
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1; % -1: inverse; 0.5 square-root
            end
            if ~exist('beta','var') || isempty(beta)
                beta=0; % coefficient before likelihood component
            end
            if ~exist('trtdeg','var') || isempty(trtdeg)
                trtdeg=false;
            end
            C_xt=exp(-.5.*kron(1./(abs(self.t-self.t')+1),(pdist2(self.x,self.x,'minkowski',self.s)./self.l).^self.s)).*kron(1./(abs(self.t-self.t')+1),ones(self.I));
            C_xt=C_xt.*self.sigma2;
            if self.spdapx && ~issparse(C_xt)
                warning('Possible memory overflow!');
            end
            if nargout>1
                C_z=C_xt+((alpha>=0).*self.jit+beta).*speye(self.N);
            end
        end
        
        function C_zv=C_zmult(self,v)
                [~,C_zv]=self.mult(v);
        end
        
        function invCv=solve(self,v,varargin)
            % joint kernel solve a function (vector)
            if size(v,1)~=self.N
                v=reshape(v,self.N,[]);
            end
            if ~self.spdapx
                [~,C_z]=self.tomat([],[],true);
                invCv=C_z\v;
            else
                invCv=zeros(size(v));
                if ~isempty(gcp('nocreate'))
                    parfor k=1:size(invCv,2)
                        [invCv(:,k),flag]=pcg(@self.C_zmult,v(:,k),varargin{:}); % invC_zv
                    end
                else
                    for k=1:size(invCv,2)
                        [invCv(:,k),flag]=pcg(@self.C_zmult,v(:,k),varargin{:}); % invC_zv
                    end
                end
            end
        end
        
        function [eigf,eigv]=eigs(self,L,upd)
            % obtain partial eigen-basis of the joint kernel
            if ~exist('L','var') || isempty(L)
                L=self.L;
            end
            if ~exist('upd','var') || isempty(upd)
                upd=false; % force to recalculate in update
            end
            if ~(isempty(self.eigf) || isempty(self.eigv)) && L<=self.L && ~upd
                [eigf,eigv]=deal(self.eigf,self.eigv);
                eigf=eigf(:,1:L); eigv=eigv(1:L);
            else
                L=min([L,self.N]);
                if ~self.spdapx
                    [~,C_z]=self.tomat;
                    [eigf,eigv,flag]=eigs(C_z,L,'lm','Tolerance',1e-10,'MaxIterations',100);
                else
                    [eigf,eigv,flag]=eigs(@self.C_zmult,self.N,L,'lm','Tolerance',1e-10,'MaxIterations',100,'IsFunctionSymmetric',true); % C_z, (IJ,L)
                end
                eigv=diag(eigv);
                if flag
                    divrg_ind=isnan(eigv);
                    eigv(divrg_ind)=0;
                    warning('%d of %d requested eigenvalues are not converged!',sum(divrg_ind),L);
                end
            end
        end
        
        function y=act(self,x,alpha)
            % obtain the action of the joint kernel
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            if ~isnumeric(alpha) || numel(alpha) ~= 1
                error('Wrong exponent of action!');
            end
            switch alpha
                case 1
                    y=self.C_zmult(x); % C_z x
                case -1
                    y=self.solve(x);
                otherwise
                    if size(x,1)~=self.N
                        x=reshape(x,self.N,[]); % (IJ,K_)
                    end
                    [eigf,eigv]=self.eigs;
                    y=eigf*(((alpha<0).*self.jit+eigv).^alpha.*(eigf'*x)); % C_z^alpha x
            end
        end
        
        function ldet=logdet(self)
            % log-determinant of the joint kernel
            [~,eigv]=self.eigs;
            ldet=sum(log(abs(eigv))); % log |C_z|
        end
        
        function [logpdf,half_ldet]=matn0pdf(self,X,nu)
            % logpdf of centered matrix normal distribution X ~ MN(0,C,nuI)
            % for dynamic spatial kernel C_xt in model II
            if ~exist('nu','var') || isempty(nu)
                nu=1;
            end
            if size(X,1)~=self.N
                X=reshape(X,self.N,[]);
            end
            if ~self.spdapx
                half_ldet=-size(X,2).*self.logdet./2;
                half_quad=X.*self.solve(X); % qaud
            else
                [eigf,eigv]=self.eigs;
                rteigv=sqrt(abs(eigv)+self.jit);
                half_ldet=-size(X,2).*sum(log(rteigv));
                half_quad=(eigf'*X)./rteigv;
            end
            quad=-.5*sum(half_quad(:).^(2-~self.spdapx))./nu;
            logpdf=half_ldet+quad;
        end
        
        function self=update(self,l,sigma2,dummy)
            % update the eigen-basis
            if exist('sigma2','var') && ~isempty(sigma2)
                sigma2_=self.sigma2;
                self.sigma2=sigma2;
                if self.store_eig
                    self.eigv=self.eigv./sigma2_.*self.sigma2;
                end
            end
            if exist('l','var') && ~isempty(l)
                self.l=l;
                if self.store_eig
                    [self.eigf,self.eigv]=self.eigs([],true);
                end
            end
            if exist('dummy','var') && ~isempty(dummy)
                dummy=[]; % (left for code consistence)
            end
        end
        
        function U=scale_Lambda(self,Lambda,opt)
            % scale Lambda with the decaying rate
            if ~exist('Lambda','var') || isempty(Lambda)
                Lambda=self.Lambda; L=self.L;
            else
                L=size(Lambda,2);
            end
            if ~exist('opt','var') || isempty(opt)
                opt='up';
            end
            if contains(opt,'up')
                alpha=1;
            elseif contains(opt,{'down','dn'})
                alpha=-1;
            else
                alpha=0;
            end
            if isnumeric(self.kappa)
                gamma=(1:L).^(-self.kappa/2); % induce Cauchy sequence only if kappa>1
            elseif contains(self.kappa,'eigCx')
                [~,gamma]=self.C_x.eigs(L); % eigenvalues of C_0x
                gamma=sqrt(abs(gamma))';
            else
                gamma=ones(1,L);
            end
            U=Lambda./gamma.^alpha;
        end
        
        function M=sample_priM(self,MU)
            % sample prior mean function (matrix normal)
            mvn0Irv=randn(self.I,self.J); % (I,J)
            M0=self.act(mvn0Irv(:),.5); % (IJ,1)
            M0=reshape(M0,self.I,self.J); % (I,J)
            if ~exist('MU','var') || isempty(MU)
                M=M0;
            else
                M=MU+M0; % (I,J)
            end
        end
        
    end
end