%%%% - high-dimensional STGP - %%%%
%%%%   Shiwei Lan              %%%%
%%%%   STAT, UIUC              %%%%
%%%%   shiwei@illinois.edu     %%%% 
%%%%   lanzithinking@gmail.com %%%%
%%%%   Copyright @ 2018        %%%%
%%%% - high-dimensional STGP - %%%%

classdef hd
    % high-dimensional Spatio-Temporal Gaussian Process
    % -- given spatial and temopral kernels separately, construct a joint
    % kernel that model the temporal evolution of spatial dependence
    properties
        C_x; % spatial kernel
        C_t; % temporal kernel
        Lambda; % time-varying eigenvalues in K-L expansion for C_x|t
        kappa=2; % decaying rate for dynamic eigenvalues
        I,J,L; % spatial/temporal dimensions, Karhunen-Loeve truncation number
        N; % joint dimension (number of total inputs per trial)
        jtkers={'sep','kron_prod','kron_sum'}; % joint kernel models
        opt=2; % joint kernel model choice
        ker_opt; % joint kernel
        jit=1e-6; % jitter added to joint kernel
        eigf,eigv; % partial (L) eigenpair of the joint kernel
        store_eig=false; % indicator whether to store eigenpair
        spdapx=false; % use speed up (e.g. parfor) or approximation; number of workers when it is numeric
    end
    methods
        function self=hd(C_x,C_t,Lambda,kappa,opt,jit,store_eig,spdapx)
            % constructor
            % initialization
            self.C_x=C_x;
            self.C_t=C_t;
            if exist('Lambda','var') && ~isempty(Lambda)
                self.Lambda=Lambda;
            else
                self.Lambda=self.C_t.eigf;
            end
            self.I=self.C_x.N;
            self.J=self.C_t.N;
            self.N=self.I*self.J;
            if size(self.Lambda,1)~=self.J
                error('Size of Lambda does not match time-domain dimension!');
            end
            self.L=size(self.Lambda,2);
            if self.L>self.I
                warning('Karhunen-Loeve truncation number cannot exceed the dimension of discrete basis!');
                self.L=self.I; self.Lambda=self.Lambda(:,1:self.I);
            end
            if exist('kappa','var') && ~isempty(kappa)
                self.kappa=kappa;
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
            if exist('jit','var') && ~isempty(jit)
                self.jit=jit;
            end
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
        end
        
        function [C_xtv,C_zv]=mult(self,v,alpha,beta)
            % joint kernel apply (mulitply) a function (vector)
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1; % -1: inverse; 0.5 square-root
            end
            if ~exist('beta','var') || isempty(beta)
                beta=self.opt==2; % coefficient before likelihood component
            end
            if ~self.spdapx
                if size(v,1)~=self.N
                    v=reshape(v,self.N,[]); % (IJ,K_)
                end
                if nargout==1
                    C_xt=self.tomat(alpha,beta,self.opt==1);
                    C_xtv=C_xt*v;
                elseif nargout>1
                    [C_xt,C_z]=self.tomat(alpha,beta,self.opt==1);
                    C_xtv=C_xt*v; C_zv=C_z*v;
                end
            else
                if size(v,1)~=self.I
                    v=reshape(v,self.I,self.J,[]); % (I,J,K_)
                end
                if contains(self.ker_opt,'sep')
                    C_xtv=self.C_t.mult(self.C_x.mult(v),'t'); % (I,J,K_)
                    if nargout>1
                        C_zv=reshape(C_xtv+((alpha>=0).*self.jit+beta).*v,self.N,[]); % (IJ,K_)
                    end
                    C_xtv=reshape(C_xtv,self.N,[]); % (IJ,K_)
                else
                    Lambda_=(self.Lambda.^self.opt+self.jit.*(alpha<0)).^alpha; Phi_x=self.C_x.eigs(self.L);
                    switch self.ker_opt
                        case 'kron_prod'
        %                     LambdaPhiv=permute(reshape(repmat(Lambda_',1,size(v,3)).*(Phi_x'*v(:,:)),self.L,self.J,[]),[2,1,3]); % (J,L,K_)
                            LambdaPhiv=permute(Lambda_'.*mtimesx(Phi_x,'t',v),[2,1,3]); % (J,L,K_)
                            PhiLambda=reshape(Phi_x,self.I,1,self.L).*reshape(Lambda_,1,self.J,self.L); % (I,J,L)
                            C_xtv=reshape(PhiLambda,self.N,[])*shiftdim(sum(LambdaPhiv)); % (IJ,K_)
                            if nargout>1
                                C_zv=squeeze(sum(PhiLambda.*reshape(self.C_t.mult(LambdaPhiv(:,:)),1,self.J,self.L,[]),3)); % (I,J,K_)
                                C_zv=reshape(C_zv+((alpha>=0).*self.jit+beta).*v,self.N,[]); % (IJ,K_)
                            end
                        case 'kron_sum'
        %                     Lambda2Phiv=repmat(Lambda_',1,size(v,3)).*(Phi_x'*v(:,:)); % (L,JK_)
                            Lambda2Phiv=Lambda_'.*mtimesx(Phi_x,'t',v); % (L,J,K_)
                            C_xtv=reshape(Phi_x*Lambda2Phiv(:,:)+(alpha>=0).*self.jit.*v(:,:),self.N,[]); % (IJ,K_)
                            if nargout>1
                                if size(v,2)>self.J
                                    v=reshape(v,self.I,self.J,[]);
                                end
                                C_zv=self.C_t.mult(v,'t'); % (I,J,K_)
                                C_zv=reshape(C_zv,self.N,[]);
                                C_zv=C_zv+beta.*C_xtv; % (IJ,K_)
                            end
                    end
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
                beta=self.opt==2; % coefficient before likelihood component
            end
            if ~exist('trtdeg','var') || isempty(trtdeg)
                trtdeg=false;
            end
            if contains(self.ker_opt,'sep')
                C_xt=kron(self.C_t.tomat(),self.C_x.tomat());
                if nargout>1
                    C_z=C_xt+((alpha>=0).*self.jit+beta).*speye(self.N);
                end
            else
                Lambda_=(self.Lambda.^self.opt+self.jit.*(alpha<0)).^alpha; Phi_x=self.C_x.eigs(self.L);
                switch self.ker_opt
                    case 'kron_prod'
                        PhiLambda=reshape(Phi_x,self.I,1,self.L).*reshape(Lambda_,1,self.J,self.L);
                        PhiLambda=reshape(PhiLambda,self.N,self.L);
                        C_xt=PhiLambda*PhiLambda';
                        if trtdeg && self.L<self.I
                            C_x0=Phi_x*(self.C_x.eigv(1:self.L).*Phi_x');
                            C_xt=C_xt+repmat(self.C_x.tomat-C_x0,self.J,self.J);
                        end
                        if nargout>1
                            C_z=C_xt.*kron(self.C_t.tomat,ones(self.I))+((alpha>=0).*self.jit+beta).*speye(self.N);
                        end
                    case 'kron_sum'
                        PhiLambda2=reshape(Phi_x,self.I,1,self.L).*reshape(Lambda_,1,self.J,self.L);
                        PhiLambda2=reshape(PhiLambda2,self.N,self.L);
                        PhiLambda2=PhiLambda2*Phi_x'+repmat((alpha>=0).*self.jit.*speye(self.I),self.J,1);
                        if trtdeg && self.L<self.I
                            C_x0=Phi_x*(self.C_x.eigv(1:self.L).*Phi_x');
                            PhiLambda2=PhiLambda2+repmat(self.C_x.tomat-C_x0,self.J,1);
                        end
                        bkdgix=self.get_bkdgix('IJI');
                        C_xt=sparse(bkdgix(:,1),bkdgix(:,2),PhiLambda2(:));
                        if nargout>1
                            C_z=beta.*C_xt+kron(self.C_t.tomat,speye(self.I));
                        end
                end
            end
        end
        
        function C_zv=C_zmult(self,v)
                [~,C_zv]=self.mult(v);
        end
        
        function invCv=solve(self,v,varargin)
            % joint kernel solve a function (vector)
            switch self.ker_opt
                case 'sep'
                    if size(v,1)~=self.I
                        v=reshape(v,self.I,self.J,[]); % (I,J,K_)
                    end
                    invCv=self.C_t.solve(self.C_x.solve(v),'t'); % invC_xtv
                    invCv=reshape(invCv,self.N,[]);
                case 'kron_prod'
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
                case 'kron_sum'
                    invCv=self.mult(v,-1); % invC_xtv
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
                switch self.ker_opt
                    case 'sep'
                        [eigf_t,eigv_t]=self.C_t.eigs(); [eigf_x,eigv_x]=self.C_x.eigs();
                        eigv=kron(eigv_t,eigv_x); eigf=kron(eigf_t,eigf_x);
                        if L<=self.C_t.L*self.C_x.L
                            eigv=eigv(1:L); eigf=eigf(:,1:L);
                        else
                            warning('Requested too many eigenvalues!');
                        end
                    case 'kron_prod'
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
                    case 'kron_sum'
                        eigv=self.Lambda'.^2; %[eigv,I_dsc]=sort(eigv(:),'descend'); % (LJ, 1)
                        eigf=kron(speye(self.J),self.C_x.eigs(self.L)); %eigf=eigf(:,I_dsc); % (IJ, LJ)
                        if L<=self.L*self.J
                            eigv=eigv(1:L); eigf=eigf(:,1:L);
                        else
                            warning('Requested too many eigenvalues!');
                        end
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
                    switch self.ker_opt
                        case {'sep','kron_prod'}
                            y=self.C_zmult(x); % C_z x
                        case 'kron_sum'
                            y=self.mult(x); % C_xt x
                    end
                case -1
                    y=self.solve(x);
                otherwise
                    switch self.ker_opt
                        case {'sep','kron_prod'}
                            if size(x,1)~=self.N
                                x=reshape(x,self.N,[]); % (IJ,K_)
                            end
                            [eigf,eigv]=self.eigs;
                            y=eigf*(((alpha<0).*self.jit+eigv).^alpha.*(eigf'*x)); % C_z^alpha x
                        case 'kron_sum'
                            y=self.mult(x,alpha); % C_xt^alpha x
                    end
            end
        end
        
        function ldet=logdet(self)
            % log-determinant of the joint kernel
            switch self.ker_opt
                case {'sep','kron_prod'}
                    [~,eigv]=self.eigs;
                    ldet=sum(log(abs(eigv))); % log |C_z|
                case 'kron_sum'
                    ldet=2.*sum(log(abs(self.Lambda(:)))); % log |C_xt|
            end
        end
        
        function [logpdf,half_ldet]=matn0pdf(self,X,nu)
            % logpdf of centered matrix normal distribution X ~ MN(0,C,nuI)
            % for dynamic spatial kernel C_xt in model II
            if ~exist('nu','var') || isempty(nu)
                nu=1;
            end
            if self.opt==2 % for C_xt
                if size(X,1)~=self.I
                    X=reshape(X,self.I,self.J,[]);
                end
                half_ldet=-size(X,3).*sum(log(abs(self.Lambda(:))));
                Phi_x=self.C_x.eigs(self.L);
%                 half_quad=(Phi_x'*X(:,:))./repmat(self.Lambda',1,size(X,3));
%                 half_quad=reshape(Phi_x'*X(:,:),self.L,self.J,[])./self.Lambda';
                half_quad=mtimesx(Phi_x,'t',X)./self.Lambda';
            else % for C_z
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
            end
            quad=-.5*sum(half_quad(:).^(2-(self.opt~=2)*~self.spdapx))./nu;
            logpdf=half_ldet+quad;
        end
        
        function self=update(self,C_x,C_t,Lambda)
            % update the eigen-basis
            if exist('C_x','var') && ~isempty(C_x)
                self.C_x=C_x; self.I=self.C_x.N;
            end
            if exist('C_t','var') && ~isempty(C_t)
                self.C_t=C_t; self.J=self.C_t.N;
            end
            if exist('Lambda','var') && ~isempty(Lambda)
                self.Lambda=Lambda; self.L=size(self.Lambda,2);
            end
            if self.store_eig
                [self.eigf,self.eigv]=self.eigs([],true);
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
            switch self.ker_opt
                case {'sep','kron_prod'}
                    M0=self.act(mvn0Irv(:),.5); % (IJ,1)
                    M0=reshape(M0,self.I,self.J); % (I,J)
                case 'kron_sum'
                    M0=self.C_t.act(mvn0Irv',.5)'; % (I,J)
            end
            if ~exist('MU','var') || isempty(MU)
                M=M0;
            else
                M=MU+M0; % (I,J)
            end
        end
        
    end
end