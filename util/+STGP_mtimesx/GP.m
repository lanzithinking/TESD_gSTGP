%%%% - Kernel for Gaussian Process - %%%%
%%%%   Shiwei Lan                    %%%%
%%%%   STAT, UIUC                    %%%%
%%%%   shiwei@illinois.edu           %%%% 
%%%%   lanzithinking@gmail.com       %%%%
%%%%   Copyright @ 2018              %%%%
%%%% - Kernel for Gaussian Process - %%%%

classdef GP
    % rbf kernel for Gaussian process
    % -- classical operations in high dimensions
    properties
        x; % input
        sigma2=1; % magnitude of the kernel
        l=.5; % correlation length of the kernel
        s=2; % smoothness of the kernel
        d,N,L; % input space dimensions, number of inputs, Karhunen-Loeve truncation number
        jit=1e-6; % jitter added to the kernel
        eigf,eigv; % partial (L) eigenpair of the kernel
        store_eig=true; % indicator whether to store eigenpair
    end
    methods
        function self=GP(x,sigma2,l,s,L,jit,store_eig)
            % constructor
            % initialization
            if exist('x','var') && ~isempty(x)
                self.x=x;
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
            [self.N,self.d]=size(self.x);
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
            if exist('store_eig','var') && ~isempty(store_eig)
                self.store_eig=store_eig;
            end
            if self.store_eig %&& isa(self,'GP')
                % obtain partial eigen-basis
                [self.eigf,self.eigv]=self.eigs;
            end
        end
        
        function Cv=mult(self,v,varargin)
            % mulitply a function (vector)
            trans=(~isempty(varargin)&&contains(varargin{1},{'t','T'}));
            if ndims(v)<=2
                if ~trans
                    multf=@mtimes;
                else
                    multf=@(a,b)mtimes(a,b')';
                end
            else
                multf=@(a,b)mtimesx(a,b,varargin{:});
            end
            if self.N<=1e3
                Cv=multf(self.tomat,v);
                if ndims(v)>2 && trans
                    Cv=permute(Cv,[2,1,3]);
                end
            else
                Cv=zeros(size(v)); 
                if ~isempty(gcp('nocreate'))
                    if ~trans
                        parfor i=1:self.N
                            row_i=exp(-.5.*(pdist2(self.x(i,:),self.x,'minkowski',self.s)./self.l).^self.s);
                            row_i(i)=row_i(i)+self.jit;
                            Cv(i,:,:)=multf(row_i,v);
                        end
                    else
                        parfor i=1:self.N
                            row_i=exp(-.5.*(pdist2(self.x(i,:),self.x,'minkowski',self.s)./self.l).^self.s);
                            row_i(i)=row_i(i)+self.jit;
                            Cv(:,i,:)=multf(row_i,v);
                        end
                    end
                else
                    S.subs=repmat({':'},1,ndims(v)); S.type='()'; i_dim=1+trans;
                    for i=1:self.N
                        row_i=exp(-.5.*(pdist2(self.x(i,:),self.x,'minkowski',self.s)./self.l).^self.s);
                        row_i(i)=row_i(i)+self.jit;
                        S.subs{i_dim}=i;
                        Cv=subsasgn(Cv,S,multf(row_i,v));
                    end
                end
                Cv=Cv.*self.sigma2; % same size of v
            end
        end
        
        function C=tomat(self)
            % return the kernel in matrix format
            C=exp(-.5.*(pdist2(self.x,self.x,'minkowski',self.s)./self.l).^self.s)+self.jit.*speye(self.N);
            C=C.*self.sigma2;
            if self.N>1e3
                warning('Possible memory overflow!');
            end
        end
        
        function invCv=solve(self,v,varargin)
            % solve a funciton (vector)
            trans=(~isempty(varargin)&&contains(varargin{1},{'t','T'}));
            mdivf=@(a,b)mldivide(a,b(:,:));
            if ndims(v)<=2
                if trans
                    mdivf=@(a,b)mldivide(a,b')';
                end
            else
                mdivf=@(a,b)reshape(mdivf(a,b),size(b,1),size(b,2),[]);
                if trans
                    mdivf=@(a,b)permute(mdivf(a,premute(b,[2,1,3])),[2,1,3]);
                end
            end
            if self.N<=1e3
                invCv=mdivf(self.tomat,v);
            else
                if trans
                    v=permute(v,[2,1,3]);
                end
                szv=size(v); invCv=zeros(szv);
                if ~isempty(gcp('nocreate'))
                    parfor k=1:prod(szv(2:end))
                        [invCv(:,k),flag]=pcg(@self.mult,v(:,k),varargin{:});
                    end
                else
                    for k=1:prod(szv(2:end))
                        [invCv(:,k),flag]=pcg(@self.mult,v(:,k),varargin{:});
                    end
                end
                if ndims(v)>2
                    invCv=reshape(invCv,szv);
                end
                if trans
                    invCv=permute(invCv,[2,1,3]);
                end
            end
        end
        
        function [eigf,eigv]=eigs(self,L,upd)
            % obtain partial eigen-basis
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
                if self.N<=1e3
                    C=self.tomat;
                    [eigf,eigv]=eigs(C,L,'lm','Tolerance',1e-10,'MaxIterations',100); % (N,L)
                else
                    [eigf,eigv]=eigs(@self.mult,self.N,L,'lm','Tolerance',1e-10,'MaxIterations',100,'IsFunctionSymmetric',true); % (N,L)
                end
                eigv=abs(diag(eigv));
            end
        end
        
        function y=act(self,x,alpha,varargin)
            % obtain the action of C^alpha
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            if ~isnumeric(alpha) || numel(alpha) ~= 1
                error('Wrong exponent of action!');
            end
            switch alpha
                case 1
                    y=self.mult(x,varargin{:});
                case -1
                    y=self.solve(x,varargin{:});
                otherwise
                    trans=(~isempty(varargin)&&contains(varargin{1},{'t','T'}));
                    if ndims(x)<=2
                        if ~trans
                            multf=@mtimes;
                        else
                            multf=@(a,b)mtimes(a,b');
                        end
                    else
                        multf=@(a,b)mtimesx(a,b,varargin{:});
                    end
                    [eigf,eigv]=self.eigs;
                    y=mtimesx(eigf,(((alpha<0).*self.jit+eigv).^alpha.*multf(eigf',x)));
                    if trans
                        if ndims(x)<=2
                            y=y';
                        else
                            y=permute(y,[2,1,3]);
                        end
                    end
            end
        end
        
        function ldet=logdet(self)
            % log-determinant of the kernel C
            [~,eigv]=self.eigs;
            ldet=sum(log(abs(eigv)));
        end
        
        function [logpdf,half_ldet]=matn0pdf(self,X,nu)
            % logpdf of centered matrix normal distribution X ~ MN(0,C,nuI)
            if ~exist('nu','var') || isempty(nu)
                nu=1;
            end
            nochol=1;
            if self.N<=1e3
                C=self.tomat;
                [cholC,nochol]=chol(C,'lower');
                if ~nochol
                    half_ldet=-size(X,2).*sum(log(diag(cholC)));
                    half_quad=cholC\X(:,:);
                end
            end
            if nochol
%                 [eigf,eigv]=self.eigs;
%                 rteigv=sqrt(abs(eigv)+self.jit);
%                 half_ldet=-size(X,2).*sum(log(rteigv));
                half_ldet=-size(X,2).*self.logdet./2;
%                 half_quad=(eigf'*X)./rteigv;
                half_quad=X.*self.solve(X);
            end
            quad=-.5*sum(half_quad(:).^(2-nochol))./nu;
            logpdf=half_ldet+quad;
        end
        
        function self=update(self,sigma2,l)
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
        end
        
        function rv=rnd(self,MU,n)
            % generate Gaussian random function (vector)
            if ~exist('MU','var') || isempty(MU)
                MU=sparse(self.N,1);
            end
            if ~exist('n','var') || isempty(n)
                n=1;
            end
            mvn0Irv=randn(self.N,n);
            rv=MU+self.act(mvn0Irv,0.5); % (N,n)
        end
        
    end
end