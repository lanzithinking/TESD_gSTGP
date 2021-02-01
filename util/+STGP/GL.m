%%%% - Kernel of Graphical Laplacian - %%%%
%%%%   Shiwei Lan                      %%%%
%%%%   STAT, UIUC                      %%%%
%%%%   shiwei@illinois.edu             %%%% 
%%%%   lanzithinking@gmail.com         %%%%
%%%%   Copyright @ 2018                %%%%
%%%% - Kernel of Graphical Laplacian - %%%%

classdef GL
    % Kernel of Graphical Laplacian 
    % C=sigma2 * (tau2 I + Lap)^(-s)
    % where Lap is the normalized graph laplacian with banded weights
    % -- classical operations in high dimensions
    properties
        g; % image graph 
        imsz; % image size
        sigma2=1; % magnitude of the kernel
        tau2=1; % (inverse) length-scale of the kernel
        l=.5; % correlation length of the weighting function
        s=2; % smoothness of the kernel
        Lap; % graph lapalcian
        d,N,L; % input space dimensions, number of inputs, Karhunen-Loeve truncation number
        jit=1e-6; % jitter added to the kernel
        eigf,eigv; % partial (L) eigenpair of the kernel
        store_eig=true; % indicator whether to store eigenpair
    end
    methods
        function self=GL(g,sigma2,tau2,s,L,jit,store_eig)
            % constructor
            % initialization
            if exist('g','var') && isa(g,'graph')
                self.g=g; % g is a graph or image size
                self.imsz=max(self.g.Nodes{:,1:end-1});
            else
                if ~isfield(g,'size')
                    error('Please provide the size of the graph!');
                end
                self.imsz=g.size;
                if ~isfield(g,'w')
                    g.w=1; % bandwidth of weighting
                end
                conn=ones(2*g.w+1);
                self.g=imageGraph(self.imsz,conn); % self.g=imageGraph3(g,conn); % for 3d graph
                if isfield(g,'mask')
                    % impose ROI
                    unmsk_idx=find(~g.mask(:));
                    for k=1:length(g.size)
                        self.g.Edges.Weight(ismember(self.g.Edges.EndNodes(:,k),unmsk_idx))=0;
                    end
                end
            end
            self.d=length(self.imsz); self.N=size(self.g.Nodes,1);
            if exist('sigma2','var') && ~isempty(sigma2)
                self.sigma2=sigma2;
            end
            if exist('tau2','var') && ~isempty(tau2)
                self.tau2=tau2;
            end
            if exist('s','var') && ~isempty(s)
                self.s=s;
            end
            if isfield(g,'l')
                self.l=g.l;
                % update weights of edges
                nz_idx=(self.g.Edges.Weight~=0);
                self.g.Edges.Weight(nz_idx)=exp(-0.5.*(self.g.Edges.Weight(nz_idx)./self.l).^self.s);
                % calculate graph laplacian
                self.Lap=spdiags(degree(self.g),0,self.N,self.N)-adjacency(self.g,'weighted');
                % normalize the graph laplacian
                rtdgl=sqrt(degree(self.g));
                self.Lap=self.Lap./rtdgl./rtdgl';
            else
                self.Lap=laplacian(self.g); % unweighted Laplacian
            end
            if self.d>=2
                % adjust for the graph size
                self.Lap=self.Lap./(self.N.^(1-2./self.d).*(log(self.N)).^(1+2./self.d));
            end
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
        
        function P=prec(self,alpha)
            % obtain the precision matrix P=tau2 I + Lap
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            P=self.tau2.*speye(self.N) + self.Lap;
            if alpha~=1
                if floor(alpha)==alpha
                    P=P^abs(alpha);
                    if alpha<0
                        P=inv(P);
                    end
                else
                    [eigf,eigv]=self.eigs([],[],-1); % (N,L)
                    P=(eigf.*((alpha<0).*self.jit+eigv)'.^alpha)*eigf';
                end
            end
        end
        
        function C=tomat(self)
            % return the kernel in matrix format C=sigma2 P^-s
            C=self.sigma2.*(self.prec(-self.s)+self.jit.*speye(self.N));
            if self.N>1e3 && ~issparse(C)
                warning('Possible memory overflow!');
            end
        end
        
        function Cv=mult(self,v)
            % mulitply a function (vector)
            if size(v,1)~=self.N
                v=reshape(v,self.N,[]);
            end
            Ps=self.prec(self.s);
            Cv=(Ps\v).*self.sigma2;
        end
        
        function invCv=solve(self,v)
            % solve a funciton (vector)
            if size(v,1)~=self.N
                v=reshape(v,self.N,[]);
            end
            Ps=self.prec(self.s);
            invCv=(Ps*v)./self.sigma2;
        end
        
        function [eigf,eigv]=eigs(self,L,upd,alpha)
            % obtain partial eigen-basis of C^alpha
            if ~exist('L','var') || isempty(L)
                L=self.L;
            end
            if ~exist('upd','var') || isempty(upd)
                upd=false; % force to recalculate in update
            end
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            if ~(isempty(self.eigf) || isempty(self.eigv)) && L<=self.L && ~upd
                [eigf,eigv]=deal(self.eigf,self.eigv);
                eigf=eigf(:,1:min([end,L])); eigv=eigv(1:min([end,L]));
            else
                L=min([L,self.N]);
                P=self.prec;
%                 [eigf,eigv,flag]=eigs(P,L,'sm','Tolerance',self.jit,'MaxIterations',100,'FailureTreatment','drop'); % (N,L)
                [eigf,eigv,flag]=eigs(P,L,'sm','Tolerance',self.jit,'MaxIterations',100); % (N,L)
                eigv=diag(eigv);
                if alpha~=-1
                    eigv=self.sigma2.^(alpha).*(eigv+(alpha>0).*self.jit).^(-self.s.*alpha);
                end
                if flag
%                     warning('%d of %d requested eigenvalues are not converged!',L-length(eigv),L);
                    divrg_ind=isnan(eigv);
                    eigv(divrg_ind)=0;
                    warning('%d of %d requested eigenvalues are not converged!',sum(divrg_ind),L);
                end
            end
        end
        
        function y=act(self,x,alpha)
            % obtain the action of C^alpha
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            if ~isnumeric(alpha) || numel(alpha) ~= 1
                error('Wrong exponent of action!');
            end
            if size(x,1)~=self.N
                x=reshape(x,self.N,[]);
            end
            if alpha==1
                y=self.mult(x);
            elseif alpha==-1
                y=self.solve(x);
%             elseif abs(alpha)==0.5 && self.N<=1e3
%                 C=self.tomat;
%                 cholC=chol(C,'lower');
%                 if alpha>=0
%                     y=cholC*x;
%                 else
%                     y=cholC\x;
%                 end
            else
                [eigf,eigv]=self.eigs;
                y=eigf*(((alpha<0).*self.jit+eigv).^alpha.*(eigf'*x));
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
            if size(X,1)~=self.N
                X=reshape(X,self.N,[]);
            end
            if self.N<=1e3
                C=self.tomat;
                cholC=chol(C,'lower');
                half_ldet=-size(X,2).*sum(log(diag(cholC)));
                half_quad=cholC\X(:,:);
            else
%                 [eigf,eigv]=self.eigs;
%                 rteigv=sqrt(abs(eigv)+self.jit);
%                 half_ldet=-size(X,2).*sum(log(rteigv));
                half_ldet=-size(X,2).*self.logdet./2;
%                 half_quad=(eigf'*X)./rteigv;
                half_quad=self.prec(self.s/2)*X./sqrt(self.sigma2);
            end
            quad=-.5*sum(half_quad(:).^2)./nu;
            logpdf=half_ldet+quad;
        end
        
        function self=update(self,sigma2,tau2,l)
            % update the eigen-basis
            if exist('sigma2','var') && ~isempty(sigma2)
                sigma2_=self.sigma2;
                self.sigma2=sigma2;
                if self.store_eig
                    self.eigv=self.eigv./sigma2_.*self.sigma2;
                end
            end
            if exist('tau2','var') && ~isempty(tau2)
                tau2_=self.tau2;
                self.tau2=tau2;
                if self.store_eig
                    self.eigv=(self.eigv.^(-1./self.s)+(-tau2_+self.tau2).*self.sigma2.^(-1./self.s)).^(-self.s);
                end
            end
            if exist('l','var') && ~isempty(l)
                nz_idx=(self.g.Edges.Weight~=0); wts_=(-2.*log(self.g.Edges.Weight(nz_idx))).^(1./self.s).*self.l;
                self.l=l;
                self.g.Edges.Weight(nz_idx)=exp(-0.5.*(wts_./self.l).^self.s);
%                 self.Lap=laplacian(self.g);
                self.Lap=spdiags(degree(self.g),0,self.N,self.N)-adjacency(self.g,'weighted');
                rtdgl=sqrt(degree(self.g));
                self.Lap=self.Lap./rtdgl./rtdgl';
                if self.d>=2
                    self.Lap=self.Lap./(self.N.^(1-2./self.d).*(log(self.N)).^(1+2./self.d));
                end
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