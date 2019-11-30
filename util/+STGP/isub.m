%%%% - high-dimensional STGP (marginal) - %%%%
%%%%   Shiwei Lan                         %%%%
%%%%   STAT, UIUC                         %%%%
%%%%   shiwei@illinois.edu                %%%% 
%%%%   lanzithinking@gmail.com            %%%%
%%%%   Copyright @ 2018                   %%%%
%%%% - high-dimensional STGP (marginal) - %%%%

classdef isub
    % high-dimensional Spatio-Temporal Gaussian Process (marginal in subspace)
    % -- given spatial and temopral kernels separately, construct the
    % marginal kernel in an intrinsic subspace in the model STGP
    properties
        stgp; % spatiotemporal GP kernel
        K=1; % number of trials in spatiotemporal data
        L; % Karhunen-Loeve truncation number
        eigf,eigv; % partial (L) eigenpair of the marginal kernel
        store_eig=false; % indicator whether to store eigenpair
    end
    methods
        function self=isub(stgp,K,L,store_eig)
            % constructor
            % initialization
            self.stgp=stgp;
            if exist('K','var')
                self.K=K;
            end
            if exist('L','var')
                self.L=L;
            else
                self.L=stgp.L;
            end
            if exist('store_eig','var')
                self.store_eig=store_eig;
            end
            if self.store_eig
                % obtain partial eigen-basis
                [self.eigf,self.eigv]=self.eigs;
            end
        end
        
        function Sv=mult(self,v,alpha)
            % action of the kernel in the intrinsic subspace (for model II only)
            % C_t^alpha Ox I_L + K^(-alpha) diag( vec^T(Lambda^(2alpha)) )
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            if ~isnumeric(alpha) || numel(alpha) ~= 1
                error('Wrong exponent of action!');
            end
            [J,L]=deal(self.stgp.J,self.stgp.L);
            if L*J<=1e3
                if size(v,1)~=L*J
                    v=reshape(v,L*J,[]); % (LJ,K_)
                end
                S=self.tomat(alpha);
                Sv=S*v;
            else
                if size(v,1)~=L
                    v=reshape(v,L,J,[]); % (L,J,K_)
                end
                Sv=(self.stgp.Lambda'.^2./self.K).^alpha.*v; % (L,J,K_)
                if size(v,3)==1
                    Sv=Sv+self.stgp.C_t.act(v',alpha)'; % (L,J)
                else
                    v=permute(v,[2,1,3]); % (J,L,K_)
                    Sv=Sv+permute(reshape(self.stgp.C_t.act(v(:,:),alpha),J,L,[]),[2,1,3]); % (L,J,K_)
                end
                Sv=reshape(Sv,L*J,[]); % (LJ,K_)
            end
        end
        
        function S=tomat(self,alpha)
            % return the marginal kernel in subspace in matrix format
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            [J,L]=deal(self.stgp.J,self.stgp.L);
            lambda2=self.stgp.Lambda'.^2; lambda2=lambda2(:)./self.K;
            S=spdiags(lambda2.^alpha,0,L*J,L*J); % (LJ,LJ)
            switch alpha
                case 1
                    C_tmat=self.stgp.C_t.tomat;
                    S=S+kron(C_tmat,speye(L)); % (LJ,LJ)
                case -1
                    invC_tmat=self.stgp.C_t.solve(speye(J));
                    S=S+kron(invC_tmat,speye(L)); % (LJ,LJ)
            end
        end
        
        function invSv=solve(self,v,alpha,varargin)
            % solving of the kernel in the intrinsic subspace (for model II only)
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            [J,L]=deal(self.stgp.J,self.stgp.L);
            if size(v,1)~=L*J
                v=reshape(v,L*J,[]); % (LJ,K_)
            end
            if alpha==-1
                lambda2=self.stgp.Lambda'.^2; lambda2=lambda2(:)./self.K;
                lambda2v=lambda2.*v;
            end
            if L*J<=1e3
                if alpha==-1
%                     S=self.tomat;
%                     invSv=(S-spdiags(lambda2,0,L*J,L*J))*(S\(lambda2.*v));
                    invSv=lambda2v-lambda2.*(self.tomat\lambda2v);
                else
                    invSv=self.tomat(alpha)\v; % (LJ,K_)
                end
            else
                if alpha==-1
%                     if size(v,2)==1
%                         invSv=reshape(self.stgp.C_t.mult(reshape(self.solve(lambda2.*v),L,J)')',L*J,1);
%                     else
%                         invSv=permute(reshape(self.solve(lambda2.*v),L,J,[]),[2,1,3]); % (J,L,K_)
%                         invSv=reshape(permute(reshape(self.stgp.C_t.mult(invSv(:,:)),J,L,[]),[2,1,3]),L*J,[]); % (L,J,L_)
%                     end
                    invSv=lambda2v-lambda2.*self.solve(lambda2v);
                else
                    Sf=@(v)self.mult(v,alpha);
                    invSv=zeros(size(v)); % (LJ,K_)
                    if ~isempty(gcp('nocreate'))
                        parfor k=1:size(invSv,2)
                            [invSv(:,k),flag]=pcg(Sf,v(:,k),varargin{:});
                        end
                    else
                        for k=1:size(invSv,2)
                            [invSv(:,k),flag]=pcg(Sf,v(:,k),varargin{:});
                        end
                    end
                end
            end
        end
        
        function [eigf,eigv]=eigs(self,L,upd,alpha,alpha_)
            % obtain partial eigen-basis of marginal kernel in subspace
            if ~exist('L','var') || isempty(L)
                L=self.L;
            end
            if ~exist('upd','var') || isempty(upd)
                upd=false; % force to recalculate in update
            end
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            if ~exist('alpha_','var') || isempty(alpha_)
                alpha_=1;
            end
            if ~(isempty(self.eigf) || isempty(self.eigv)) && L<=self.L && ~upd
                [eigf,eigv]=deal(self.eigf,self.eigv);
                eigf=eigf(:,1:L); eigv=eigv(1:L).^alpha;
            else
                L=min([L,self.stgp.L*self.stgp.J]);
                if self.stgp.L*self.stgp.J<=1e3
                    if alpha>0
                        S=self.tomat(alpha_);
                    else
                        S=self.solve(speye(self.stgp.L*self.stgp.J),alpha_);
                    end
                    [eigf,eigv,flag]=eigs(S,L,'lm','Tolerance',1e-10,'MaxIterations',100,'FailureTreatment','drop'); % (LJ,self.L)
                else
                    if alpha>0
                        Sf=@(v)self.mult(v,alpha_);
                    else
                        Sf=@(v)self.solve(v,alpha_);
                    end
                    [eigf,eigv,flag]=eigs(Sf,self.stgp.L*self.stgp.J,L,'lm','Tolerance',1e-10,'MaxIterations',100,'IsFunctionSymmetric',true,'FailureTreatment','drop'); % (LJ,self.L)
                end
                eigv=diag(eigv).^abs(alpha);
                if flag
                    warning('%d of %d requested eigenvalues are not converged!',L-length(eigv),L);
                end
            end
        end
        
        function y=act(self,x,alpha,alpha_)
            % obtain the action of S(alpha_)^alpha
            if size(x,1)~=self.stgp.L*self.stgp.J
                x=reshape(x,self.stgp.L*self.stgp.J,[]);
            end
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            if ~isnumeric(alpha) || numel(alpha) ~= 1
                error('Wrong exponent of action!');
            end
            if ~exist('alpha_','var') || isempty(alpha_)
                alpha_=1;
            end
            if alpha==1
                y=self.mult(x,alpha_);
            elseif alpha==-1
                y=self.solve(x,alpha_);
            elseif abs(alpha)==0.5 && size(x,1)<=1e3
                S=self.tomat(alpha_);
                cholS=chol(S,'lower');
                if alpha>=0
                    y=cholS*x;
                else
                    y=cholS\x;
                end
            else
                [eigf,eigv]=self.eigs([],[],alpha,alpha_);
                y=eigf*(eigv.*(eigf'*x));
            end
        end
        
        function ldet=logdet(self)
            % log-determinant of the marginal kernel
            [~,eigv]=self.eigs;
            ldet=sum(log(abs(eigv)));
        end
        
        function self=update(self,stgp)
            % update the eigen-basis
            if exist('stgp','var') && ~isempty(stgp)
                self.stgp=stgp;
                if self.store_eig
                    [self.eigf,self.eigv]=self.eigs([],true);
                end
            end
        end
        
    end
end