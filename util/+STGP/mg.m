%%%% - high-dimensional STGP (marginal) - %%%%
%%%%   Shiwei Lan                         %%%%
%%%%   STAT, UIUC                         %%%%
%%%%   shiwei@illinois.edu                %%%% 
%%%%   lanzithinking@gmail.com            %%%%
%%%%   Copyright @ 2018                   %%%%
%%%% - high-dimensional STGP (marginal) - %%%%

classdef mg
    % high-dimensional Spatio-Temporal Gaussian Process (marginal)
    % -- given spatial and temopral kernels separately, construct the
    % marginal kernel in the model STGP
    properties
        stgp; % spatiotemporal GP kernel
        K=1; % number of trials in spatiotemporal data
        nz_var=1; % noise variance
        L; % Karhunen-Loeve truncation number
        eigf,eigv; % partial (L) eigenpair of the marginal kernel
        store_eig=false; % indicator whether to store eigenpair
        isub; % STGP marginal kernel in an intrinsic subspace
    end
    methods
        function self=mg(stgp,K,nz_var,L,store_eig)
            % constructor
            % initialization
            self.stgp=stgp;
            if exist('K','var')
                self.K=K;
            end
            if exist('nz_var','var')
                self.nz_var=nz_var;
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
            self.isub=feval([strtok(class(stgp),'.'),'.isub'],self.stgp,self.K,self.L,self.store_eig);
        end
        
        function mgCv=mult(self,v)
            % marginal kernel apply (mulitply) a function (vector)
            switch self.stgp.ker_opt % posterior covariance action
                case 'kron_prod'
                    [~,mgCv]=self.stgp.mult(v,[],self.nz_var/self.K);
                case 'kron_sum'
                    [~,mgCv]=self.stgp.mult(v,[],1/self.K);
            end
        end
        
        function mgC=tomat(self)
            % return the marginal kernel in matrix format
            switch self.stgp.ker_opt % posterior covariance action
                case 'kron_prod'
                    [~,mgC]=self.stgp.tomat([],self.nz_var/self.K);
                case 'kron_sum'
                    [~,mgC]=self.stgp.tomat([],1/self.K);
            end
        end
        
        function invmgCv=solve(self,v,woodbury,varargin)
            % marginal kernel solve a function (vector)
            if ~exist('woodbury','var') || isempty(woodbury)
                woodbury=false;
            end
            if self.stgp.N<=1e3
                if size(v,1)~=self.stgp.N
                    v=reshape(v,self.stgp.N,[]);
                end
                invmgCv=self.tomat\v;
            else
                if self.stgp.opt==2 && woodbury
                    if size(v,1)~=self.stgp.I
                        v=reshape(v,self.stgp.I,self.stgp.J,[]); % (I,J,K_)
                    end
                    Phi_x=self.stgp.C_x.eigs(self.L);
                    if size(v,3)==1
                        Phiv=Phi_x'*v; % (L,J)
                        PhiinvSPhiv=Phi_x*reshape(self.isub.solve(Phiv),self.stgp.L,self.stgp.J); % (I,J)
                        projv=v-Phi_x*Phiv; % (I,J)
                        invmgCv=self.stgp.C_t.solve(projv')'+PhiinvSPhiv; % (I,J)
                    else
                        Phiv=Phi_x'*v(:,:); % (L,JK_)
                        PhiinvSPhiv=reshape(Phi_x*reshape(self.isub.solve(Phiv),self.stgp.L,[]),self.stgp.I,self.stgp.J,[]); % (I,J,K_)
                        projv=permute(reshape(v(:,:)-Phi_x*Phiv,self.stgp.I,self.stgp.J,[]),[2,1,3]); % (J,I,K_)
                        invmgCv=permute(reshape(self.stgp.C_t.solve(projv(:,:)),self.stgp.J,self.stgp.I,[]),[2,1,3])+PhiinvSPhiv; % (I,J,K_)
                    end
                    invmgCv=reshape(invmgCv,self.stgp.N,[]); % (IJ,K_)
                else
                    if size(v,1)~=self.stgp.N
                        v=reshape(v,self.stgp.N,[]);
                    end
                    invmgCv=zeros(size(v));
                    if ~isempty(gcp('nocreate'))
                        parfor k=1:size(invmgCv,2)
                            [invmgCv(:,k),flag]=pcg(@self.mult,v(:,k),varargin{:});
                        end
                    else
                        for k=1:size(invmgCv,2)
                            [invmgCv(:,k),flag]=pcg(@self.mult,v(:,k),varargin{:});
                        end
                    end
                end
            end
        end
        
        function [eigf,eigv]=eigs(self,L,upd)
            % obtain partial eigen-basis of marginal kernel
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
                L=min([L,self.stgp.N]);
                if self.stgp.N<=1e3
                    mgC=self.tomat;
                    [eigf,eigv]=eigs(mgC,L,'lm','Tolerance',1e-10,'MaxIterations',100);
                else
                    [eigf,eigv]=eigs(@self.mult,self.stgp.N,L,'lm','Tolerance',1e-10,'MaxIterations',100,'IsFunctionSymmetric',true); % (IJ,L)
                end
                eigv=diag(eigv);
            end
        end
        
        function y=act(self,x,alpha)
            % obtain the action of mgC^alpha
            if ~exist('alpha','var') || isempty(alpha)
                alpha=1;
            end
            if ~isnumeric(alpha) || numel(alpha) ~= 1
                error('Wrong exponent of action!');
            end
            if size(x,1)~=self.stgp.N
                x=reshape(x,self.stgp.N,[]);
            end
            switch alpha
                case 1
                    y=self.mult(x);
                case -1
                    y=self.solve(x,true);
                otherwise
                    [eigf,eigv]=self.eigs;
                    y=eigf*(eigv.^alpha.*(eigf'*x));
            end
        end
        
        function ldet=logdet(self,matdet)
            % log-determinant of the marginal kernel
            if ~exist('matdet','var') || isempty(matdet)
                matdet=false;
            end
            if self.stgp.opt==2 && matdet
                ldet=self.isub.logdet + (self.stgp.I-self.stgp.L).*self.stgp.C_t.logdet;
            else
                [~,eigv]=self.eigs;
                ldet=sum(log(abs(eigv)));
            end
        end
        
        function [logpdf,half_ldet]=matn0pdf(self,X,nu)
            % logpdf of centered matrix normal distribution X ~ MN(0,C,nuI)
            if ~exist('nu','var') || isempty(nu)
                nu=1;
            end
            if size(X,1)~=self.stgp.N
                X=reshape(X,self.stgp.N,[]);
            end
            if self.stgp.opt==2
                half_ldet=-size(X,2).*self.logdet(true)./2;
                quad=X.*self.solve(X,true);
                quad=-.5.*sum(quad(:))./nu;
            else
                [eigf,eigv]=self.eigs;
                rteigv=sqrt(abs(eigv));
                half_ldet=-size(X,2).*sum(log(rteigv));
                half_quad=(eigf'*X)./rteigv;
                quad=-.5*sum(half_quad(:).^2)./nu;
            end
            logpdf=half_ldet+quad;
        end
        
        function self=update(self,stgp,nz_var)
            % update the eigen-basis
            if exist('stgp','var') && ~isempty(stgp)
                self.stgp=stgp;
                if self.store_eig
                    [self.eigf,self.eigv]=self.eigs([],true);
                end
                % update isub
                self.isub=self.isub.update(self.stgp);
            end
            if exist('nz_var','var') && ~isempty(nz_var) && self.stgp.opt==1
                nz_var_=self.nz_var;
                self.nz_var=nz_var;
                if self.store_eig
                    self.eigv=self.eigv+(-nz_var_+self.nz_var)./self.K;
                end
            end
        end
        
        function [M,MU]=sample_postM(self,y)
            % sample posterior mean function (matrix normal)
            Ybar=mean(y,3); % (I,J)
            mvn0Irv=randn(self.stgp.I,self.stgp.J); % (I,J)
            switch self.stgp.ker_opt
                case 'kron_prod'
                    [~,MU]=self.stgp.mult(self.solve(Ybar(:)));% (IJ,1)
                    M0=self.stgp.act(self.act(mvn0Irv(:).*sqrt(self.nz_var/self.K),-0.5),.5); % (IJ,1)
                case 'kron_sum'
                    MU=self.stgp.C_t.mult(reshape(self.solve(Ybar(:),true),self.stgp.I,self.stgp.J)')';% (I,J)
                    if 0%self.stgp.N<=1e3
                        C_tI_x=kron(self.stgp.C_t.tomat,speye(self.stgp.I));
                        Sigma=C_tI_x*self.solve(self.stgp.tomat./self.K);
                        chol_Sigma=chol(Sigma,'lower');
                        M0=reshape(chol_Sigma*mvn0Irv(:),self.stgp.I,self.stgp.J);
                    else
                        Phi_x=self.stgp.C_x.eigs(self.L);
                        Phirv=Phi_x'*mvn0Irv; % (L,J)
                        PhihalfinvSPhirv=Phi_x*reshape(self.isub.act(Phirv,-0.5,-1),self.stgp.L,self.stgp.J); % (I,J)
                        projrv=mvn0Irv-Phi_x*Phirv; % (I,J)
                        M0=self.stgp.C_t.act(projrv',.5)'+PhihalfinvSPhirv; % (I,J)
                    end
            end
            M=MU+M0;
            if size(M,1)~=self.stgp.I
                M=reshape(M,self.stgp.I,self.stgp.J); % (I,J)
            end
        end
        
        function [MU,SIGMA]=predM(self,y,C_E,C_ED)
            % predict the mean function based on block (cross) covariances of new points
            % C_E=C_m(z_*,z_*); C_ED=C_m(z_*,Z)
            Y_bar=mean(y,3);
            MU=C_ED*self.solve(Y_bar(:),true);
            if nargout>1
                SIGMA=C_E-C_ED*self.solve(C_ED',true);
            end
        end
        
    end
end