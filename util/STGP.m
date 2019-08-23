%%%% - Spatio-Temporal Gaussian Process - %%%%
%%%%   Shiwei Lan                         %%%%
%%%%   CMS, CalTech                       %%%%
%%%%   slan@caltech.edu                   %%%% 
%%%%   lanzithinking@gmail.com            %%%%
%%%%   Copyright @ 2018                   %%%%
%%%% - Spatio-Temporal Gaussian Process - %%%%

classdef STGP
    % Spatio-Temporal Gaussian Process
    % -- given spatial and temopral kernels separately, construct a joint
    % kernel that model the temporal evolution of spatial dependence
    properties
        C_x; % spatial kernel
        Phi_x,Lambda_x; % eigenpair of spatial kernel
        C_t; % temporal kernel
        Lambda; % time-varying eigenvalues in K-L expansion for C_x|t
        I,J,L; % spatial/temporal dimensions, Karhunen-Loeve truncation number
        jtkers={'kron_prod','kron_sum'}; % joint kernel models
        opt=2; % joint kernel model choice
        ker_opt; % joint kernel
        jit=1e-6; % jitter added to joint kernel
        trtdeg=true; % special treatment of degeneracy
        bkdgix=[]; % index of block diagonal matrix C_xt (model II)
    end
    methods
        function obj=STGP(C_x,C_t,Lambda,opt,jit,trtdeg)
            % constructor
            % initialization
            obj.C_x=C_x;
            obj.C_t=C_t;
            if ~exist('Lambda','var') || isempty(Lambda)
                obj.Lambda=eig(C_t);
            else
                obj.Lambda=Lambda;
            end
            obj.I=size(obj.C_x,1);
            obj.J=size(obj.C_t,1);
            if size(obj.Lambda,1)~=obj.J
                error('Size of Lambda does not match time-domain dimension!');
            end
            obj.L=size(obj.Lambda,2);
            if obj.L>obj.I
                warning('Karhunen-Loeve truncation number cannot exceed the size of spatial basis!');
                obj.L=obj.I; obj.Lambda=obj.Lambda(:,1:obj.I);
            end
            if exist('opt','var')
                if isnumeric(opt)
                    obj.opt=opt;
                elseif ischar(opt)
                    obj.opt=1:length(obj.jtkers);
                    obj.opt=obj.opt(contains(obj.jtkers,opt));
                else
                    error('Wrong option!');
                end
            end
            obj.ker_opt=obj.jtkers{obj.opt};
            if exist('jit','var')
                obj.jit=jit;
            end
            if exist('trtdeg','var')
                obj.trtdeg=trtdeg;
            end
            % obtain spatial basis
            obj=obj.get_spat();
            % obtain index of block diagonal matrix C_xt (model II)
            if obj.opt==2
                obj=obj.get_bkdgix('IJI');
            end
        end
        
        function obj=get_spat(obj)
            % obtain spatial basis
            [obj.Phi_x,Lambda_x]=eigs(obj.C_x,obj.L); % (I,L)
            obj.Lambda_x=diag(Lambda_x);
        end
        
        function obj=get_bkdgix(obj,ord)
            % obtain index of block diagonal matrix (IxI)*J
            if ~exist('ord','var')
                ord='IIJ';
            end
            [i_dgix,j_dgix]=find(kron(speye(obj.J),ones(obj.I)));
            if ~strcmp(ord,'IIJ')
                perm=[strfind(ord,'I'),strfind(ord,'J')];
                i_dgix=permute(reshape(i_dgix,obj.I,obj.I,obj.J),perm);
                j_dgix=permute(reshape(j_dgix,obj.I,obj.I,obj.J),perm);
            end
            obj.bkdgix=[i_dgix(:),j_dgix(:)];
        end
        
        function [C_z,C_xt]=get_jtker(obj,Lambda,pow)
            % obtain (power of) joint kernel
            if ~exist('Lambda','var') || isempty(Lambda)
                Lambda=obj.Lambda; L=obj.L;
            else
                L=size(Lambda,2);
            end
            if ~exist('pow','var')
                pow=1; % -1: inverse; 0.5 square-root
            end
            Lambda_p = Lambda.^pow;
            switch obj.ker_opt
                case 'kron_prod'
                    PhiLambda=reshape(obj.Phi_x,obj.I,1,[]).*reshape(Lambda_p,1,obj.J,[]);
                    PhiLambda=reshape(PhiLambda,obj.I*obj.J,[]);
                    C_xt=PhiLambda*PhiLambda';
                    if obj.trtdeg && L<obj.I
                        C_x0=(obj.Phi_x.*obj.Lambda_x')*obj.Phi_x';
                        C_xt=C_xt+repmat(obj.C_x-C_x0,obj.J,obj.J);
                    end
                    C_z=C_xt.*kron(obj.C_t,ones(obj.I))+obj.jit.*speye(obj.I*obj.J);
                case 'kron_sum'
                    PhiLambda2=reshape(obj.Phi_x,obj.I,1,[]).*reshape(Lambda_p.^2,1,obj.J,[]);
                    PhiLambda2=reshape(PhiLambda2,obj.I*obj.J,[]);
                    PhiLambda2=PhiLambda2*obj.Phi_x'+repmat(obj.jit.*speye(obj.I),obj.J,1);
                    if obj.trtdeg && L<obj.I
                        C_x0=(obj.Phi_x.*obj.Lambda_x')*obj.Phi_x';
                        PhiLambda2=PhiLambda2+repmat(obj.C_x-C_x0,obj.J,1);
                    end
                    C_xt=sparse(obj.bkdgix(:,1),obj.bkdgix(:,2),PhiLambda2(:));
                    C_z=C_xt+kron(obj.C_t,speye(obj.I));
            end
        end
        
        function [mgC,chol_mgC]=get_margcov(obj,Lambda,K,nz_var)
            % obtain marginal covariance kernel
            if ~exist('Lambda','var') || isempty(Lambda)
                Lambda=obj.Lambda;
            end
            if ~exist('K','var')
                K=1;
            end
            if ~exist('nz_var','var')
                nz_var=1;
            end
            [C_z,C_xt]=obj.get_jtker(Lambda);
            switch obj.ker_opt
                case 'kron_prod'
                    mgC=C_z+nz_var/K.*speye(obj.I*obj.J);
                case 'kron_sum'
                    mgC=kron(obj.C_t,speye(obj.I))+C_xt./K;
            end
            if nargout>1
                try
                    chol_mgC=chol(mgC,'lower');
                catch
                    warning('Non-PSD covariance encountered');
                    chol_mgC=chol(mgC+obj.jit.*speye(obj.I*obj.J),'lower');
                end
            end
        end
        
        function [logpdf,logdet]=matn0pdf(obj,Lambda,X,v)
            % logpdf of centered matrix normal distribution X ~ MN(0,C,vI)
            if ~exist('Lambda','var') || isempty(Lambda)
                Lambda=obj.Lambda; L=obj.L;
            else
                L=size(Lambda,2);
            end
            if ~exist('v','var')
                v=1;
            end
            if ~obj.trtdeg
                % method 1 (non-degeneracy)
                [C_z,C_xt]=obj.get_jtker();
                switch obj.ker_opt
                    case 'kron_prod'
                        C_=C_z; C_z=[];
                    case 'kron_sum'
                        C_=C_xt; C_xt=[];
                end
                try
                    chol_C=chol(C_,'lower');
                catch
                    warning('Non-PSD covariance encountered');
                    chol_C=chol(C_+obj.jit.*speye(obj.I*obj.J),'lower');
                end
                logdet=-size(X,1+obj.opt).*sum(log(diag(chol_C)));
                half_quad=chol_C\X(:,:);
            else
                % method 2
                switch obj.ker_opt
                    case 'kron_prod'
                        C_z=obj.get_jtker();
                        [Phi_z,Lambda2_z]=eigs(C_z,L);Lambda_z=sqrt(diag(Lambda2_z));
                        logdet=-size(X,2).*sum(log(Lambda_z));
                        half_quad=Lambda_z.^(-1).*(Phi_z'*X);
                    case 'kron_sum'
                        K=size(X,3);
                        logdet=-K.*sum(log(abs(Lambda(:))));
%                         half_quad=(obj.Phi_x'*X(:,:))./repmat(Lambda',1,K);
                        half_quad=reshape(obj.Phi_x'*X(:,:),L,obj.J,K)./Lambda';
                end
            end
            quad=-.5*sum(half_quad(:).^2)./v;
            logpdf=logdet+quad;
        end
        
        function [mu,chol_Sigma]=post_mean(obj,y,nz_var)
            % posterior of mean function (matrix normal)
            K=size(y,3);
            [C_z,C_xt]=obj.get_jtker();
            switch obj.ker_opt
                case 'kron_prod'
                    Sigma0=C_z/(C_z+nz_var/K.*speye(obj.I*obj.J));
                    Sigma=Sigma0.*(nz_var/K);
                case 'kron_sum'
                    C_tI_x=kron(obj.C_t,speye(obj.I));
                    Sigma0=C_tI_x/(C_tI_x+C_xt./K);
                    Sigma=Sigma0*C_xt./K;
            end
            try
                chol_Sigma=chol(Sigma,'lower');
            catch
                warning('Non-PSD covariance encountered');
                chol_Sigma=chol(Sigma+obj.jit.*speye(obj.I*obj.J),'lower');
            end
            Y_bar=mean(y,3);
            mu=Sigma0*Y_bar(:);
        end
        
        function [mu,Sigma]=pred_mean(obj,y,nz_var,C_E,C_ED)
            % prediction of mean function based on block (cross) covariances of new points
            % C_E=C_m(z_*,z_*); C_ED=C_m(z_*,Z)
            K=size(y,3);
            mgC=obj.get_margcov([],K,nz_var);
            Sigma0=C_ED/mgC;
            Y_bar=mean(y,3);
            mu=Sigma0*Y_bar(:);
            Sigma=C_E-Sigma0*C_ED';
        end
        
    end
end