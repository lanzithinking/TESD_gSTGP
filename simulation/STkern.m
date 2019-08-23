% This is to form spatio-temporal kernels

function [C_z,C_xt]=STkern(C_x,C_t,Lambda,opt,full)
if ~exist('Lambda','var') || isempty(Lambda)
    Lambda=eig(C_t);
end
if ~exist('opt','var')
    opt='kron_prod';
end
if ~exist('full','var')
    full=false;
end

J=size(C_t,1);
if size(Lambda,1)~=J
    error('Size of Lambda does not match time-domain dimension!');
end
L=size(Lambda,2);
if isnumeric(C_x) && issymmetric(C_x)
    I=size(C_x,1);
    % adjust Lambda if necessary
    if L>I
        L=I; Lambda=Lambda(:,1:L);
    end
    % find eigen-basis from spatial kernel
    [Phi_x,Lambda_x]=eigs(C_x,L); % (I,L)
    Lambda_x=diag(Lambda_x);
elseif isstruct(C_x)
    % already eigendecomposed
    Phi_x=C_x.eigV; Lambda_x=C_x.eigD; C_x=C_x.C;
    I=size(C_x,1); d_phi=size(Phi_x,2);
    % adjust Lambda or (Phi_x,Lambda_x) if necessary
    if L>d_phi
        L=d_phi; Lambda=Lambda(:,1:L);
    else
        Phi_x=Phi_x(:,1:L); Lambda_x=Lambda_x(1:L);
    end
    % check orthonormality
    if max(max(abs(Phi_x'*Phi_x-eye(L))))>1e-6
        Phi_x=orth(Phi_x);
    end
end

switch opt
    case 'kron_prod'
        PhiLambda=reshape(Phi_x,I,1,[]).*reshape(Lambda,1,J,[]);
        PhiLambda=reshape(PhiLambda,I*J,[]);
        C_xt=PhiLambda*PhiLambda';
        if L<I
            C_x0=(Phi_x.*Lambda_x')*Phi_x';
            C_xt=C_xt+kron(ones(J),C_x-C_x0);
        end
        C_z=C_xt.*kron(C_t,ones(I))+1e-6.*speye(I*J);
    case 'kron_sum'
        PhiLambda2=reshape(Phi_x,I,1,[]).*reshape(Lambda.^2,1,J,[]);
        PhiLambda2=reshape(PhiLambda2,I*J,[]);
        PhiLambda2=PhiLambda2*Phi_x';
        if L<I
            C_x0=(Phi_x.*Lambda_x')*Phi_x';
            PhiLambda2=PhiLambda2+repmat(C_x-C_x0,J,1);
        end
        C_xt=mat2cell(PhiLambda2,I.*ones(J,1));
        C_xt_full=sparse(blkdiag(C_xt{:}))+1e-6.*speye(I*J);
        if full
            C_xt=C_xt_full;
        end
        C_z=C_xt_full+kron(C_t,speye(I));
end
