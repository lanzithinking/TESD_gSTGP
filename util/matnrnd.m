% generate sample(s) from a matrix Normal distribution

function X=matnrnd(M,U,V,N)
if ~exist('M','var') || isempty(M)
    M=0;
end
[I,J,K]=size(M);
if ~exist('U','var') || isempty(U)
    U=eye(I);
end
if ~exist('V','var') || isempty(V)
    V=eye(J);
end
if ~exist('N','var')
    N=max(1,K);
end

if istril(U)
    chol_U=U;
else
   chol_U=chol(U,'lower');
end
if istriu(V)
    chol_V=V;
else
    chol_V=chol(V);
end

X=randn(I,J,N);
for n=1:N
    X(:,:,n)=M(:,:,min(n,end))+chol_U(:,:,min(n,end))*X(:,:,n)*chol_V(:,:,min(n,end));
end
