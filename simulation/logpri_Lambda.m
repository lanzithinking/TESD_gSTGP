% log-prior function of Lambda

function output=logpri_Lambda(Lambda,C_tilt,sigma2,quad_only)
if ~exist('sigma2','var')
    sigma2=1;
end
if ~exist('quad_only','var')
    quad_only=false;
end
J=size(C_tilt,1);

% quad=Lambda.*(kron(C_tilt,C_tilt)\Lambda); quad=sum(quad(:))./sigma2^2;
% Lambda3=reshape(Lambda,J,J,L);
tLambda3=permute(Lambda,[2,1,3]);
rhalf_quad=C_tilt\Lambda(:,:); lhalf_quad=permute(reshape(C_tilt\tLambda3(:,:),J,J,[]),[2,1,3]);
quad=sum(rhalf_quad(:).*lhalf_quad(:))./sigma2^2;
if quad_only
    output=quad;
else
    sz_Lambda=size(Lambda); L=sz_Lambda(end);
    logdet=-2*J*L*sum(log(diag(chol(C_tilt))));
    logpri=logdet-0.5.*quad;
    output=logpri;
end

end