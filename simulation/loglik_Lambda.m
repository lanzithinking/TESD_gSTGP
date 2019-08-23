% log-likelihood function of Lambda

function loglik=loglik_Lambda(Lambda,nz_var,stgp,y)
if ~exist('y','var')
    y=[];
end

% update kernel
stgp.Lambda=Lambda;
% obtain marginal covariance
K=size(y,3);
mgC=stgp.get_margcov([],K,nz_var);
Y_bar=mean(y,3);
loglik=matn0pdf(Y_bar(:),mgC);
if stgp.opt==2
    [logpdf,logdet]=stgp.matn0pdf([],y-Y_bar);
    loglik=loglik+logpdf-logdet./K;
end

end