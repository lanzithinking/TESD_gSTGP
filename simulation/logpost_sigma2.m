% log-posterior density of sigma2 (for eps, or t only)

function logpost=logpost_sigma2(sigma2,C0_pri,C0_lik,y,a,b,out_opt)
if ~exist('out_opt','var')
    out_opt=1;
end

% obtain marginal covariance
K=size(y,3);
mdl1_ind=isdiag(C0_lik);
mgC=sigma2(end).*C0_pri+K^(-1).*sigma2(1)^mdl1_ind.*C0_lik;

% calculate log-likelihood
loglik=zeros(1,2);
Y_bar=mean(y,3); Y_bar=Y_bar(:);
loglik((2-mdl1_ind):2)=matn0pdf(Y_bar,mgC);
if any(out_opt==1) && mdl1_ind
    Y2_bar=sum(y(:).^2)./K; Y_bar2=Y_bar'*Y_bar;
    loglik(1)=loglik(1)-size(C0_pri,1)*(K-1)/2*log(sigma2(1))-0.5*K./sigma2(1).*(Y2_bar-Y_bar2);
end

% calculate log-prior
logpri=zeros(1,2);
if length(out_opt)==1
    logpri(out_opt)=-(a+1).*log(sigma2(min(out_opt,end)))-b./sigma2(min(out_opt,end));
else
    logpri(out_opt)=-(a+1).*log(sigma2(out_opt))-b./sigma2(out_opt);
end
logpri(out_opt==~mdl1_ind)=0;

% output log-posterior
logpost=loglik(out_opt)+logpri(out_opt);

end
