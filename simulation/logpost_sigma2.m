% log-posterior density of sigma2 (for eps, or t only)

function logpost=logpost_sigma2(sigma2,mgC,y,a,b,out_opt)
if ~exist('out_opt','var')
    out_opt=1;
end

% update marginal covariance
[I,J,K]=deal(mgC.stgp.I,mgC.stgp.J,mgC.K);
mgC=mgC.update(mgC.stgp.update([],mgC.stgp.C_t.update(sigma2(end))),sigma2(1));

% calculate log-likelihood
loglik=zeros(1,2);
Y_bar=mean(y,3); Y_bar=Y_bar(:);
loglik(1+(mgC.stgp.opt==2):2)=mgC.matn0pdf(Y_bar);
if any(out_opt==1) && mgC.stgp.opt~=2
    Y2_bar=sum(y(:).^2)./K; Y_bar2=Y_bar'*Y_bar;
    loglik(1)=loglik(1)-I*J*(K-1)/2*log(sigma2(1))-0.5*K./sigma2(1).*(Y2_bar-Y_bar2);
end

% calculate log-prior
logpri=zeros(1,2);
% if length(out_opt)==1
%     logpri(out_opt)=-(a+1).*log(sigma2(min(out_opt,end)))-b./sigma2(min(out_opt,end));
% else
    logpri(out_opt)=-(a+1).*log(sigma2(out_opt))-b./sigma2(out_opt);
% end
logpri(out_opt==(mgC.stgp.opt==2))=0;

% output log-posterior
logpost=loglik(out_opt)+logpri(out_opt);

end
