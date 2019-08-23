% log-posterior density of eta

function [logpost,stgp0]=logpost_eta(eta,sigma2,ker,stgp0,y,m,V,out_opt)
if ~exist('out_opt','var')
    out_opt=1;
end

% update kernels
if any(out_opt==1)
    stgp0.C_x=exp(-.5.*ker{1}.dist.*exp(-ker{1}.s.*eta(1)))+ker{1}.jit;
    stgp0=stgp0.get_spat();
end
if any(out_opt==2)
    stgp0.C_t=exp(-.5.*ker{2}.dist.*exp(-ker{2}.s.*eta(2)))+ker{2}.jit;
end

% calculate log-likelihood
loglik=zeros(1,3);
if any(ismember([1,2],out_opt))
    stgp0.C_t=sigma2(2).*stgp0.C_t;
%     stgp0.Lambda=Lambda; % updated outside
    % obtain marginal covariance
    K=size(y,3);
    mgC=stgp0.get_margcov([],K,sigma2(1));
    Y_bar=mean(y,3);
    loglik(1:2)=matn0pdf(Y_bar(:),mgC);
    if any(out_opt==1) && stgp0.opt==2
        [logpdf,logdet]=stgp0.matn0pdf([],y-Y_bar);
        loglik(1)=loglik(1)+logpdf-logdet./K;
    end
end
if any(out_opt==3)
    C0_tilt=exp(-.5.*ker{end}.dist.*exp(-ker{end}.s.*eta(end)))+ker{end}.jit;
    loglik(3)=matn0pdf(stgp0.Lambda.*(1:size(stgp0.Lambda,2)).^(ker{end}.kappa/2),C0_tilt,sigma2(end));
end

% calculate log-prior
logpri=zeros(1,3);
if length(out_opt)==1
    logpri(out_opt)=-.5*(eta(min(out_opt,end))-m).^2./V;
else
    logpri(out_opt)=-.5*(eta(out_opt)-m).^2./V;
end

% output log-posterior
logpost=loglik(out_opt)+logpri(out_opt);

end
