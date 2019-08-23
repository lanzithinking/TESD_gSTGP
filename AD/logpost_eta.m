% log-posterior density of eta

function logpost=logpost_eta(eta,mgC,y,m,V,out_opt,M,varargin)
if ~exist('out_opt','var')
    out_opt=1;
end
if exist('M','var') && ~isempty(M)
    intM=false;
else
    intM=true;
end

% update kernels
if any(out_opt==1)
    mgC=mgC.update(mgC.stgp.update(mgC.stgp.C_x.update([],exp(eta(1)))));
end
if any(out_opt==2)
    mgC=mgC.update(mgC.stgp.update([],mgC.stgp.C_t.update([],exp(eta(2)))));
end

% calculate log-likelihood
loglik=zeros(1,3);
if any(ismember([1,2],out_opt))
    if intM
        Y_bar=mean(y,3);
        loglik(1:2)=mgC.matn0pdf(Y_bar(:));
        if any(out_opt==1) && mgC.stgp.opt==2
            [logpdf,half_ldet]=mgC.stgp.matn0pdf(y-Y_bar);
            loglik(1)=loglik(1)+logpdf-half_ldet./mgC.K;
        end
    else
        switch mgC.stgp.ker_opt
            case 'kron_prod'
                loglik(1:2)=mgC.stgp.matn0pdf(M(:));
            case 'kron_sum'
                if any(out_opt==1)
                    loglik(1)=mgC.stgp.matn0pdf(y-M);
                end
                if any(out_opt==2)
                    loglik(2)=mgC.stgp.C_t.matn0pdf(M');
                end
        end
    end
    % adjust gamma_l=rteigv(C_x) when kappa is set to 'eigCx'
    if any(out_opt==1) && ischar(mgC.stgp.kappa)&&contains(mgC.stgp.kappa,'eigCx')
        loglik(1)=loglik(1)-.5*mgC.stgp.J*mgC.stgp.C_x.logdet;
    end
end
if any(out_opt==3)
    sigma2_u=varargin{1}.sigma2;
    C0_u=varargin{1}.update(1,exp(eta(end)));
    loglik(3)=C0_u.matn0pdf(mgC.stgp.scale_Lambda,sigma2_u);
end

% calculate log-prior
logpri=zeros(1,3);
if length(out_opt)==1
    logpri(out_opt)=-.5*(eta(min(out_opt,end))-m).^2./V; % take care of eta(3)
else
    logpri(out_opt)=-.5*(eta(out_opt)-m).^2./V;
end

% output log-posterior
logpost=loglik(out_opt)+logpri(out_opt);

end
