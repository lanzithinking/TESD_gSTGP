% log-likelihood function of Lambda

function loglik=loglik_Lambda(Lambda,mgC,y,M)
if ~exist('y','var')
    y=[];
end
if exist('M','var') && ~isempty(M)
    intM=false;
else
    intM=true;
end

% update marginal kernel
mgC=mgC.update(mgC.stgp.update([],[],Lambda));

if intM
    Y_bar=mean(y,3);
    loglik=mgC.matn0pdf(Y_bar(:));
    if mgC.stgp.opt==2
        [logpdf,half_ldet]=mgC.stgp.matn0pdf(y-Y_bar);
        loglik=loglik+logpdf-half_ldet./mgC.K;
    end
else
    switch mgC.stgp.ker_opt
        case {'sep','kron_prod'}
            loglik=mgC.stgp.matn0pdf(M(:));
        case 'kron_sum'
            loglik=mgC.stgp.matn0pdf(y-M);
    end
end

end