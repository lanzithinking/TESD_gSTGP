% logpdf of centered matrix normal distribution
% X ~ MN(0,C,vI)

function logpdf=matn0pdf(X,C,v,tol)
if ~exist('v','var')
    v=1;
end
if ~exist('tol','var')
    tol=1e-8;
end

if istril(C)
    chol_C=C;
    half_logdet=sum(log(diag(chol_C)));
    half_quad=chol_C\X;
else
    try
        chol_C=chol(C,'lower');
        half_logdet=sum(log(diag(chol_C)));
        half_quad=chol_C\X;
    catch
        warning('Non-PSD matrix encountered; use LDL decomposition instead...');
        [L_C,D_C]=ldl(C);
        d_C=diag(D_C); pos_ind=d_C>tol;
        ld_idx=find(diag(D_C,-1)~=0); blk2_idx=[ld_idx';ld_idx'+1]; blk2_idx=blk2_idx(:);
        pos_ind(blk2_idx)=false;
        half_logdet=sum(log(d_C(pos_ind)))./2;
        half_quad=L_C\X; half_quad=half_quad(pos_ind,:);
        half_quad=half_quad./sqrt(d_C(pos_ind));
        fprintf('There are %d 2-blcoks in LDL decomposition.\n', length(ld_idx));
    end
end
logpdf=-size(X,2).*half_logdet-.5*sum(half_quad(:).^2)./v;

end