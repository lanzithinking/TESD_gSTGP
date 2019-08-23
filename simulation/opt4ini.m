% optimization for MCMC initialization
% for multiple trials

function [sigma2,eta,M,Lambda,objf]=opt4ini(sigma2,eta,M,Lambda,y,ker,a,b,m,V,mdl_opt,opt_id,jtopt,Nmax,thld)
if ~exist('Nmax','var')
    Nmax=100;
end
if ~exist('thld','var')
    thld=1e-3;
end
if ~exist('opt_id','var') || isempty(opt_id)
    opt_id=true(1,4);
end
if ~exist('jtopt','var') || isempty(jtopt)
    jtopt=true;
end

% dimension
[I,J,K]=size(y); L=size(Lambda,2);

% constant updates
dlta=[I*J*K,I*J,J*L]./2;
alpha=a+dlta;

% initialization
stgp=STGP(ker{1}.C,ker{2}.C,Lambda,mdl_opt);
objf=nan(1,4);

% optimization setting
opts_unc=optimoptions('fminunc','Algorithm','quasi-newton','display','off','MaxIterations',100);
% opts_unc_g=optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'display','off','MaxIterations',100);
opts_con=optimoptions('fmincon','Algorithm','sqp','display','off','MaxIterations',100);

% optimization
fprintf('Optimizing parameters...\n');
prog=0.05:0.05:1;
tic;
for iter=1:Nmax
    % record current value
    sigma2_=sigma2;
    eta_=eta;
    M_=M; % (I,J)
    Lambda_=Lambda; % (J,L)
    objf_=objf;
    
    % display the progress
    if ismember(iter,floor(Nmax.*prog))
        fprintf('%.0f%% of max iterations completed.\n',100*iter/Nmax);
    end
    
    if opt_id(1)
        % normalize C_t in stgp before updating sigma2
        stgp.C_t=stgp.C_t./sigma2(2);
        [C_z,C_xt]=stgp.get_jtker();
        
        % update sigma2
        switch mdl_opt
            case 'kron_prod'
                C0_pri=C_z; C0_lik=speye(stgp.I*stgp.J);
                % sigma2_eps
                if ~jtopt
                    logf{1}=@(q)logpost_sigma2([q,sigma2(2)],C0_pri,C0_lik,y,a(1),b(1),1);
                    [sigma2(1),nl_sigma2(1)]=fmincon(@(q)-logf{1}(q),sigma2(1),[],[],[],[],0,[],[],opts_con);
                end
            case 'kron_sum'
                C0_pri=sparse(kron(stgp.C_t,speye(stgp.I))); C0_lik=C_xt;
        end
        % sigma2_t
        if ~jtopt
            logf{2}=@(q)logpost_sigma2([sigma2(1),q],C0_pri,C0_lik,y,a(2),b(2),2);
            [sigma2(2),nl_sigma2(2)]=fmincon(@(q)-logf{2}(q),sigma2(2),[],[],[],[],0,[],[],opts_con);
        else
            % jointly optimize
            logF=@(q)logpost_sigma2(q,C0_pri,C0_lik,y,a(1:2),b(1:2),1:2);
            [sigma2(1:2),nl_sigma2]=fmincon(@(q)-sum(logF(q)),sigma2(1:2),[],[],[],[],zeros(2,1),[],[],opts_con);
        end
        % sigma2_tilt
        Lambda_til=Lambda.*(1:L).^(ker{3}.kappa/2);
        quad=Lambda_til.*(ker{3}.C\Lambda_til);
        dltb(3)=0.5.*sum(quad(:)).*sigma2(3);
        beta=b+dltb;
        sigma2(3)=beta(3)./(alpha(3)+1);
        objf(1)=sum(nl_sigma2)-(log(gampdf(1./sigma2(3),alpha(3),1./beta(3)))-2*log(sigma2(3)));
    end
    if opt_id(2)
        % update eta
        if ~jtopt
            % eta_x
            logf{1}=@(q)logpost_eta([q,eta(2)],sigma2(1:2),ker(1:2),stgp,y,m(1),V(1),1);
            [eta(1),nl_eta(1)]=fminunc(@(q)-logf{1}(q),eta(1),opts_unc);
            % eta_t
            logf{2}=@(q)logpost_eta([eta(1),q],sigma2(1:2),ker(1:2),stgp,y,m(2),V(2),2);
            [eta(2),nl_eta(2)]=fminunc(@(q)-logf{2}(q),eta(2),opts_unc);
            % eta_tilt
            logf{3}=@(q)logpost_eta(q,sigma2(3),ker(3),stgp,y,m(3),V(3),3);
            [eta(3),nl_eta(3)]=fminunc(@(q)-logf{3}(q),eta(3),opts_unc);
        else
            % jointly optimize
            logF=@(q)logpost_eta(q,sigma2,ker,stgp,y,m,V,1:3);
            [eta,nl_eta]=fminunc(@(q)-sum(logF(q)),eta,opts_unc);
        end
        objf(2)=sum(nl_eta);
    end
    if any(opt_id(1:2))
        % update kernels
        for k=1:length(ker)
            ker{k}.C=sigma2(k)^(k~=1).*(exp(-.5.*ker{k}.dist.*exp(-ker{k}.s.*eta(k)))+ker{k}.jit);
        end
        stgp.C_x=ker{1}.C; stgp=stgp.get_spat(); stgp.C_t=ker{2}.C;
    end
    if opt_id(3)
        % update Lambda
        logLik_Lambda=@(q)loglik_Lambda(q,sigma2(1),stgp,y);
        [Lambda,objf(3)]=fminunc(@(q)-logLik_Lambda(q)-matn0pdf(q,ker{3}.C),Lambda,opts_unc);
%         % to-do
%         logf_Lambda=@(q)geom_Lambda(q,sigma2(1),stgp,y);
%         [Lambda,objf(4)]=fminunc(@(q)logf_Lambda(q),Lambda,opts_unc_g);
        % update Lambda
        stgp.Lambda=Lambda;
    end
    if opt_id(4)
        % update M
        [mu,chol_Sigma]=stgp.post_mean(y,sigma2(1));
        M=reshape(mu,I,J);
        objf(4)=sum(log(diag(chol_Sigma)));
    end
    
    % display current objective function
%     fprintf(['Objective function values: ',repmat('%.4f, ',1,length(objf)),'at iteration %d.\n'], objf, iter);
    fprintf('Objective function values: f_sigma2=%.4f, f_eta=%.4f, f_Lambda=%.4f, and f_M=%.4f at iteration %d.\n', objf, iter);
    
    % break if condition satisfied
    if iter>1
        dif(1)=max(abs(sigma2_-sigma2)); dif(2)=max(abs(eta_-eta));
        dif(3)=max(abs(Lambda_(:)-Lambda(:))); dif(4)=max(abs(M_(:)-M(:))); 
        if all(abs(objf_-objf)<thld) || all(dif<thld)
            fprintf('Optimization breaks at iteration %d.\n',iter);
            break;
        end
    end

end

% count time
time=toc;
fprintf('Time used: %.2f seconds.\n',time);
