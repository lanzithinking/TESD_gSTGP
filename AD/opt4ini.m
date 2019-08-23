% optimization for MCMC initialization
% for multiple trials

function [sigma2,eta,M,Lambda,mgC,ker3,objf]=opt4ini(sigma2,eta,M,Lambda,mgC,ker3,y,a,b,m,V,opt_id,intM,jtopt,Nmax,thld)
if ~exist('opt_id','var') || isempty(opt_id)
    opt_id=true(1,4);
end
if ~exist('intM','var') || isempty(intM)
    intM=true;
end
if ~exist('jtopt','var') || isempty(jtopt)
    jtopt=true;
end
if ~exist('Nmax','var')
    Nmax=100;
end
if ~exist('thld','var')
    thld=1e-3;
end

% compatible setting for M
% if ~intM
%     opt_id(end)=true;
% end

% dimension
[I,J,K]=size(y); L=size(Lambda,2);

% constant updates
dlta=[I*J*K,I*J,J*L]./2;
alpha=a+dlta;

% initialization
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
    
    if opt_id(1)
        % update sigma2
        if intM
            if mgC.stgp.opt==1
                % sigma2_eps
                if ~jtopt
                    logf{1}=@(q)logpost_sigma2([q,sigma2(2)],mgC,y,a(1),b(1),1);
                    [sigma2(1),nl_sigma2(1)]=fmincon(@(q)-logf{1}(q),sigma2(1),[],[],[],[],0,[],[],opts_con);
                end
            end
            % sigma2_t
            if ~jtopt
                logf{2}=@(q)logpost_sigma2([sigma2(1),q],mgC,y,a(2),b(2),2);
                [sigma2(2),nl_sigma2(2)]=fmincon(@(q)-logf{2}(q),sigma2(2),[],[],[],[],mgC.stgp.C_t.jit,[],[],opts_con);
            else
                % joint optimize
                logF=@(q)logpost_sigma2(q,mgC,y,a(1:2),b(1:2),1:2);
                [sigma2(1:2),nl_sigma2]=fmincon(@(q)-sum(logF(q)),sigma2(1:2),[],[],[],[],zeros(2,1),[],[],opts_con);
            end
        else
            switch mgC.stgp.opt
                case 1
                    y_ctr=y-M;
                    dltb(1)=0.5.*sum(y_ctr(:).^2);
                    dltb(2)=0.5.*(M(:)'*mgC.stgp.solve(M(:))).*sigma2(2);
                case 2
                    dltb(1)=0;
                    quad=M.*mgC.stgp.C_t.solve(M')';
                    dltb(2)=0.5.*sum(quad(:)).*sigma2(2);
            end
        end
        % sigma2_u
        U=mgC.stgp.scale_Lambda(Lambda);
        quad=U.*ker3.solve(U);
        dltb(3)=0.5.*sum(quad(:)).*sigma2(3);
        beta=b+dltb;
        sigma2_all=beta./(alpha+1); % optimize
        nl_sigma2_all=-(log(gampdf(1./sigma2_all,alpha,1./beta))-2*log(sigma2_all));
        idx2upd=(intM*3+(1-intM)*mgC.stgp.opt):3;
        sigma2(idx2upd)=sigma2_all(idx2upd); nl_sigma2(idx2upd)=nl_sigma2_all(idx2upd);
        objf(1)=sum(nl_sigma2);
        % update kernels
        ker3=ker3.update(sigma2(3));
        mgC=mgC.update(mgC.stgp.update([],mgC.stgp.C_t.update(sigma2(2))),sigma2(1));
    end
    % setting for eta and Lambda
    if intM
        M=[];
    end
    if opt_id(2)
        % update eta
        if ~jtopt
            % eta_x
            logf{1}=@(q)logpost_eta([q,eta(2)],mgC,y,m(1),V(1),1,M);
            [eta(1),nl_eta(1)]=fminunc(@(q)-logf{1}(q),eta(1),opts_unc);
            % eta_t
            logf{2}=@(q)logpost_eta([eta(1),q],mgC,y,m(2),V(2),2,M);
            [eta(2),nl_eta(2)]=fminunc(@(q)-logf{2}(q),eta(2),opts_unc);
            % eta_u
            logf{3}=@(q)logpost_eta(q,mgC,y,m(3),V(3),3,M,ker3);
            [eta(3),nl_eta(3)]=fminunc(@(q)-logf{3}(q),eta(3),opts_unc);
        else
            % joint optimize
            logF=@(q)logpost_eta(q,mgC,y,m,V,1:3,M,ker3);
            [eta,nl_eta]=fminunc(@(q)-sum(logF(q)),eta,opts_unc);
        end
        objf(2)=sum(nl_eta);
         % update kernels
        ker3=ker3.update([],exp(eta(3)));
        mgC=mgC.update(mgC.stgp.update(mgC.stgp.C_x.update([],exp(eta(1))),mgC.stgp.C_t.update([],exp(eta(2)))));
    end
    if opt_id(3)
        % update Lambda
        logLik_Lambda=@(q)loglik_Lambda(q,mgC,y,M);
        [Lambda,objf(3)]=fminunc(@(q)-logLik_Lambda(q)-ker3.matn0pdf(q),Lambda,opts_unc);
        % update marginal kernel
        mgC=mgC.update(mgC.stgp.update([],[],Lambda));
    end
    if opt_id(4)
        % update M
        [~,MU]=mgC.sample_postM(y);
        M=reshape(MU,I,J);
        objf(4)=mgC.stgp.logdet-mgC.logdet;
        switch mgC.stgp.ker_opt
            case 'kron_prod'
                objf(4)=objf(4)+I*J*log(sigma2(1));
            case 'kron_sum'
                objf(4)=objf(4)+I*mgC.stgp.C_t.logdet;
        end
        objf(4)=objf(4)./2;
    end
    
    % display the progress
    if ismember(iter,floor(Nmax.*prog))
        fprintf('%.0f%% of max iterations completed.\n',100*iter/Nmax);
    end
    
    % display current objective function
%     fprintf(['Objective function values: ',repmat('%.4f, ',1,length(objf)),'at iteration %d.\n'], objf, iter);
    fprintf('Objective function values: f_sigma2=%.4f, f_eta=%.4f, f_Lambda=%.4f, and f_M=%.4f at iteration %d.\n', objf, iter);
    
    % break if condition satisfied
    if iter>1
        dif(1)=max(abs(sigma2_-sigma2)); dif(2)=max(abs(eta_-eta));
        dif(3)=max(abs(Lambda_(:)-Lambda(:))); 
        if ~isempty(M)
            dif(4)=max(abs(M_(:)-M(:))); 
        end
        if all(abs(objf_-objf)<thld) || all(dif<thld)
            fprintf('Optimization breaks at iteration %d.\n',iter);
            break;
        end
    end

end

% count time
time=toc;
fprintf('Time used: %.2f seconds.\n',time);
