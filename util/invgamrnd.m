% invgamrnd(A,B) returns an array of random numbers chosen from the
% inverse gamma distribution with shape parameter A and scale parameter B.  
% The size of R is the common size of A and B if both are arrays.  If
% either parameter is a scalar, the size of R is the size of the other
% parameter.
% 
% R = invgamrnd(A,B,M,N,...) or R = invgamrnd(A,B,[M,N,...]) returns an
% M-by-N-by-... array.
%
% Shiwei Lan @STAT-UIUC 2019, shiwei@illinois.edu

function R = invgamrnd(A,B,varargin)

logB=log10(abs(B));
ordB=floor(logB);
scaledB=10.^(logB-ordB);

R = 10.^ordB./gamrnd(A,1./scaledB,varargin{:});

end