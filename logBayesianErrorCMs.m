function [logBayesFactor,log_p_same,log_p_different]  = logBayesianErrorCMs(CM1,CM2)
% [logBayesFactor,log_p_same,log_p_different]  = logBayesianErrorCMs(CM1,CM2)
%
% Computes the log Bayes Factor for the comparison of two hypotheses about 
% confusion matrices CM1 and CM2.
%   H_same: The errors (off-diagonal elements) of CM1 and CM2 come from the
%           same distribution.
%   H_different: The errors of CM1 and CM2 from different distributions.
%
% Input: CM1, CM2 - confusion matrices with absolute counts of cases
%        CM1 and CM2 must contain only non-negative integers, they must be
%        square and have the same size.
%
% Output: logBayesFactor: log Bayes Factor for comparing H_same and H_different.
%                         computed as: logBayesFactor = log_p_same - log_p_different;
%         log_p_same: log likelihood of H_same
%         log_p_different: log likelihood of H_different.
%
%   All outputs are in natural logarithm. To get log to the base N (e.g., 10), 
%   simply divide results by log(N).
%

%
% This is the Matlab implementation of the The Bayesian Test for comparing
% classifier errors described in detail in:
%
% Emanuele Olivetti and Dirk B. Walther (2015) A Bayesioan Test for
% Comparing CLassifier Errors, Proceedings of the 3rd International 
% Workshop on Pattern Recognition in NeuroImaging. Stanford, CA.
%
% Please cite this article if you use the method.
%
% Copyright (2015)
% Emanuele Olivetti (olivetti@fbk.eu)
% Dirk Bernhardt-Walther (bernhardt-walther@psych.utoronto.ca)
%

  % if run without input arguments, run with examples from the paper
  if nargin == 0
    fprintf('Usage: [logBayesFactor,log_p_same,log_p_different]  = logBayesianErrorCMs(CM1,CM2).\n\n');
    fprintf('Running with examples from the Olivetti and Walther (2015) paper:\n');
    CM1 = [[5,5,3,3];[1,13,0,2];[2,0,13,1];[4,2,4,6]]
    CM2 = [[6,5,3,2];[5,7,2,3];[3,2,7,4];[2,4,4,6]]
  end
  
  % check inputs
  if any(abs(round(CM1(:))-CM1(:)) > 0) | any(abs(round(CM2(:))-CM2(:)))
    error('Confusion matrices must be integer counts.');
  end
  if any(CM1(:) < 0) | any(CM2(:) < 0)
    error('Confusion matrices must be non-negative.');
  end
  if (size(CM1,1) ~= size(CM1,2)) || (size(CM2,1) ~= size(CM2,2))
    error('Confusion matrices must be square.');
  end
  if size(CM1,1) ~= size(CM2,1)
    error('Confusion matrices must have the same size.');
  end

  % extract E matrices, which only contain errors (off-diagonal elements)
  numClasses = size(CM1,1);
  offdiagIdx = find(1-eye(numClasses));
  CM1 = CM1';
  E1 = reshape(CM1(offdiagIdx),numClasses-1,numClasses)';
  CM2 = CM2';
  E2 = reshape(CM2(offdiagIdx),numClasses-1,numClasses)';
  
  if nargin == 0
    E1,E2
  end
  
  % using a flat prior
  alpha = ones(size(E1));

  % compute log likelihood for H_same according to equation 3
  log_p_same = sum(log_coeff(E1) + log_coeff(E2) - log_coeff(E1+E2) + log_multivariate_polya(E1 + E2, alpha));

  % compute log likelihood for H_different according to equation 4
  log_p_different = sum(log_multivariate_polya(E1,alpha) + log_multivariate_polya(E2,alpha));

  % compute log Bayes Factor according to equation 10
  logBayesFactor = log_p_same - log_p_different;

  if (nargin == 0) || (nargout == 0)
    fprintf('L_same = %g\n',exp(log_p_same));
    fprintf('L_different = %g\n',exp(log_p_different));
    fprintf('BF = %g\n',exp(logBayesFactor));
  end
end

% helper function for computing the terms of the first factor in eq. 3 in log space
function c = log_coeff(x)
  c = gammaln(sum(x,2) + 1) - sum(gammaln(x + 1),2);
end

% multivariate Polya distribution in log space
function likelihood = log_multivariate_polya(x,alpha)
  N = sum(x,2);
  A = sum(alpha,2);
  likelihood = gammaln(N+1) + gammaln(A) - gammaln(N + A) + ...
               sum(gammaln(x + alpha) - gammaln(alpha) - gammaln(x+1),2);
end
