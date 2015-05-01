# -*- coding: iso-8859-15 -*-

"""Here we test the similarity of the off-diagonal elements of two
confusion matrices. Monte Carlo and exact likelihoods are computed, as
well as the Bayes factor.
"""

import numpy as np
from scipy.special import gammaln
from multivariate_polya import multivariate_polya, log_multivariate_polya_vectorized


if __name__ == '__main__':

    C1 = np.array([[ 5,  5,  3,  3],
                   [ 1, 13,  0,  2],
                   [ 2,  0, 13,  1],
                   [ 4,  2,  4,  6]])

    C2 = np.array([[ 6,  5,  3,  2],
                   [ 5,  7,  2,  3],
                   [ 3,  2,  7,  4],
                   [ 2,  4,  4,  6]])
    

    print "Bayesian test: H1 vs H2"
    print "H1: C1 & C2 off-diagonal elements come from the same distribution."
    print "H2: C1 & C2 off-diagonal elements come from two different distributions."
    print
    print "C1:"
    print C1
    print "C2:"
    print C2

    n_classes = C1.shape[0]

    alpha = np.ones(n_classes - 1)
    E1 = C1[np.eye(n_classes)==0].reshape(n_classes, n_classes-1)
    E2 = C2[np.eye(n_classes)==0].reshape(n_classes, n_classes-1)
    total1 = E1.sum(1)
    total2 = E2.sum(1)

    print
    def log_coeff(x):
        return gammaln(x.sum() + 1.0) - gammaln(x + 1).sum()

    log_p_E1E2_given_H1 = np.sum([log_coeff(E1[i]) + log_coeff(E2[i]) - log_coeff(E1[i] + E2[i]) + log_multivariate_polya_vectorized(E1[i] + E2[i], alpha) for i in range(n_classes)])
    log_p_E1E2_given_H2 = np.sum([log_multivariate_polya_vectorized(E1[i], alpha) + log_multivariate_polya_vectorized(E2[i], alpha) for i in range(n_classes)])

    print "Exact logp(E1,E2 | H1):", log_p_E1E2_given_H1
    print "Exact logp(E1,E2 | H2):", log_p_E1E2_given_H2
    logB12 = log_p_E1E2_given_H1 - log_p_E1E2_given_H2
    print "Bayes factor 1/2 : exp(%s) = %s" % (logB12, np.exp(logB12))
    logB21 = log_p_E1E2_given_H2 - log_p_E1E2_given_H1
    print "Bayes factor 2/1 : exp(%s) = %s" % (logB21, np.exp(logB21))
