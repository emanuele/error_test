# -*- coding: iso-8859-15 -*-

import numpy as np
from numpy.random import multinomial, dirichlet
from scipy.special import gamma, gammaln
from numpy import logaddexp, log

def multinomial_pmf(x, p):
    """Compute the multinomial probability mass function (pmf), aka
    likelihood. Naive implementation.

    See: http://en.wikipedia.org/wiki/Multinomial_distribution#Probability_mass_function
    """
    return gamma(x.sum() + 1.0) / gamma(x + 1).prod() * (p ** x).prod()


def multinomial_pmf_vectorized(x, p):
    """Compute the multinomial probability mass function (pmf), aka
    likelihood. Naive vectorized implementation.
    """
    x = np.atleast_2d(x)
    return gamma(x.sum(1) + 1.0) / gamma(x + 1.0).prod(1) * (p ** x).prod(1)
    

def log_multinomial_pmf(x, p):
    """Compute the log of the multinomial probability mass function
    (pmf), aka likelihood.
    """
    return gammaln(x.sum() + 1.0) - gammaln(x + 1.0).sum() + (x * log(p)).sum()


def log_multinomial_pmf_vectorized(x, p):
    """Compute the log of the multinomial probability mass function
    (pmf), aka likelihood.
    """
    x = np.atleast_2d(x)
    return gammaln(x.sum(1) + 1.0) - gammaln(x + 1.0).sum(1) + (x * log(p)).sum(1)


def multivariate_polya(x, alpha):
    """Multivariate Pólya PDF. Basic implementation.
    """
    x = np.atleast_1d(x).flatten()
    alpha = np.atleast_1d(alpha).flatten()
    assert(x.size==alpha.size)
    N = x.sum()
    A = alpha.sum()
    likelihood = factorial(N) * gamma(A) / gamma(N + A)
    # likelihood = gamma(A) / gamma(N + A)
    for i in range(len(x)):
        likelihood /= factorial(x[i])
        likelihood *= gamma(x[i] + alpha[i]) / gamma(alpha[i])
    return likelihood


if __name__ == '__main__':

    seed = 0
    iterations = 50000

    # Example 1:
    # C1 = np.array([[0.31, 0.31, 0.19, 0.19],
    #                [0.06, 0.81, 0.00, 0.13],
    #                [.13 , 0.0, .81, .06],
    #                [.25, .13, .25, .38]])

    # C1 = np.round(C1 * 16).astype(np.int)

    # C2 = np.array([[0.4, 0.3, 0.19, 0.11],
    #                [0.3, 0.41, 0.13, 0.16],
    #                [.17 , 0.11, .45, .26],
    #                [.15, .23, .22, .39]])

    # C2 = np.round(C2 * 16).astype(np.int)


    # Example 2: C1 and C2 have different patterns
    C1 = np.array([[1, 6, 5],
                   [1, 9, 5],
                   [8, 3, 1]])

    C2 = np.array([[6, 5, 1],
                   [8, 6, 1],
                   [2, 4, 6]])

    # Example 3: C1 and C2 have pretty patterns
    # C1 = np.array([[5, 6, 1],
    #                [5, 9, 1],
    #                [8, 3, 1]])

    # C2 = np.array([[6, 5, 1],
    #                [8, 6, 1],
    #                [6, 4, 2]])

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
    total1 = C1.sum(1) - C1.diagonal()
    total2 = C2.sum(1) - C2.diagonal()
    C1 = C1[np.eye(n_classes)==0].reshape(n_classes, n_classes-1)
    C2 = C2[np.eye(n_classes)==0].reshape(n_classes, n_classes-1)


    print
    print "Basic two-level MonteCarlo sampling (%d iterations)." % iterations
    np.random.seed(seed) # for reproducibility
    counter_H1 = 0
    counter_H2 = 0
    for i in range(iterations):
        theta = dirichlet(alpha, size=n_classes)
        C1_candidate_H1 = np.array([multinomial(total1[j], theta[j]) for j in range(n_classes)])
        C2_candidate_H1 = np.array([multinomial(total2[j], theta[j]) for j in range(n_classes)])

        if (C1_candidate_H1==C1).all() and (C2_candidate_H1==C2).all():
            counter_H1 += 1

        theta1 = dirichlet(alpha, size=n_classes)
        theta2 = dirichlet(alpha, size=n_classes)
        C1_candidate_H2 = np.array([multinomial(total1[j], theta1[j]) for j in range(n_classes)])
        C2_candidate_H2 = np.array([multinomial(total2[j], theta2[j]) for j in range(n_classes)])

        if (C1_candidate_H2==C1).all() and (C2_candidate_H2==C2).all():
            counter_H2 += 1

    print "counter_H1:", counter_H1
    print "counter_H2:", counter_H2

    p_C1C2_given_H1 = counter_H1 / float(iterations)
    p_C1C2_given_H2 = counter_H2 / float(iterations)
    print "Approximate p(C1,C2 | H1):", p_C1C2_given_H1
    print "Approximate p(C1,C2 | H2):", p_C1C2_given_H2    

    print
    print "Now, a more accurate MonteCarlo sampling based on multinomial_pmf (%d iterations)." % iterations
    p_C1C2_given_H1 = 0.0
    p_C1C2_given_H2 = 0.0
    np.random.seed(seed) # for reproducibility
    for i in range(iterations):
        theta = dirichlet(alpha, size=n_classes)
        tmp = np.prod([multinomial_pmf(C1[j], theta[j]) for j in range(n_classes)])
        tmp *= np.prod([multinomial_pmf(C2[j], theta[j]) for j in range(n_classes)])
        p_C1C2_given_H1 += tmp

        theta1 = dirichlet(alpha, size=n_classes)
        theta2 = dirichlet(alpha, size=n_classes)
        tmp = np.prod([multinomial_pmf(C1[j], theta1[j]) for j in range(n_classes)])
        tmp *= np.prod([multinomial_pmf(C2[j], theta2[j]) for j in range(n_classes)])
        p_C1C2_given_H2 += tmp

        
    p_C1C2_given_H1 = p_C1C2_given_H1 / float(iterations)
    p_C1C2_given_H2 = p_C1C2_given_H2 / float(iterations)
    print "Approximate p(C1,C2 | H1):", p_C1C2_given_H1
    print "Approximate p(C1,C2 | H2):", p_C1C2_given_H2    
    print "Bayes factor 1/2 :", p_C1C2_given_H1 / p_C1C2_given_H2
    print "Bayes factor 2/1 :", p_C1C2_given_H2 / p_C1C2_given_H1    


    print
    print "Now a MUCH faster implementation using vectorized code (%d iterations)." % iterations
    np.random.seed(seed) # for reproducibility
    theta = dirichlet(alpha, size=(iterations, n_classes))
    p_C1C2_given_H1 = np.prod([multinomial_pmf_vectorized(C1[j], theta[:,j,:]) for j in range(n_classes)], 0)
    p_C1C2_given_H1 *= np.prod([multinomial_pmf_vectorized(C2[j], theta[:,j,:]) for j in range(n_classes)], 0)

    theta1 = dirichlet(alpha, size=(iterations, n_classes))
    theta2 = dirichlet(alpha, size=(iterations, n_classes))
    p_C1C2_given_H2 = np.prod([multinomial_pmf_vectorized(C1[j], theta1[:,j,:]) for j in range(n_classes)], 0)
    p_C1C2_given_H2 *= np.prod([multinomial_pmf_vectorized(C2[j], theta2[:,j,:]) for j in range(n_classes)], 0)

    p_C1C2_given_H1 = p_C1C2_given_H1.mean()
    p_C1C2_given_H2 = p_C1C2_given_H2.mean()
    print "Approximate p(C1,C2 | H1):", p_C1C2_given_H1
    print "Approximate p(C1,C2 | H2):", p_C1C2_given_H2
    print "Bayes factor 1/2 :", p_C1C2_given_H1 / p_C1C2_given_H2
    print "Bayes factor 2/1 :", p_C1C2_given_H2 / p_C1C2_given_H1



    print
    print "Now a MUCH faster implementation using vectorized code in log space (%d iterations)." % iterations
    np.random.seed(seed) # for reproducibility
    theta = dirichlet(alpha, size=(iterations, n_classes))
    log_p_C1C2_given_H1 = np.sum([log_multinomial_pmf_vectorized(C1[j], theta[:,j,:]) for j in range(n_classes)], 0)
    log_p_C1C2_given_H1 += np.sum([log_multinomial_pmf_vectorized(C2[j], theta[:,j,:]) for j in range(n_classes)], 0)

    theta1 = dirichlet(alpha, size=(iterations, n_classes))
    theta2 = dirichlet(alpha, size=(iterations, n_classes))
    log_p_C1C2_given_H2 = np.sum([log_multinomial_pmf_vectorized(C1[j], theta1[:,j,:]) for j in range(n_classes)], 0)
    log_p_C1C2_given_H2 += np.sum([log_multinomial_pmf_vectorized(C2[j], theta2[:,j,:]) for j in range(n_classes)], 0)

    p_C1C2_given_H1 = np.exp(log_p_C1C2_given_H1).mean()
    p_C1C2_given_H2 = np.exp(log_p_C1C2_given_H2).mean()
    print "Approximate p(C1,C2 | H1):", p_C1C2_given_H1
    print "Approximate p(C1,C2 | H2):", p_C1C2_given_H2
    print "Bayes factor 1/2 :", p_C1C2_given_H1 / p_C1C2_given_H2
    print "Bayes factor 2/1 :", p_C1C2_given_H2 / p_C1C2_given_H1


    print
    print "...and now... the EXACT computation!"
    from multivariate_polya import multivariate_polya, log_multivariate_polya_vectorized

    def coeff(x):
        return gamma(x.sum() + 1.0) / gamma(x + 1).prod()

    p_C1C2_given_H1 = np.prod([coeff(C1[i]) * coeff(C2[i]) / coeff(C1[i] + C2[i]) * multivariate_polya(C1[i] + C2[i], alpha) for i in range(n_classes)])
    p_C1C2_given_H2 = np.prod([multivariate_polya(C1[i], alpha) * multivariate_polya(C2[i], alpha) for i in range(n_classes)])

    print "Exact p(C1,C2 | H1):", p_C1C2_given_H1
    print "Exact p(C1,C2 | H2):", p_C1C2_given_H2
    print "Bayes factor 1/2 :", p_C1C2_given_H1 / p_C1C2_given_H2
    print "Bayes factor 2/1 :", p_C1C2_given_H2 / p_C1C2_given_H1

