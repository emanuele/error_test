import numpy as np
from numpy.random import multinomial, dirichlet

if __name__ == '__main__':

    np.random.seed(0)

    # total = np.array([20, 20, 20, 20])
    # diagonal = np.array([10, 5, 7, 8])


    C1 = np.array([[0.31, 0.31, 0.19, 0.19],
                   [0.06, 0.81, 0.00, 0.13],
                   [.13 , 0.0, .81, .06],
                   [.25, .13, .25, .38]])

    C1 = np.round(C1 * 16).astype(np.int)

    C2 = np.array([[0.4, 0.3, 0.19, 0.11],
                   [0.3, 0.41, 0.13, 0.16],
                   [.17 , 0.11, .45, .26],
                   [.15, .23, .22, .39]])

    C2 = np.round(C2 * 16).astype(np.int)


    C1 = np.array([[1, 6, 5],
                   [1, 9, 5],
                   [8, 3, 1]])

    C2 = np.array([[6, 5, 1],
                   [8, 6, 1],
                   [2, 4, 6]])

    n_classes = C1.shape[0]

    alpha = np.ones(n_classes-1)

    total1 = C1.sum(1) - C1.diagonal()
    total2 = C2.sum(1) - C2.diagonal()

    C1 = C1[np.eye(n_classes)==0].reshape(n_classes, n_classes-1)
    C2 = C2[np.eye(n_classes)==0].reshape(n_classes, n_classes-1)

    iterations = 5000000

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
