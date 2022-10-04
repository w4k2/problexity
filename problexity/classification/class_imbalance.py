"""
Class imbalance measures
"""

import numpy as np

def c1(X, y):
    """
    Calculates the Entropy of Class Proportions (C1) metric.

    .. math::

        C1=1+\\frac{1}{log(n_c)}\sum^{n_c}_{i=1}p_{c_{i}}log(p_{c_{i}})

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: C1 score
    """
    y = np.copy(y)

    classes = np.unique(y).shape[0]
    p = np.zeros((classes))
    for c in range(classes):
        p[c] = np.sum(y==c)/y.shape[0]

    return 1+(1/np.log(classes))*np.sum(p*np.log(p))

def c2(X, y):
    """
    Calculates the Imbalance Ratio (C2) metric.

    .. math::

        C2=1-\\frac{1}{IR}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: C2 score
    """
    y = np.copy(y)

    classes = np.unique(y).shape[0]
    p = np.zeros((classes))
    for c in range(classes):
        p[c] = np.sum(y==c)/np.sum(y!=c)

    ir = 0.5 * np.sum(p)
    return 1 - (1/ir)