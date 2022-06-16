"""
Class imbalance measures.
"""

import numpy as np

def C1(X_input, y_input):
    """
    Calculates the Entropy of Class Proportions (C1) metric.

    .. math::

        C1=-\\frac{1}{log(n_c)}\sum_{i=1}^{n_c}p_{c_i}log(p_{c_i})

    :type X_input: array-like, shape (n_samples, n_features)
    :param X_input: Dataset.
    :type y_input: array-like, shape (n_samples)
    :param y_input: Labels.

    :rtype: float
    :returns: C1 score.
    """
    y = np.copy(y_input)

    classes = np.unique(y).shape[0]
    p = np.zeros((classes))
    for c in range(classes):
        p[c] = np.sum(y==c)/y.shape[0]

    return 1+(1/np.log(classes))*np.sum(p*np.log(p))

def C2(X_input, y_input):
    """
    Calculates the Imbalance Ratio (C2) metric.

    .. math::

        C2=-1-\\frac{1}{IR}

    :type X_input: array-like, shape (n_samples, n_features)
    :param X_input: Dataset.
    :type y_input: array-like, shape (n_samples)
    :param y_input: Labels.

    :rtype: float
    :returns: C2 score.
    """
    y = np.copy(y_input)

    classes = np.unique(y).shape[0]
    p = np.zeros((classes))
    for c in range(classes):
        p[c] = np.sum(y==c)/np.sum(y!=c)

    ir = 0.5 * np.sum(p)
    return 1 - (1/ir)