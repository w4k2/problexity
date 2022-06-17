"""
Dimensionality measures
"""
import numpy as np
from sklearn.decomposition import PCA

def T2(X, y):
    """
    Calculates the Average number of features per dimension (T2) metric.

    .. math::

        T2=\\frac{m}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset.
    :type y: array-like, shape (n_samples)
    :param y: Labels.

    :rtype: float
    :returns: T2 score.
    """
    X = np.copy(X)
    return X.shape[1]/X.shape[0]

def T3(X, y):
    """
    Calculates the Average number of PCA dimensions per points (T3) metric.

    .. math::

        T3=\\frac{m'}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset.
    :type y: array-like, shape (n_samples)
    :param y: Labels.

    :rtype: float
    :returns: T3 score.
    """
    X = np.copy(X)
    y = np.copy(y)
    
    pca = PCA(n_components=0.95)
    X_new = pca.fit_transform(X,y)

    return X_new.shape[1]/X.shape[0]

def T4(X, y):
    """
    Calculates the Ration of the PCA dimension to the original dimension (T4) metric.

    .. math::

        T4=\\frac{m'}{m}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset.
    :type y: array-like, shape (n_samples)
    :param y: Labels.

    :rtype: float
    :returns: T4 score.
    """
    X = np.copy(X)
    y = np.copy(y)
    
    pca = PCA(n_components=0.95)
    X_new = pca.fit_transform(X,y)

    return X_new.shape[1]/X.shape[1]