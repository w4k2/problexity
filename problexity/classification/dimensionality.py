"""
Dimensionality measures
"""
import numpy as np
from sklearn.decomposition import PCA

def t2(X, y):
    """
    Calculates the Average number of features per dimension (T2) metric. 

    To obtaint this measure, the number of dimensions describing the dataset is divided by the number of instances.

    .. math::

        T2=\\frac{m}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: T2 score
    """
    X = np.copy(X)
    return X.shape[1]/X.shape[0]

def t3(X, y):
    """
    Calculates the Average number of PCA dimensions per points (T3) metric.

    To obtain this measure, first, the number of PCA components needed to represent 95% of data variability is calculated. Then, the value is divided by the instance number in the dataset.

    .. math::

        T3=\\frac{m'}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: T3 score
    """
    X = np.copy(X)
    y = np.copy(y)
    
    pca = PCA(n_components=0.95)
    X_new = pca.fit_transform(X,y)

    return X_new.shape[1]/X.shape[0]

def t4(X, y):
    """
    Calculates the Ration of the PCA dimension to the original dimension (T4) metric.

    To obtain this measure, the number of PCA components needed to represent 95% of data variability is divided by the original number of dimensions. This measure describes the proportion of relevant dimensions in the dataset.

    .. math::

        T4=\\frac{m'}{m}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: T4 score
    """
    X = np.copy(X)
    y = np.copy(y)
    
    pca = PCA(n_components=0.95)
    X_new = pca.fit_transform(X,y)

    return X_new.shape[1]/X.shape[1]