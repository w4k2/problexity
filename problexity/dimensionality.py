"""
Dimensionality measures
"""
import numpy as np
from sklearn.decomposition import PCA

def T2(X_input, y_input):
    """
    Calculates the Average number of features per dimension (T2) metric.

    .. math::

        C1=-\\frac{1}{log(n_c)}\sum_{i=1}^{n_c}p_{c_i}log(p_{c_i})

    :type X_input: array-like, shape (n_samples, n_features)
    :param X_input: Dataset.
    :type y_input: array-like, shape (n_samples)
    :param y_input: Labels.

    :rtype: float
    :returns: T2 score.
    """
    X = np.copy(X_input)
    return X.shape[1]/X.shape[0]

def T3(X_input, y_input):
    """
    Calculates the Average number of PCA dimensions per points (T3) metric.

    .. math::

        C1=-\\frac{1}{log(n_c)}\sum_{i=1}^{n_c}p_{c_i}log(p_{c_i})

    :type X_input: array-like, shape (n_samples, n_features)
    :param X_input: Dataset.
    :type y_input: array-like, shape (n_samples)
    :param y_input: Labels.

    :rtype: float
    :returns: T3 score.
    """
    X = np.copy(X_input)
    y = np.copy(y_input)
    
    pca = PCA(n_components=0.95)
    X_new = pca.fit_transform(X,y)

    return X_new.shape[1]/X.shape[0]

def T4(X_input, y_input):
    """
    Calculates the Ration of the PCA dimension to the original dimension (T4) metric.

    .. math::

        C1=-\\frac{1}{log(n_c)}\sum_{i=1}^{n_c}p_{c_i}log(p_{c_i})

    :type X_input: array-like, shape (n_samples, n_features)
    :param X_input: Dataset.
    :type y_input: array-like, shape (n_samples)
    :param y_input: Labels.

    :rtype: float
    :returns: T4 score.
    """
    X = np.copy(X_input)
    y = np.copy(y_input)
    
    pca = PCA(n_components=0.95)
    X_new = pca.fit_transform(X,y)

    return X_new.shape[1]/X.shape[1]