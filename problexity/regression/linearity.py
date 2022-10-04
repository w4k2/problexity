"""
Linearity measures for regression task
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def l1(X, y, normalize=True):
    """
    Calculates the mean absolute error (L1) metric. 

    Measure returns average error of linear regression model. By default performs a 0-1 interval normalization.

    .. math::

        L1=\sum_{i=1}^{n}\\frac{|\epsilon_i|}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: L1 score
    """
    if normalize:
        X = np.copy(X)
        y = np.copy(y)
        for f_id in range(X.shape[1]):
            X[:,f_id] -= np.min(X[:,f_id])
            X[:,f_id] /= np.max(X[:,f_id])
        y -= np.min(y)
        y /=np.max(y)

    linreg = LinearRegression().fit(X,y)
    pred = linreg.predict(X)
    residuals = np.abs(y-pred)

    return np.sum(residuals)/X.shape[0]

def l2(X, y, normalize = True):
    """
    Calculates the residuals variance (L2) metric. 

    Measure returns average of squared residuals of linear regression model. By default performs a 0-1 interval normalization.

    .. math::

        L2=\sum_{i=1}^{n}\\frac{\epsilon_i^2}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: L2 score
    """
    if normalize:
        X = np.copy(X)
        y = np.copy(y)
        for f_id in range(X.shape[1]):
            X[:,f_id] -= np.min(X[:,f_id])
            X[:,f_id] /= np.max(X[:,f_id])
        y -= np.min(y)
        y /=np.max(y)

    linreg = LinearRegression().fit(X,y)
    pred = linreg.predict(X)
    residuals = y-pred

    return np.sum(np.power(residuals,2))/X.shape[0]