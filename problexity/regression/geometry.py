"""
Geometry, topology and density measures
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def l3(X, y, normalize=True):
    """
    Calculates the nom-linearity of a linear regressor (L3) measure.

    Linearly interpolates both input (X) and output (y) values of each pair of samples with similar output values. Generated l=n-1 synthetic samples and then measures the mean squared error of a linear regressor, fitted with original data and evaluated on synthetic points. By default performs a normalization of samples.
    
    .. math::

        L3=\\frac{1}{l}\sum_{i=1}^{l}(f(x'_i) - y'_i)^2

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: L3 score
    """

    X = np.copy(X)
    y = np.copy(y)

    if normalize:
        for f_id in range(X.shape[1]):
            X[:,f_id] -= np.min(X[:,f_id])
            X[:,f_id] /= np.max(X[:,f_id])
        y -= np.min(y)
        y /=np.max(y)

    sorted_y_idx = np.argsort(y)

    a_idx = sorted_y_idx[:-1]
    b_idx = sorted_y_idx[1:]
    pairs = np.concatenate((a_idx[:, np.newaxis], b_idx[:, np.newaxis]), axis=1)
    rand = np.random.random(pairs.shape[0]).reshape(-1,1)

    #interpolate input X
    distance = X[pairs[:,0]]-X[pairs[:,1]]
    rand_distance = rand * distance
    X_new = X[pairs[:,1]] + rand_distance

    #interpolate output y
    distance = y[pairs[:,0]]-y[pairs[:,1]]
    rand_distance = rand[:,0] * distance
    y_new = y[pairs[:,1]] + rand_distance

    err = LinearRegression().fit(X,y).predict(X_new) - y_new
    return np.sum(np.power(err, 2))/X_new.shape[0]


def s4(X, y, normalize=True):
    """
    Calculates the non-linearity of a nearest neighbor regressor (S4) measure.

    Linearly interpolates both input (X) and output (y) values of each pair of samples with similar output values. Generated l=n-1 synthetic samples and then measures the mean squared error of a nearest neighbor regessor, fitted with original data and evaluated on synthetic points. By default performs a normalization of samples.
    
    .. math::

        S4=\\frac{1}{l}\sum_{i=1}^{l}(NN(x'_i) - y'_i)^2

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: S4 score
    """

    X = np.copy(X)
    y = np.copy(y)

    if normalize:
        for f_id in range(X.shape[1]):
            X[:,f_id] -= np.min(X[:,f_id])
            X[:,f_id] /= np.max(X[:,f_id])
        y -= np.min(y)
        y /=np.max(y)

    sorted_y_idx = np.argsort(y)

    a_idx = sorted_y_idx[:-1]
    b_idx = sorted_y_idx[1:]
    pairs = np.concatenate((a_idx[:, np.newaxis], b_idx[:, np.newaxis]), axis=1)
    rand = np.random.random(pairs.shape[0]).reshape(-1,1)

    #interpolate input X
    distance = X[pairs[:,0]]-X[pairs[:,1]]
    rand_distance = rand * distance
    X_new = X[pairs[:,1]] + rand_distance

    #interpolate output y
    distance = y[pairs[:,0]]-y[pairs[:,1]]
    rand_distance = rand[:,0] * distance
    y_new = y[pairs[:,1]] + rand_distance

    err = KNeighborsRegressor().fit(X,y).predict(X_new) - y_new
    return np.sum(np.power(err, 2))/X_new.shape[0]
    
def t2(X, y):
    """
    Calculates the average number of examples per dimension (T2) measure.

    Returns number of samples per number of features. Higher values indicate simpler problems.
        
    .. math::

        T2=\\frac{n}{d}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: T2 score
    """
    return X.shape[0]/X.shape[1]