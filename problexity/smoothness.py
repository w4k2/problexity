"""
Smoothness measures
"""
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor

def s1(X, y, normalize=True):
    """
    Calculates the output distribution (S1) measure.

    Calculates complexity based on a similarity of instances adjacent in minimum spanning tree (MST). Returns the average difference of labels (y), of samples connected by MST. By default a 0-1 interval normalization is performed.

    .. math::

        S1=\\frac{1}{n}\sum_{i,j \in MST}|y_i - y_j|

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: S1 score
    """

    X = np.copy(X)
    y = np.copy(y)

    if normalize:
        for f_id in range(X.shape[1]):
            X[:,f_id] -= np.min(X[:,f_id])
            X[:,f_id] /= np.max(X[:,f_id])
        y -= np.min(y)
        y /=np.max(y)
    
    dist = distance_matrix(X, X)
    graph = csr_matrix(dist)
    mst = minimum_spanning_tree(graph)
    mst = mst.toarray()
    coordinates = np.argwhere(mst>0)

    diff = np.abs(y[coordinates][:,0] - y[coordinates][:,1])

    return np.sum(diff)/X.shape[0]

def s2(X, y, normalize=True):
    """
    Calculates the input distribution (S2) measure.

    Calculates complexity based on a similarity of features (X) of instances with close output values (y). Returns the average euclidean norm of difference of input values, of samples neighbouring after sorting them by output values. By default a 0-1 interval normalization is performed.

    .. math::

        S2=\\frac{1}{n}\sum_{i=2}^{n}||x_i-x_{i-1}||_2

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: S2 score
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

    a = X[sorted_y_idx[:-1]]
    b = X[sorted_y_idx[1:]]
    diff = np.linalg.norm(a-b, axis=1)

    return np.sum(diff)/X.shape[0]

def s3(X, y, normalize=True):
    """
    Calculates the error of nearest neighbor regressor (S3) measure.

    Returns mean squared error of a 1-nearest neighbor regressor, established during leave-one-out procedure. By default, the data in normalized with 0-1 interval normalization.

    .. math::

        S3=\\frac{1}{n}\sum_{i=1}^{n}(NN(x_i)-y_i)^2

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: S3 score
    """
    X = np.copy(X)
    y = np.copy(y)

    if normalize:
        for f_id in range(X.shape[1]):
            X[:,f_id] -= np.min(X[:,f_id])
            X[:,f_id] /= np.max(X[:,f_id])
        y -= np.min(y)
        y /=np.max(y)

    loo = LeaveOneOut()

    err = []
    for train_index, test_index in loo.split(X):
        err.append(KNeighborsRegressor(n_neighbors=1).fit(X[train_index], y[train_index]).predict(X[test_index]) - y[test_index])

    return np.sum(np.power(err,2))/X.shape[0]