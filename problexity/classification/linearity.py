"""
Linearity measures
"""
import numpy as np
from sklearn.svm import LinearSVC

def l1(X, y):
    """
    Calculates the Sum of the error distance by linear programming (L1) metric.

    Uses Linear SVM classifier. Measures distance of incorrectly classified samples from the SVM hyperplane.

    .. math::

        L1=\\frac{SumErrorDist}{1+SumErrorDist}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels of binary classification task ([0,1])

    :rtype: float
    :returns: L1 score
    """    
    X = np.copy(X)
    y = np.copy(y)

    svm = LinearSVC().fit(X, y)
    incorrect = svm.predict(X)!=y
    
    w_norm = np.linalg.norm(svm.coef_)
    dist = np.abs(svm.decision_function(X) / w_norm)
    
    sed = np.sum(dist[incorrect])/y.shape[0]
    return sed/(1+sed)

def l2(X, y):
    """
    Calculates the Error rate of linear classifier (L2) metric.

    Returns error rate of Linear SVM classifer used within the dataset.

    .. math::

        L2=\\frac{\sum^{n}_{i=1}I(h(x_i)\\neq y_i)}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels of binary classification task ([0,1])

    :rtype: float
    :returns: L2 score
    """

    X = np.copy(X)
    y = np.copy(y)

    svm = LinearSVC()
    pred = svm.fit(X, y).predict(X)

    return np.sum(pred!=y)/y.shape[0]


def l3(X, y):
    """
    Calculates the Non linearity of linear classifier (L3) metric.

    Linearly interpolating instances of each class generate the additional instances of the problem, which are used to calculate this measure. The class of original instances determines the label of an augmented point. The Linear SVM classifier is used to classify the synthesized points of the dataset. The number of synthetic points is equal to the original dataset size.
    
    .. math::

        L3=\\frac{1}{l}\sum^{l}_{i=1}I(h_T(x'_i) \\neq y'_i)

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels of binary classification task ([0,1])

    :rtype: float
    :returns: L3 score
    """
    
    X = np.copy(X)
    y = np.copy(y)

    try:
        pairs_0 = np.array([np.random.choice(np.argwhere(y==0).flatten(), 2, replace=False) for i in range(np.sum(y==0))])
        pairs_1 = np.array([np.random.choice(np.argwhere(y==1).flatten(), 2, replace=False) for i in range(np.sum(y==1))])
    except:
        # Protect againt single class sample
        pairs_0 = np.array([np.random.choice(np.argwhere(y==0).flatten(), 2, replace=True) for i in range(np.sum(y==0))])
        pairs_1 = np.array([np.random.choice(np.argwhere(y==1).flatten(), 2, replace=True) for i in range(np.sum(y==1))])

    pairs = np.concatenate((pairs_0, pairs_1), axis=0)
    y_new = np.concatenate((np.zeros(pairs_0.shape[0]), np.ones(pairs_1.shape[0])), axis=0)
    rand = np.random.random(pairs.shape[0]).reshape(-1,1)

    distance = X[pairs[:,0]]-X[pairs[:,1]]    
    rand_distance = rand * distance
    
    X_new = X[pairs[:,1]] + rand_distance

    svm = LinearSVC()
    pred = svm.fit(X, y).predict(X_new)

    return np.sum(pred!=y_new)/y_new.shape[0]
