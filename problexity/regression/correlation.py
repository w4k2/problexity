
"""
Feature correlation measures
"""
from scipy.stats import spearmanr
import numpy as np
from sklearn.linear_model import LinearRegression

def c1(X, y):
    """
    Calculates the maximum feature correlationto the output (C1) metric. 

    Measure returns maximum value out of all feature-output Spearman correlation absolute value. Higher values indicate simpler problems.

    .. math::

        C1=max_{j=1,..,d}|\\rho(x^j, y)|

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: C1 score
    """
    corr = np.zeros((X.shape[1]))
    for f_id in range(X.shape[1]):
        corr[f_id] = np.abs(spearmanr(X[:,f_id], y).correlation)
    
    return np.max(corr)

def c2(X, y):
    """
    Calculates the average feature correlationto the output (C2) metric. 

    Measure returns average value of all feature-output Spearman correlation absolute value. Higher values indicate simpler problems.

    .. math::

        C2=\sum^{d}_{j=1}\\frac{|\\rho(x^j, y)|}{d}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: C2 score
    """
    corr = np.zeros((X.shape[1]))
    for f_id in range(X.shape[1]):
        corr[f_id] = np.abs(spearmanr(X[:,f_id], y).correlation)
    
    return np.sum(corr)/X.shape[1]

def c3(X, y):
    """
    Calculates the individual feature efficiency (C3) metric. 

    Measure is calculated based on a number of examples that have to be removed in order to obtain a high correlation value. Removes samples based on residual value of linear regression model. The iterations limit of 1000 was introduced.

    .. math::

        C3=min_{j=1}^{d}\\frac{n^j}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: C3 score
    """
    high_correlation_treshold = 0.9
    limit = 1000

    njn = np.zeros((X.shape[1]))

    _X = np.copy(X)
    _y = np.copy(y)

    for f_id in range(X.shape[1]):
        linreg = LinearRegression().fit(_X[:,f_id].reshape(-1,1),_y)
        y_pred = linreg.predict(_X[:,f_id].reshape(-1,1))

        residuals = np.abs(_y - y_pred)
        mask = np.ones_like(residuals).astype(bool)

        f_correlation = np.abs(spearmanr(_X[:,f_id], _y).correlation)
        cnt = 0
        while(f_correlation<=high_correlation_treshold):
            to_remove = np.argmax(residuals)
            residuals[to_remove]=-np.inf
            mask[to_remove]=0

            f_correlation = np.abs(spearmanr(_X[:,f_id][mask], _y[mask]).correlation)
            if cnt==limit:
                print('Breaking C3 loop due to iterations limit')
                break
           
        num_removed = np.sum(mask==0)
        njn[f_id] = num_removed/X.shape[0]
    
    return np.min(njn)

  
def c4(X, y, normalize = True):
    """
    Calculates the collective feature efficiency (C4) metric. 

    It sequentially analyzes the features with the greatest correlation to the output until all the features are used or all instances are removed. Samples with low resudual value are removed. A metric is computed based on the number of samples remaining after removal procedure. By default, 0-1 interval normalization is used. The iterations limit of 1000 was introduced.

    .. math::

        C4=\\frac{\#\{x_i||\epsilon_i|>0.1\}_{T_l}}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels

    :rtype: float
    :returns: C4 score
    """
    treshold = 0.1
    features_used = np.zeros((X.shape[1])).astype(bool)
    corr = np.zeros((X.shape[1]))
    limit = 1000

    _X = np.copy(X)
    _y = np.copy(y)

    if normalize:
        for f_id in range(X.shape[1]):
            _X[:,f_id] -=np.min(_X[:,f_id])
            _X[:,f_id] /=np.max(_X[:,f_id])
        _y -= np.min(_y)
        _y /=np.max(_y)

    cnt = 0
    while (np.sum(features_used)<X.shape[1]) and len(_y)>0:

        for f_id in range(X.shape[1]):
            corr[f_id] = np.abs(spearmanr(_X[:,f_id], _y).correlation)
        corr[features_used]=-np.inf

        curr_f = np.argmax(corr)
        features_used[curr_f] = 1

        linreg = LinearRegression().fit(_X[:,curr_f].reshape(-1,1),_y)
        y_pred = linreg.predict(_X[:,curr_f].reshape(-1,1))

        residuals = np.abs(_y - y_pred)

        small_residuals_mask = np.zeros_like(residuals)
        small_residuals_mask[residuals<=treshold] = 1

        remaining_n = np.sum(small_residuals_mask==False)

        _X = _X[small_residuals_mask==False]
        _y = _y[small_residuals_mask==False]

        if cnt==limit:
            print('Breaking C4 loop due to iterations limit')
            break

    return remaining_n/X.shape[0]