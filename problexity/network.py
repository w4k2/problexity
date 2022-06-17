import numpy as np
from scipy.spatial import distance_matrix
import gower
from igraph import Graph

def _get_graph(X, y):
    X = np.copy(X)
    y = np.copy(y)

    epsilon= 0.15
    dist = gower.gower_matrix(X)

    for i in range(y.shape[0]):
        dist[i,i]=np.nan
    
    #normalize
    dist = dist-np.nanmin(dist)
    dist = dist/np.nanmax(dist)

    # build graph
    edges = np.zeros(dist.shape)
    edges[dist<epsilon] = 1

    # rm
    dist_y = distance_matrix(y[:, np.newaxis], y[:, np.newaxis])
    other_classes = np.argwhere(dist_y==1)
    edges[other_classes[:,0], other_classes[:,1]] = 0

    return edges


def density(X, y):
    """
    Calculates the Density metric.

    .. math::

        Density =1 - \\frac{2|E|}{n(n-1)}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset.
    :type y: array-like, shape (n_samples)
    :param y: Labels.

    :rtype: float
    :returns: Density score.
    """
    
    X = np.copy(X)
    y = np.copy(y)
    
    edges = _get_graph(X, y)

    return 1-(np.sum(edges)/(y.shape[0]*(y.shape[0]-1)))


def clsCoef(X, y):
    """
    Calculates the Clustering Coefficient metric.

    .. math::

        ClsCoef=1-\\frac{1}{n}\sum^{n}_{i=1}\\frac{2|e_{jk} : v_j, v_k \in N_i|}{k_i(k_i-1)}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset.
    :type y: array-like, shape (n_samples)
    :param y: Labels.

    :rtype: float
    :returns: Clustering Coefficient score.
    """
    
    X = np.copy(X)
    y = np.copy(y)
    
    edges = _get_graph(X, y)

    sum_neighbours = np.sum(edges, axis=0)
    sum_conncetions = np.zeros(y.shape[0])

    #edges between its neighbours
    for i, item_egdes in enumerate(edges):
        neighbours = np.argwhere(item_egdes==1).flatten()

        #connections between n and other neighbours
        sum_conncetions[i] = np.sum([edges[n, neighbours] for n in neighbours])

    mask = (sum_neighbours-1)>0

    sum_neighbours = sum_neighbours[mask]
    sum_conncetions = sum_conncetions[mask]

    return 1 - (np.sum(sum_conncetions/(sum_neighbours*(sum_neighbours-1))))/y.shape[0]

def hubs(X, y):
    """
    Calculates the Hubs metric.

    .. math::

        Hubs=1-\\frac{1}{n}\sum^{n}_{i=1}hub(v_i)

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset.
    :type y: array-like, shape (n_samples)
    :param y: Labels.

    :rtype: float
    :returns: Hubs score.
    """
    
    X = np.copy(X)
    y = np.copy(y)
    
    edges = _get_graph(X, y)

    g = Graph.Adjacency(edges)
    hub = g.hub_score()
    return 1 - (np.sum(hub))/y.shape[0]