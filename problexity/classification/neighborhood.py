"""
Neighborhood measures
"""
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph

def n1(X, y):
    """
    Calculates the Fraction of borderline points (N1) metric.

    The Minimum Spanning Three is generated over input instances. The measure is computed by calculating the number of edges in the MST between instances of different classes over a total number of samples. 

    .. math::

        N1=\\frac{1}{n} \sum^{n}_{i=1}I((x_i, x_j) \in MST \wedge y_i \\neq y_j)

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels of binary classification task ([0,1])

    :rtype: float
    :returns: N1 score
    """

    X = np.copy(X)
    y = np.copy(y)

    dist = distance_matrix(X, X)
    graph = csr_matrix(dist)
    mst = minimum_spanning_tree(graph)
    mst = mst.toarray()
    coordinates = np.argwhere(mst>0)

    return (np.sum(np.sum(y[coordinates], axis=1)==1)/2)/y.shape[0]

def n2(X, y):
    """
    Calculates the Ratio of intra/extra class NN distance (N2) metric.

    The measure depends on the distances of each problem instance to its nearest neighbor of the same class and the distance to the nearest neighbor of a different class. According to the proportions of those values, the final measure is calculated.

    .. math::

        N2=\\frac{infra\_extra}{1+infra\_extra}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels of binary classification task ([0,1])

    :rtype: float
    :returns: N2 score
    """

    X = np.copy(X)
    y = np.copy(y)

    X_0 = X[y==0]
    X_1 = X[y==1]

    try:
        infra_0 = distance_matrix(X_0, X_0)
        infra_1 = distance_matrix(X_1, X_1)

        extra_0 = distance_matrix(X_0, X_1)
        extra_1 = distance_matrix(X_1, X_0)
    except:
        return np.nan

    infra_extra = (np.sum(infra_0)+np.sum(infra_1))/(np.sum(extra_0)+np.sum(extra_1))
    return infra_extra/(1+infra_extra)


def n3(X, y):
    """
    Calculates the Error rate of NN classifier (N3) metric.

    Measure is determined by the error rate of the One Nearest Neighbor Classifier in the Leave One Out evaluation protocol.

    .. math::

        N3=\\frac{\sum^{n}_{i=1}I(NN(x_i) \\neq y_i)}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels of binary classification task ([0,1])

    :rtype: float
    :returns: N3 score
    """
    X = np.copy(X)
    y = np.copy(y)

    neighbors = kneighbors_graph(X, n_neighbors=1, mode='connectivity', include_self=False).nonzero()[1]

    correct = y == y[neighbors]
    acc = np.sum(correct)/len(correct)

    return 1-acc

def n4(X, y):
    """
    Calculates the Nonlinearity of NN classifier (N4) metric.

    The measure is determined by the error rate of k - Nearest Neighbor Classifier on synthetic points, generated by linearly interpolating original instances. The Classifier is fitted on original points and evaluated on synthetic instances.

    .. math::

        N4=\\frac{1}{l}\sum^{l}_{i=1}I(NN_T(x'_i) \\neq y'_i)

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels of binary classification task ([0,1])

    :rtype: float
    :returns: N4 score
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

    knn = KNeighborsClassifier()
    pred = knn.fit(X, y).predict(X_new)

    return np.sum(pred!=y_new)/y_new.shape[0]

def _find_radius(D, i):
    j = np.argmin(D[i])
    di = D[i, j]
    k = np.argmin(D[j])
    if i == k:
        return di/2
    else:
        dt = _find_radius(D, j)
        return di - dt

def t1(X, y):
    """
    Calculates the Fraction of hyperspheres covering data (T1) metric.

    The measure is described by the number of hyperspheres needed to cover the data divided by a number of instances. First, a hypersphere is generated for each problem sample. A sample lies in the center of the hypersphere. Its radius is dependent on the distance to the instance of another class. The hyperspheres are eliminated if a different hypersphere already covers the center instance. The elimination starts from the hyperspheres with the largest radiuses and continues to the ones with smaller radiuses. The hyperspheres that were not eliminated are taken into account during the calculation of complexity.
    
    .. math::

        T1=\\frac{\#Hyperspheres(T)}{n}

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels of binary classification task ([0,1])

    :rtype: float
    :returns: T1 score
    """

    X = np.copy(X)
    y = np.copy(y)

    dist = distance_matrix(X, X)
    dist_y = distance_matrix(y[:, np.newaxis], y[:, np.newaxis])

    same_classes = np.argwhere(dist_y==0)
    dist[same_classes[:,0], same_classes[:,1]] = np.inf

    radiuses = np.zeros((y.shape[0]))

    for x_id in range(y.shape[0]):
        radiuses[x_id] = _find_radius(dist, x_id)

    #start with largest radiuses
    ordered_r = np.flip(np.argsort(radiuses))
    covered = np.zeros((X.shape[0]))

    hyper = 0
    for r_id in ordered_r:

        class_y = y[r_id]
        radius = radiuses[r_id]
        point = X[r_id]

        if np.isnan(radius): #already absorbed
            continue

        dist_to_point = distance_matrix(X, point[np.newaxis, :])
        dist_to_point[y!=class_y] = np.inf

        dist_to_point = dist_to_point.flatten()
        absorbed = np.argwhere(dist_to_point<radius)

        hyper+=1

        covered[absorbed] = True
        covered[r_id] = True

        radiuses[absorbed] = np.nan

        if False not in covered:
            break

    return hyper/X.shape[0]

def lsc(X, y):
    """
    Calculates the Local set average cardinality (LSC) metric.

    The measure is dependent on the distances between instances and the distances to the instances' nearest enemies – the nearest sample of the opposite class. The number of cases that lie closer to the sample than its closest enemy is taken into account during calculation. 
    
    .. math::

        LSC=1-\\frac{1}{n_2}\sum^{n}_{i=1} |LS(x_i)|

    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels of binary classification task ([0,1])

    :rtype: float
    :returns: LSC score
    """

    X = np.copy(X)
    y = np.copy(y)

    dist = distance_matrix(X, X)
    dist_y = distance_matrix(y[:, np.newaxis], y[:, np.newaxis])

    same_classes = np.argwhere(dist_y==0)
    dist_enemies = np.copy(dist)
    dist_enemies[same_classes[:,0], same_classes[:,1]] = np.inf

    nearest_enemies = np.argmin(dist_enemies, axis=0)
    ne_dist = dist[np.arange(y.shape[0]), nearest_enemies]

    ls = np.argwhere(dist<ne_dist)
    return 1 - (ls.shape[0]/np.power(y.shape[0],2))