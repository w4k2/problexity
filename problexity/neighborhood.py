import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

def N1(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)
    
    dist = distance_matrix(X, X)
    graph = csr_matrix(dist)
    mst = minimum_spanning_tree(graph)
    mst = mst.toarray()
    coordinates = np.argwhere(mst>0)

    return (np.sum(np.sum(y[coordinates], axis=1)==1)/2)/y.shape[0]

def N2(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    X_0 = X[y==0]
    X_1 = X[y==1]
    
    infra_0 = distance_matrix(X_0, X_0)
    infra_1 = distance_matrix(X_1, X_1)

    extra_0 = distance_matrix(X_0, X_1)
    extra_1 = distance_matrix(X_1, X_0)

    infra_extra = (np.sum(infra_0)+np.sum(infra_1))/(np.sum(extra_0)+np.sum(extra_1))
    return infra_extra/(1+infra_extra)


def N3(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    loo = LeaveOneOut()

    correct = []
    for train_index, test_index in loo.split(X):
        correct.append(KNeighborsClassifier(n_neighbors=1).fit(X[train_index], y[train_index]).predict(X[test_index]) == y[test_index])

    acc = np.sum(np.array(correct))/len(correct)
    return 1-acc


def N4(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    pairs_0 = np.array([np.random.choice(np.argwhere(y==0).flatten(), 2, replace=False) for i in range(np.sum(y==0))])
    pairs_1 = np.array([np.random.choice(np.argwhere(y==1).flatten(), 2, replace=False) for i in range(np.sum(y==1))])

    pairs = np.concatenate((pairs_0, pairs_1), axis=0)
    y_new = np.concatenate((np.zeros(pairs_0.shape[0]), np.ones(pairs_1.shape[0])), axis=0)
    rand = np.random.random(pairs.shape[0]).reshape(-1,1)

    distance = X[pairs[:,0]]-X[pairs[:,1]]    
    rand_distance = rand * distance
    
    X_new = X[pairs[:,1]] + rand_distance

    knn = KNeighborsClassifier()
    pred = knn.fit(X, y).predict(X_new)

    return np.sum(pred!=y_new)/y_new.shape[0]


def find_radius(D, i):
    j = np.argmin(D[i])
    di = D[i, j]
    k = np.argmin(D[j])
    if i == k:
        return di/2
    else:
        dt = find_radius(D, j)
        return di - dt

def T1(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)
    
    dist = distance_matrix(X, X)
    dist_y = distance_matrix(y[:, np.newaxis], y[:, np.newaxis])

    same_classes = np.argwhere(dist_y==0)
    dist[same_classes[:,0], same_classes[:,1]] = np.inf

    radiuses = np.zeros((y.shape[0]))

    for x_id in range(y.shape[0]):
        radiuses[x_id] = find_radius(dist, x_id)

    #start with largest radiuses
    ordered_r = np.flip(np.argsort(radiuses))
    covered = np.zeros((X.shape[0]))

    hyper = 0
    for r_id in ordered_r:

        class_y = y[r_id]
        radius = radiuses[r_id]
        point = X[r_id]

        if radius == 0: #already absorbed
            continue

        dist_to_point = distance_matrix(X, point[np.newaxis, :])
        dist_to_point[y!=class_y] = np.inf

        dist_to_point = dist_to_point.flatten()
        absorbed = np.argwhere(dist_to_point<radius)

        hyper+=1

        covered[absorbed] = True
        covered[r_id] = True

        radiuses[absorbed] = 0

        if False not in covered:
            break
    
    return hyper/X.shape[0]

def LSC(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)
    
    dist = distance_matrix(X, X)
    dist_y = distance_matrix(y[:, np.newaxis], y[:, np.newaxis])

    same_classes = np.argwhere(dist_y==0)
    dist_enemies = np.copy(dist)
    dist_enemies[same_classes[:,0], same_classes[:,1]] = np.inf

    nearest_enemies = np.argmin(dist_enemies, axis=0)
    ne_dist = dist[np.arange(y.shape[0]), nearest_enemies]

    ls = np.argwhere(dist<ne_dist)    
    return 1 - (ls.shape[0]/np.power(y.shape[0],2))