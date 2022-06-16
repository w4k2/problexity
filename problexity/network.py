import numpy as np
from scipy.spatial import distance_matrix
# import gower
# from igraph import Graph

def density(X_input, y_input):

    epsilon= 0.15
    
    X = np.copy(X_input)
    y = np.copy(y_input)
    
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

    return 1-(np.sum(edges)/(y.shape[0]*(y.shape[0]-1)))


def clsCoef(X_input, y_input):

    epsilon= 0.15
    
    X = np.copy(X_input)
    y = np.copy(y_input)
    
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

    sum_neighbours = np.sum(edges, axis=0)
    sum_conncetions = np.zeros(y.shape[0])

    #edges between it's neighbours
    for i, item_egdes in enumerate(edges):
        neighbours = np.argwhere(item_egdes==1).flatten()

        #connections between n and other neighbours
        sum_conncetions[i] = np.sum([edges[n, neighbours] for n in neighbours])

    mask = (sum_neighbours-1)>0

    sum_neighbours = sum_neighbours[mask]
    sum_conncetions = sum_conncetions[mask]

    return 1 - (np.sum(sum_conncetions/(sum_neighbours*(sum_neighbours-1))))/y.shape[0]

def hubs(X_input, y_input):

    epsilon= 0.15
    
    X = np.copy(X_input)
    y = np.copy(y_input)
    
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

    g = Graph.Adjacency(edges)
    hub = g.hub_score()
    return 1 - (np.sum(hub))/y.shape[0]