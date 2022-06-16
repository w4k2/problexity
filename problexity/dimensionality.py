import numpy as np
from sklearn.decomposition import PCA

def T2(X_input, y_input):

    X = np.copy(X_input)
    return X.shape[1]/X.shape[0]

def T3(X_input, y_input):

    X = np.copy(X_input)
    y = np.copy(y_input)
    
    pca = PCA(n_components=0.95)
    X_new = pca.fit_transform(X,y)

    return X_new.shape[1]/X.shape[0]

def T4(X_input, y_input):

    X = np.copy(X_input)
    y = np.copy(y_input)
    
    pca = PCA(n_components=0.95)
    X_new = pca.fit_transform(X,y)

    return X_new.shape[1]/X.shape[1]