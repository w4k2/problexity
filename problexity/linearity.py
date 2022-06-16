import numpy as np
from sklearn.svm import LinearSVC

def L1(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    svm = LinearSVC()
    pred = svm.fit(X_input, y_input).predict(X_input)

    X_mistakes = X[pred!=y]
    y_mistakes = y[pred!=y]

    fun = (np.dot(svm.coef_.flatten(),X_mistakes.T))+svm.intercept_    
    loss = np.sum( np.max((np.zeros((len(y_mistakes))), 1-(y_mistakes*fun)), axis=0))
    sed = loss/y.shape[0]
    return sed/(1+sed)

def L2(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    svm = LinearSVC()
    pred = svm.fit(X, y).predict(X)

    return np.sum(pred!=y)/y.shape[0]


def L3(X_input, y_input):
    
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

    svm = LinearSVC()
    pred = svm.fit(X, y).predict(X_new)

    return np.sum(pred!=y_new)/y_new.shape[0]