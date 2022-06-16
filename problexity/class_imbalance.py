import numpy as np

def C1(X_input, y_input):

    X = np.copy(X_input)
    y = np.copy(y_input)

    classes = 2

    p = np.zeros((classes))
    for c in range(classes):
        p[c] = np.sum(y==c)/y.shape[0]

    return 1+(1/np.log(classes))*np.sum(p*np.log(p))

def C2(X_input, y_input):

    X = np.copy(X_input)
    y = np.copy(y_input)

    classes = 2
    p = np.zeros((classes))

    for c in range(classes):
        p[c] = np.sum(y==c)/np.sum(y!=c)

    ir = 0.5 * np.sum(p)
    return 1 - (1/ir)