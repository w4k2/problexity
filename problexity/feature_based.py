import numpy as np

def F1(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    X_0 = X[y==0]
    X_1 = X[y==1]

    X_0_prop = X_0.shape[0]/X.shape[0]
    X_1_prop = X_1.shape[0]/X.shape[0]

    X_0_mean = np.mean(X_0, axis = 0)
    X_1_mean = np.mean(X_1, axis = 0)

    X_0_std = np.std(X_0, axis = 0)
    X_1_std = np.std(X_1, axis = 0)

    l = (X_0_prop*X_1_prop*np.power((X_0_mean - X_1_mean),2)) + (X_1_prop*X_0_prop*np.power((X_1_mean - X_0_mean),2))
    m = (X_0_prop*(np.power(X_0_std,2))) + (X_1_prop*(np.power(X_1_std,2)))
    r_all = l/m

    return 1 / (1+np.max(r_all))

def F1v(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    X_0 = X[y==0]
    X_1 = X[y==1]

    X_0_prop = X_0.shape[0]/X.shape[0]
    X_1_prop = X_1.shape[0]/X.shape[0]

    X_0_mean = np.mean(X_0, axis = 0)
    X_1_mean = np.mean(X_1, axis = 0)

    sigm0 = np.zeros((X.shape[1],X.shape[1]))
    sigm1 = np.zeros((X.shape[1],X.shape[1]))
    
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            sigm0[i,j] = np.mean((X_0[:,i] - X_0_mean[i]) * (X_0[:,j] - X_0_mean[j]))
    
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            sigm1[i,j] = np.mean((X_1[:,i] - X_1_mean[i]) * (X_1[:,j] - X_1_mean[j]))
    
    B = np.dot((X_0_mean - X_1_mean).reshape(-1,1),(X_0_mean - X_1_mean).reshape(1,-1))

    W = sigm0*X_0_prop + sigm1*X_1_prop
    W_1 = np.linalg.pinv(W)

    d = np.dot(W_1,(X_0_mean - X_1_mean))
    dt = d.T

    df = (np.dot(np.dot(dt,B),d))/(np.dot(np.dot(dt,W),d))
    if np.isnan(df):
        df = 0

    return 1/(1+df)


def F2(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    X_0 = X[y==0]
    X_1 = X[y==1]

    minmax = np.min((np.max(X_0, axis=0), np.max(X_1, axis=0)), axis=0)   
    maxmin = np.max((np.min(X_0, axis=0), np.min(X_1, axis=0)), axis=0)
    maxmax = np.max((np.max(X_0, axis=0), np.max(X_1, axis=0)), axis=0)
    minmin = np.min((np.min(X_0, axis=0), np.min(X_1, axis=0)), axis=0)

    f_overlap = np.max((np.zeros((X.shape[1])), (minmax - maxmin)), axis=0)
    f_range = maxmax - minmin

    return np.product(f_overlap/f_range)

def F3(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    X_0 = X[y==0]
    X_1 = X[y==1]

    minmax = np.min((np.max(X_0, axis=0), np.max(X_1, axis=0)), axis=0)   
    maxmin = np.max((np.min(X_0, axis=0), np.min(X_1, axis=0)), axis=0)

    n = np.zeros((X.shape[1]))

    for i in range(X.shape[1]):
        n[i] = np.sum((X[:,i]>maxmin[i]) & (X[:,i]<minmax[i]))
    
    return np.min(n/X.shape[0])


def F4(X_input, y_input):
    
    X = np.copy(X_input)
    y = np.copy(y_input)

    T = np.zeros(X.shape[0]) #classified = 1, else 0 
    features_used = np.zeros(X.shape[1]) #used = 1, else 0

    while np.sum(T)<X.shape[0] and np.sum(features_used)<X.shape[1]:
        m1 = T==0
        m2 = y==0
        m3 = y==1

        X_0_temp = X[m1 & m2]
        X_1_temp = X[m1 & m3]
        # not yet classified
        if X_0_temp.shape[0]==0 or X_1_temp.shape[0]==0:
            break

        minmax_temp = np.min((np.max(X_0_temp, axis=0), np.max(X_1_temp, axis=0)), axis=0)   
        maxmin_temp = np.max((np.min(X_0_temp, axis=0), np.min(X_1_temp, axis=0)), axis=0)
      
        n_mask = np.zeros((X.shape)).astype(bool)
        n_mask[T==1,:] = 1

        for i in range(X.shape[1]):
            a = X[:,i]<maxmin_temp[i]
            b = X[:,i]>minmax_temp[i] 
            # not overlaping

            n_mask[:,i] = a | b

        discriminance = np.sum(n_mask, axis=0)
        most_discriminant = np.flip(np.argsort(discriminance))

        for i in most_discriminant:
            if features_used[i]==0:
                features_used[i] = 1
                T[n_mask[:,i]] = 1
                break

    return np.sum(T==0)/y.shape[0]