import numpy as np
def softmax(X):
    Z = X -  np.max(X, axis=1, keepdims=True)
    S = np.exp(Z)
    return S/np.sum(S,axis=1,keepdims=True)


def accuracy(y_pred,y_test):
    return y_pred[y_pred==y_test].size/y_test.size