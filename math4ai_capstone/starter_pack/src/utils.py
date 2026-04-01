import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_pred,y_test):
    return np.mean(y_pred==y_test)


def softmax(X):
    Z = X -  np.max(X, axis=1, keepdims=True)
    S = np.exp(Z)
    return S/np.sum(S,axis=1,keepdims=True)
    
def one_hot_encoder(y,labels):
    Y = np.zeros((y.shape[0],labels.size))
    for i, val in enumerate(y):
        idx, = np.where(labels==val)
        Y[i,idx] = 1
    return Y

def plot_hist(y,idxs,axs):
    bins = np.arange(y.min()-0.5, y.max()+1.5, 1)
    axs[0].set_xlabel("Labels")
    axs[0].set_ylabel("Count")
    axs[0].set_xticks(np.unique(y))
    axs[0].set_title("Orjinal Data")
    axs[0].hist(y,bins=bins,rwidth=0.8,color="orange")

    axs[1].set_xlabel("Labels")
    axs[1].set_ylabel("Count")
    axs[1].set_title("Bias edilmiş data")
    axs[1].set_xticks(np.unique(y))
    axs[1].hist(y[idxs],bins=bins,rwidth=0.8,color="orange")

def make_bias(y,cls):
    idx = np.where(y==cls)[0]
    r_idx = np.random.choice(idx,int(np.floor(0.95*len(idx))),replace=False)
    m = np.ones_like(y,dtype=bool)
    m[r_idx]=False
    return m