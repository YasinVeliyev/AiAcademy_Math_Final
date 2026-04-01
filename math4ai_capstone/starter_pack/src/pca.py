import numpy as np
import matplotlib.pyplot as plt




def plot_scree(X_d_train):
    X_centered = X_d_train - np.mean(X_d_train,axis=0,keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance = (S**2) / (X_centered.shape[0] - 1)
    total_var = np.sum(explained_variance)
    var_ratio = explained_variance / total_var

    plt.title("Scree plot")
    plt.plot(np.cumsum(var_ratio))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.savefig("scree_plot.png", bbox_inches="tight", dpi=150)
    plt.show()


def pca_dimensions(X_train):
    X_mean = np.mean(X_train,axis=0,keepdims=True)
    X_centered = X_train - X_mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_mean,X_centered,Vt




    
