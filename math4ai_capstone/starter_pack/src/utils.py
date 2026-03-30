import numpy as np
import matplotlib.pyplot as plt

def softmax(X):
    Z = X -  np.max(X, axis=1, keepdims=True)
    S = np.exp(Z)
    return S/np.sum(S,axis=1,keepdims=True)


def accuracy(y_pred,y_test):
    return y_pred[y_pred==y_test].size/y_test.size


def make_bias(y,cls):
    idx = np.where(y==cls)[0]
    r_idx = np.random.choice(idx,int(np.floor(0.9*len(idx))),replace=False)
    m = np.ones_like(y,dtype=bool)
    m[r_idx]=False
    return m

def plot_hist(y,idxs):
    fig, axs = plt.subplots(1, 2,figsize=(18, 6),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    bins = np.arange(y.min()-0.5, y.max()+1.5, 1)
    axs[0].set_xticks(np.unique(y))
    axs[0].set_title("Orjinal Data")
    axs[0].hist(y,bins=bins,rwidth=0.8,color="orange")

    axs[1].set_title("Bias edilmiş data")
    axs[1].set_xticks(np.unique(y))
    axs[1].hist(y[idxs],bins=bins,rwidth=0.8,color="orange")




def test_model(model,X,y,X_val,y_val,epochs=1000):
    model._initialize_weights(X,y)
    train_loss=[]
    val_loss = []
    Y_train = np.zeros((X.shape[0],np.unique(y).size))
    for i, val in enumerate(y):
        Y_train[i, val] = 1
    
    Y_val = np.zeros((X_val.shape[0],np.unique(y_val).size))
    for i, val in enumerate(y_val):
        Y_val[i, val] = 1

    for j in range(epochs):
        A = model._forward(X)
        y_train_pred = model._predict(X) 
        train_loss.append(model._calculate_loss(Y_train,y_train_pred))
        
        y_val_pred = model._predict(X_val)
        val_loss.append(model._calculate_loss(Y_val,y_val_pred))
        
        model.back_propagation(X,Y_train)
        model._optimize()
    arr = np.hstack([train_loss,val_loss])

    # plt.ylim(0,np.quantile(arr,0.95)*1.1)
    # plt.yticks(np.linspace(arr.min(),np.quantile(arr,0.95),10))
    plt.plot(range(len(train_loss)),train_loss,label="Train Loss")
    plt.plot(range(len(val_loss)),val_loss,label="Validation Loss")
    plt.legend()
    plt.grid()
    print(f"Train Accuracy:{accuracy(model.predict(X),y):.4f}\nValidation Accuracy:{accuracy(model.predict(X_val),y_val):.4f}\n")