import numpy as np
import matplotlib.pyplot as plt


class SoftMaxClassification:
    def __init__(self,penalty = "l2",lamda = 1e-4,learning_rate = 0.05,max_iter = 5000,batch_size=64):
        self.penalty = "l2"
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.loss = []
    
    def fit(self,X_train,y_train):
        self.n,self.m = X_train.shape
        number_of_class = np.unique(y_train).size
        Y = np.eye(number_of_class)[y_train]
        self.W = np.random.randn(number_of_class,self.m+1)*0.01
        X = np.hstack((X_train,np.ones((self.n,1))))
        idxs = np.arange(self.n)
        for _ in range(self.max_iter):
            np.random.shuffle(idxs)
            X_shuffled = X[idxs]
            Y_shuffled = Y[idxs]
            for j in range(0,self.n,self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                Y_batch = Y_shuffled[j:j+self.batch_size]
                self.back_propagation(X_batch,Y_batch)
            softmax = self.softmax(X)
            L = - np.mean(np.sum(Y*np.log(softmax+1e-9),axis=1)) + self._l2()
            self.loss.append(L)
            
    def _l2(self):
        return self.lamda * np.sum(np.power(self.W[:,:-1],2))
        
    
    def softmax(self,X):
        Z = X @ self.W.T
        Z -= np.max(Z, axis=1, keepdims=True)
        S = np.exp(Z)
        return S/np.sum(S,axis=1,keepdims=True)
            
    def back_propagation(self,X,Y):
        softmax = self.softmax(X)
        dLdZ = softmax - Y
        dZdW = X
        reg_term = 2 * self.lamda * self.W
        reg_term[:,-1]= 0
        dLdW =  (dLdZ.T @ X)/X.shape[0] + reg_term
        self.W -= self.learning_rate * dLdW
        
    def predict(self,X_val):
        X_val = np.hstack((X_val,np.ones((X_val.shape[0],1))))
        result = self.softmax(X_val)
        return np.argmax(result,axis=1)
        
    def plot_loss(self):
        plt.figure(figsize=(8,6))
        plt.plot(range(self.max_iter),self.loss)
        plt.ylabel("Log Loss")
        plt.yticks(np.linspace(min(self.loss), max(self.loss), 10))
        plt.xlabel("Iteration")
        plt.grid()
        plt.show()
        