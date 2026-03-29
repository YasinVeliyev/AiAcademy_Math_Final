import numpy as np
from utils import softmax
import copy


class SoftMaxClassification:
    def __init__(self,penalty = "l2",lamda = 1e-4,learning_rate = 0.05,max_iter = 200,batch_size=32,optimizer="sgd"):
        self.penalty = penalty
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.loss = []
        self.cache = {}
        self.optimizer=optimizer
    
    def fit(self,X_train,y_train):
        self.loss = []
        self._initialize_weights(X_train,y_train)
        
        Y = np.zeros((self.n,len(self.n_class)))
        for i, val in enumerate(y_train):
            Y[i,np.where(self.n_class == val)] = 1
            
        idxs = np.arange(self.n)
        for _ in range(self.max_iter):
            np.random.shuffle(idxs)
            X_shuffled = X_train[idxs]
            Y_shuffled = Y[idxs]
            for j in range(0,self.n,self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                Y_batch = Y_shuffled[j:j+self.batch_size]
                self._forward(X_batch)
                self.back_propagation(X_batch,Y_batch)
                self.optimize()
                
            A = self._forward(X_train)   
            L =self._calculate_loss(Y,A)
            self.loss.append(L)
            
    def _initialize_weights(self,X_train,y_train):
        self.n,self.m = X_train.shape
        self.n_class = np.unique(y_train)
        self.W = np.random.randn(len(self.n_class),self.m)*np.sqrt(1/self.m) 
        self.b = np.zeros((1,len(self.n_class)))
    
    def _l2(self):
        return 0.5 * self.lamda * np.sum(np.power(self.W,2))

    def _forward(self,X_val):
        Z = X_val @ self.W.T + self.b
        self.cache["A"] = softmax(Z)
        return self.cache["A"]
         
    def back_propagation(self,X,Y):
        dLdZ = (self.cache.get("A",0) - Y)/X.shape[0]
        dZdW = X
        reg_term = self.lamda * self.W
        self.cache["dW"] =  dLdZ.T @ X+ reg_term
        self.cache["db"] = np.sum(dLdZ, axis=0, keepdims=True)
        
    def optimize(self):  
        self.W -= self.learning_rate * self.cache["dW"]
        self.b -= self.learning_rate * self.cache["db"]
        
        
    def predict(self,X_val):
        A = self._forward(X_val)
        return np.argmax(A,axis=1)

    def _calculate_loss(self,Y,y_predict):
        return -np.mean(np.sum(Y*np.log(y_predict + 1e-9),axis=1)) + self._l2()
    
    def plot_decison_boundary(self,X,y,ax=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1,1,figsize=(8,6))
        if X.shape[1] > 2:
            raise ValueError("Only for 2 dimension")
            
        w_diff = self.W[0] - self.W[1]
        b_diff = self.b[0,0] - self.b[0,1]
        X1 = np.sort(X[:,1])
        X0 = -(X1 * w_diff[1] + b_diff)/w_diff[0]
         
        ax.set_title(f"Decision Boundary Optimizer:{self.optimizer.capitalize()};\nLearning rate:{self.learning_rate};Loss:{self.loss[-1]:.4f};Iteration:{self.max_iter}")
        ax.set_ylabel("X1")
        ax.set_xlabel("X0")
        ax.scatter(X[:,0],X[:,1],c=y)
        ax.plot(X0,X1,color="red")
     


    @classmethod
    def gradient_check(cls,X,y,params = {},epsilon=1e-7):
        model = cls(**copy.deepcopy(params))
        model.fit(X,y)
        
        Y = np.zeros((model.n,len(model.n_class)))
        for i, val in enumerate(y):
            Y[i, val] = 1
        
        model._forward(X)
        model.back_propagation(X, Y)    
        
        grad_analytic = np.copy(model.cache["dW"])
        grad_num = np.zeros_like(grad_analytic)
        
        for i in range(grad_analytic.shape[0]):
            for j in range(grad_analytic.shape[1]):
                
                model.W[i,j] += epsilon
                y_plus = model._forward(X)
                L1 = model._calculate_loss(Y,y_plus)
                model.W[i,j] -= 2*epsilon
                
                y_minus = model._forward(X)
                L2 = model._calculate_loss(Y,y_minus)
                model.W[i,j] += epsilon
                
                grad_num[i,j] = (L1 - L2)/(2*epsilon)
                
        diff = np.linalg.norm(grad_analytic - grad_num) / (np.linalg.norm(grad_analytic) + np.linalg.norm(grad_num))
        print("Gradient check difference:", diff)
            
        return diff
   
    
    def plot_loss(self,ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,6))
        
        ax.set_title(f"Optimizer:{self.optimizer.capitalize()};Learning rate:{self.learning_rate}")
        ax.plot(range(len(self.loss)),self.loss)
        ax.set_ylabel("Log Loss")
        ax.set_yticks(np.linspace(min(self.loss), max(self.loss), 10))
        ax.set_xlabel("Iteration")
        ax.grid()
        
        
        
        
        