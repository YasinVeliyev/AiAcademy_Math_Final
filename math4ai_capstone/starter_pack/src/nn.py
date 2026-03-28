import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import softmax

class NeuralNetwork():
    def __init__(self,size=None,lamda = 1e-4,batch_size=32,learning_rate = 0.05,optimizer="SGD",epochs = 200,activation_functions = (np.tanh,softmax)):
        self.size = size if size else []
        self.batch_size = batch_size
        self.learning_rate=learning_rate
        self.cache = {}
        self.optimizer=optimizer
        self.lamda = lamda
        self.epochs = epochs
        self.loss=[]
        self.activation_functions = activation_functions
       
        
    
    def fit(self,X_train,y_train):
        self._initialize_weights(X_train,y_train)
        
        Y = np.zeros((self.n,len(self.n_class)))
        for i, val in enumerate(y_train):
            Y[i,np.where(self.n_class == val)] = 1
            
        idxs = np.arange(self.n)
        for e in range(self.epochs):
            np.random.shuffle(idxs)
            X_shuffled = X_train[idxs]
            Y_shuffled = Y[idxs]
            for j in range(0,self.n,self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                Y_batch = Y_shuffled[j:j+self.batch_size]
                
                self._forward(X_batch)
                self.back_propagation(X_batch,Y_batch)
                self._optimize()
                


            y_pred = self._predict(X_train)
            L = self._calculate_loss(Y,y_pred)
            self.loss.append(L)
            
    def _initialize_weights(self,X_train,y_train):
        self.n,self.m = X_train.shape
        self.n_class = np.unique(y_train)
        
        size = [self.m] + self.size + [len(self.n_class)]
        
        self.w = [np.random.randn(size[i+1],size[i])*np.sqrt(1/size[i]) for i in range(len(size)-1)]
        self.b = [np.zeros((1,size[i+1])) for i in range(len(size)-1)]
    
    def _forward(self,X_batch):
        A = X_batch
        for i,(w,b) in enumerate(zip(self.w,self.b)):
            activation = self.activation_functions[i]
            A = activation(A @ w.T + b)
            self.cache[f"A{i}"] = A

    def _calculate_loss(self,Y,y_predict):
        return -np.mean(np.sum(Y*np.log(y_predict + 1e-9),axis=1)) + self._l2()
    
    def _optimize(self):
        if self.optimizer=="SGD":
            self.sgd()
        elif self.optimizer=="adam":
            self.adam()
        else:
            self.momentum()
        
    def back_propagation(self,X,Y):
        dLdZ= (self.cache["A1"] - Y)/Y.shape[0]
        dZdW2 = self.cache["A0"]
        self.cache["dW2"] = dLdZ.T @ dZdW2 + self.lamda*self.w[1]
        self.cache["db2"] = np.sum(dLdZ, axis=0, keepdims=True)
        
        dLdH =  dLdZ @ self.w[1]
        dLdK = dLdH * (1 - self.cache["A0"]**2)
        self.cache["dW1"] = dLdK.T @ X + self.lamda * self.w[0]
        self.cache["db1"] = np.sum(dLdK, axis=0, keepdims=True)
        
        # print("softmax",self.cache["softmax"].shape)
        # print("dLdZ",dLdZ.shape)
        # print("dZdW2",dZdW2.shape)
        # print("dLdH",dLdH.shape)
        # print("dLdK",dLdK.shape)

    def _l2(self):
        return 0.5 * self.lamda * np.sum([np.sum(np.power(w,2)) for w in self.w]) 
    
    def sgd(self):
        for i in range(len(self.w)):
            self.w[i]-= self.learning_rate * self.cache[f"dW{i+1}"]
            self.b[i]-= self.learning_rate * self.cache[f"db{i+1}"]

        
      

    def momentum(self,beta = 0.9):
        if not hasattr(self,"v_w"):
            self.v_w = [np.zeros_like(w) for w in self.w]
            self.v_b = [np.zeros_like(b) for b in self.b]
            
        for i in range(len(self.w)):
            self.v_w[i] = beta*self.v_w[i] + (1-beta)*self.cache[f"dW{i+1}"]
            self.v_b[i] = beta*self.v_b[i] + (1-beta)*self.cache[f"db{i+1}"]

            self.w[i] -= self.learning_rate * self.v_w[i]
            self.b[i] -= self.learning_rate * self.v_b[i]
        

    def adam(self,beta_1=0.9,beta_2=0.999, epsilon=1e-8):
        if not hasattr(self, "m_w"):
            self.m_w = [np.zeros_like(w) for w in self.w]
            self.v_w = [np.zeros_like(w) for w in self.w]
            self.m_b = [np.zeros_like(b) for b in self.b]
            self.v_b = [np.zeros_like(b) for b in self.b]
            self.t = 0
        self.t += 1

        for i in range(len(self.w)):
            self.m_w[i] = beta_1 * self.m_w[i] + (1 - beta_1) * self.cache[f"dW{i+1}"]
            self.v_w[i] = beta_2 * self.v_w[i] + (1 - beta_2) * (self.cache[f"dW{i+1}"]**2)
            m_hat = self.m_w[i] / (1 - beta_1**self.t)
            v_hat = self.v_w[i] / (1 - beta_2**self.t)
            self.w[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            self.m_b[i] = beta_1 * self.m_b[i] + (1 - beta_1) * self.cache[f"db{i+1}"]
            self.v_b[i] = beta_2 * self.v_b[i] + (1 - beta_2) * (self.cache[f"db{i+1}"]**2)
            mb_hat = self.m_b[i] / (1 - beta_1**self.t)
            vb_hat = self.v_b[i] / (1 - beta_2**self.t)
            self.b[i] -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + epsilon)

    def _predict(self, X):
        A = X
        for i,(w,b) in enumerate(zip(self.w,self.b)):
            A = self.activation_functions[i](A @ w.T + b)
        return A

    @classmethod
    def gradient_check(cls,X,y,params = {},epsilon=1e-7):
        model = cls(**copy.deepcopy(params))
        model.fit(X,y)
        
        Y = np.zeros((model.n,len(model.n_class)))
        for i, val in enumerate(y):
            Y[i, val] = 1
        model._forward(X)
        model.back_propagation(X, Y)    
        max_diff = 0;
        for l in range(len(model.w)):
            grad_analytic = np.copy(model.cache[f"dW{l+1}"])
            grad_num = np.zeros_like(grad_analytic)
            for i in range(grad_analytic.shape[0]):
                for j in range(grad_analytic.shape[1]):
                    model.w[l][i,j] += epsilon
                    y_plus = model._predict(X)
                    L1 = model._calculate_loss(Y,y_plus)
    
                    model.w[l][i,j] -= 2*epsilon
                    y_minus = model._predict(X)
                    L2 = model._calculate_loss(Y,y_minus)
                    model.w[l][i,j] += epsilon
                    grad_num[i,j] = (L1 - L2)/(2*epsilon)
                
            diff = np.linalg.norm(grad_analytic - grad_num) / (np.linalg.norm(grad_analytic) + np.linalg.norm(grad_num))
            print("Gradient check difference:", diff)
            max_diff = max(max_diff,diff)
        return max_diff
    
    def predict(self, X):
        result = self._predict(X)
        return np.argmax(result,axis = 1)
        
    def plot_loss(self):
        plt.figure(figsize=(8,6))
        plt.title("Training Loss")
        plt.plot(range(len(self.loss)),self.loss)
        plt.ylabel(f"Log Loss ({self.optimizer.capitalize()} optimizer)")
        plt.yticks(np.linspace(min(self.loss), max(self.loss), 10))
        plt.xlabel("Epocs")
        plt.grid()
        plt.show()