
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(self,size=[],lamda = 1e-4,batch_size=32,learning_rate = 0.05,optimizer="SGD",epocs = 200):
        self.size = size
        self.batch_size = batch_size
        self.learning_rate=learning_rate
        self.cache = {}
        self.optimizer=optimizer
        self.lamda = lamda
        self.epocs = epocs
        self.loss=[]
        
        
    def fit(self,X_train,y_train):
        self.n,self.m = X_train.shape
        self.n_class = np.unique(y_train)
        
        Y = np.zeros((len(self.n_class), self.n))
        for i, val in enumerate(y_train):
            Y[val, i] = 1
    
        self.size.append(len(self.n_class))
        self.size.insert(0,self.m)
        
        self.w = [np.random.randn(self.size[i+1],self.size[i])*0.01 for i in range(len(self.size)-1)]
        self.b = [np.random.randn(self.size[i+1],1)*0.01 for i in range(len(self.size)-1)]
        idxs = np.arange(self.n)
        for e in range(self.epocs):
            np.random.shuffle(idxs)
            X_shuffled = X_train[idxs]
            Y_shuffled = Y[:,idxs]
            for j in range(0,self.n,self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                Y_batch = Y_shuffled[:,j:j+self.batch_size]
            
                self.cache["tanh"] = np.tanh(self.w[0] @ X_batch.T + self.b[0])
                self.cache["softmax"] = self.softmax(self.w[-1] @ self.cache["tanh"] + self.b[-1])
                self.back_propagation(X_batch,Y_batch)
                if self.optimizer=="SGD":
                    self.sgd()
                elif self.optimizer=="adam":
                    self.adam()
                else:
                    self.momentum()


            y_pred = self._predict(X_train)
            L = - np.mean(np.sum(Y*np.log(y_pred + 1e-9),axis=0))
            self.loss.append(L)
        
    
    
    def back_propagation(self,X,Y):
        dLdZ= (self.cache["softmax"] - Y)/self.batch_size
        dZdW2 = self.cache["tanh"]
        self.cache["dW2"] = dLdZ @ dZdW2.T + self.lamda*self.w[1]
        self.cache["db2"] = np.sum(dLdZ, axis=1, keepdims=True)

        dLdH = self.w[1].T @ dLdZ
        dLdK = dLdH * (1 - self.cache["tanh"]**2)
        self.cache["dW1"] = (dLdK @ X) + (1e-4 * self.w[0])
        self.cache["db1"] = np.sum(dLdK, axis=1, keepdims=True)

        
    
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
    
    def softmax(self,X):
        Z = X -  np.max(X, axis=0, keepdims=True)
        S = np.exp(Z)
        return S/np.sum(S,axis=0,keepdims=True)

    def _predict(self, X):
        tanh = np.tanh(self.w[0] @ X.T + self.b[0])
        softmax = self.softmax(self.w[1] @ tanh + self.b[1])
        return softmax

    def predict(self, X):
        result = self._predict(X)
        return np.argmax(result.T,axis = 1)