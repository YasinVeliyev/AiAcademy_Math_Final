import numpy as np
import copy
import matplotlib.pyplot as plt


from utils import *




class NeuralNetwork():
    def __init__(self,size=[32],lamda = 1e-4,batch_size=64,learning_rate = None, optimizer="sgd",epochs = 200,activation_functions = (np.tanh,softmax),init_weights=True):
        self.size = size
        self.batch_size = batch_size
        if not learning_rate:
            self.learning_rate = 0.001 if optimizer=="adam" else 0.05
        else:
            self.learning_rate = learning_rate
        self.cache = {}
        self.optimizer=optimizer
        self.lamda = lamda
        self.epochs = epochs
        self.loss=[]
        self.activation_functions = activation_functions
        self.init_weights = init_weights
    
    def fit(self,X_train,y_train):
        self.n,self.m = X_train.shape
        self.labels = np.unique(y_train)
        
        if self.init_weights:
            self._initialize_weights(X_train,y_train) 
        Y = one_hot_encoder(y_train,self.labels)
        
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
        
        size = [self.m] + self.size + [len(self.labels)]
        
        self.W = [np.random.randn(size[i+1],size[i])*np.sqrt(1/size[i]) for i in range(len(size)-1)]
        self.b = [np.zeros((1,size[i+1])) for i in range(len(size)-1)]
    
    def _forward(self,X_batch):
        A = X_batch
        for i,(w,b) in enumerate(zip(self.W,self.b)):
            activation = self.activation_functions[i]
            A = activation(A @ w.T + b)
            self.cache[f"A{i}"] = A

    def _calculate_loss(self,Y,y_predict):
        return -np.mean(np.sum(Y*np.log(y_predict + 1e-9),axis=1)) + self._l2()
    
    def _optimize(self):
        if self.optimizer=="sgd":
            self.sgd()
        elif self.optimizer=="adam":
            self.adam()
        else:
            self.momentum()
        
    def back_propagation(self,X,Y):
        dLdZ= (self.cache["A1"] - Y)/Y.shape[0]
        dZdW2 = self.cache["A0"]
        self.cache["dW2"] = dLdZ.T @ dZdW2 + self.lamda*self.W[1]
        self.cache["db2"] = np.sum(dLdZ, axis=0, keepdims=True)
        
        dLdH =  dLdZ @ self.W[1]
        dLdK = dLdH * (1 - self.cache["A0"]**2)
        self.cache["dW1"] = dLdK.T @ X + self.lamda * self.W[0]
        self.cache["db1"] = np.sum(dLdK, axis=0, keepdims=True)
        
        # print("softmax",self.cache["softmax"].shape)
        # print("dLdZ",dLdZ.shape)
        # print("dZdW2",dZdW2.shape)
        # print("dLdH",dLdH.shape)
        # print("dLdK",dLdK.shape)

    def _l2(self):
        return 0.5 * self.lamda * np.sum([np.sum(np.power(w,2)) for w in self.W]) 
    
    def sgd(self):
        for i in range(len(self.W)):
            self.W[i]-= self.learning_rate * self.cache[f"dW{i+1}"]
            self.b[i]-= self.learning_rate * self.cache[f"db{i+1}"]

        
      

    def momentum(self,beta = 0.9):
        if not hasattr(self,"v_w"):
            self.v_w = [np.zeros_like(w) for w in self.W]
            self.v_b = [np.zeros_like(b) for b in self.b]
            
        for i in range(len(self.W)):
            self.v_w[i] = beta*self.v_w[i] + (1-beta)*self.cache[f"dW{i+1}"]
            self.v_b[i] = beta*self.v_b[i] + (1-beta)*self.cache[f"db{i+1}"]

            self.W[i] -= self.learning_rate * self.v_w[i]
            self.b[i] -= self.learning_rate * self.v_b[i]
        

    def adam(self,beta_1=0.9,beta_2=0.999, epsilon=1e-8):
        if not hasattr(self, "m_w"):
            self.m_w = [np.zeros_like(w) for w in self.W]
            self.v_w = [np.zeros_like(w) for w in self.W]
            self.m_b = [np.zeros_like(b) for b in self.b]
            self.v_b = [np.zeros_like(b) for b in self.b]
            self.t = 0
        self.t += 1

        for i in range(len(self.W)):
            self.m_w[i] = beta_1 * self.m_w[i] + (1 - beta_1) * self.cache[f"dW{i+1}"]
            self.v_w[i] = beta_2 * self.v_w[i] + (1 - beta_2) * (self.cache[f"dW{i+1}"]**2)
            m_hat = self.m_w[i] / (1 - beta_1**self.t)
            v_hat = self.v_w[i] / (1 - beta_2**self.t)
            self.W[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            self.m_b[i] = beta_1 * self.m_b[i] + (1 - beta_1) * self.cache[f"db{i+1}"]
            self.v_b[i] = beta_2 * self.v_b[i] + (1 - beta_2) * (self.cache[f"db{i+1}"]**2)
            mb_hat = self.m_b[i] / (1 - beta_1**self.t)
            vb_hat = self.v_b[i] / (1 - beta_2**self.t)
            self.b[i] -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + epsilon)

    def _predict(self, X):
        A = X
        for i,(w,b) in enumerate(zip(self.W,self.b)):
            A = self.activation_functions[i](A @ w.T + b)
        return A

    @classmethod
    def gradient_check(cls,X,y,params = {},epsilon=1e-7):
        model = cls(**copy.deepcopy(params))
        model.fit(X,y)
        
        Y = one_hot_encoder(y,model.labels)
        model._forward(X)
        model.back_propagation(X, Y)    
        max_diff = 0
        for l in range(len(model.W)):
            grad_analytic = np.copy(model.cache[f"dW{l+1}"])
            grad_num = np.zeros_like(grad_analytic)
            for i in range(grad_analytic.shape[0]):
                for j in range(grad_analytic.shape[1]):
                    model.W[l][i,j] += epsilon
                    y_plus = model._predict(X)
                    L1 = model._calculate_loss(Y,y_plus)
    
                    model.W[l][i,j] -= 2*epsilon
                    y_minus = model._predict(X)
                    L2 = model._calculate_loss(Y,y_minus)
                    model.W[l][i,j] += epsilon
                    grad_num[i,j] = (L1 - L2)/(2*epsilon)
                
            diff = np.linalg.norm(grad_analytic - grad_num) / (np.linalg.norm(grad_analytic) + np.linalg.norm(grad_num))
            max_diff = max(max_diff,diff)
        print(f"Gradient check difference for {cls.__name__}:", max_diff)
        return max_diff
    
    def predict(self, X):
        result = self._predict(X)
        return np.argmax(result,axis = 1)
        
    def plot_decision_boundary(self, X, y, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X
    
        x0_min, x0_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        x1_min, x1_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    
        xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 200),
                             np.linspace(x1_min, x1_max, 200))
        if X.shape[1] > 2:
            grid_original = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
            Z = self.predict(grid_original)
        else:
            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
    
        Z = Z.reshape(xx.shape)
    
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.contour(xx, yy, Z, colors="red", linewidths=1)
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors="k", linewidths=0.5)
        ax.set_title(f"{self.__class__.__name__}\nDecision Boundary {'(PCA 2D)' if X.shape[1] > 2 else ''} \nOptimizer:{self.optimizer.capitalize()}; Hidden Layer Size:{self.size};\nLearning rate:{self.learning_rate};Epochs:{self.epochs};Loss: {self.loss[-1]:.4f}")
        ax.set_xlabel("PC1" if X.shape[1] > 2 else "X0")
        ax.set_ylabel("PC2" if X.shape[1] > 2 else "X1")
                
    def plot_loss(self,ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.set_title(f"{self.__class__.__name__}\nOptimizer:{self.optimizer.capitalize()}; Hidden Layer Size:{self.size};\nLearning rate:{self.learning_rate}")
        ax.plot(range(len(self.loss)),self.loss)
        ax.set_ylabel(f"Training Log Loss")
        ax.set_xlabel("Epochs")
        ax.set_yticks(np.linspace(min(self.loss), max(self.loss), 10))
        ax.grid()
        
        