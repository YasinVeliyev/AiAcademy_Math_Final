import numpy as np
import copy
from scipy import stats
import matplotlib.pyplot as plt

from utils import *






class Validation():
    def __init__(self,estimator,X_train,y_train,X_val,y_val,X_test,y_test,params = {},epochs = 200,cv=5):
        self.estimator=estimator
        self.cv = cv
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.results = []
        self.models = []
        self.best_epoch= None
        self.params = params
        self.epochs = epochs
        self.W = None
        self.b = None
        
    def fit(self):
        self.select_best_params_and_epochs()
        for i in range(self.cv):
            np.random.seed(i)
            model = self.estimator(**copy.deepcopy(self.params),epochs=self.best_epoch,init_weights=False);
            model.W = copy.deepcopy(self.W)
            model.b = copy.deepcopy(self.b)
            model.fit(self.X_train,self.y_train)
            y_pred = model.predict(self.X_test)
            self.results.append(accuracy(y_pred,self.y_test))
            self.models.append(model)
        return max(self.results),self.models[np.argmax(self.results)]

    def select_best_params_and_epochs(self):
        loss = np.inf
        model = self.estimator(**copy.deepcopy(self.params))
        model.labels = np.unique(self.y_train)
        model.n,model.m = self.X_train.shape
        model._initialize_weights(self.X_train,self.y_train)
        
        Y_train = one_hot_encoder(self.y_train,model.labels)
        Y_val = one_hot_encoder(self.y_val,model.labels)
        
        for j in range(self.epochs):
            A = model._forward(self.X_train)

            y_val_pred = model._predict(self.X_val)
            v_loss = model._calculate_loss(Y_val,y_val_pred)
            
            if v_loss < loss:
                self.W = copy.deepcopy(model.W)
                self.b = copy.deepcopy(model.b)
                self.best_epoch = j
                loss = v_loss
                
            model.back_propagation(self.X_train,Y_train)
            model._optimize()
        
    
    def report(self,ax=None):
        loss = [m.loss[-1] for m in self.models]
        l_m = np.mean(loss)
        a_m = np.mean(self.results)
        l_s = np.std(loss,ddof=1)
        a_s = np.std(self.results,ddof=1)
        
        critic_value = stats.t.ppf(0.975, df=self.cv-1)
        confidence_interval_for_loss = (l_m -critic_value*l_s/np.sqrt(self.cv),l_m +critic_value*l_s/np.sqrt(self.cv))
        confidence_interval_for_accuracy = (a_m -critic_value*a_s/np.sqrt(self.cv),a_m +critic_value*a_s/np.sqrt(self.cv))
        low_a, high_a = confidence_interval_for_accuracy
        low_l, high_l = confidence_interval_for_loss
        
        print(f"Accuracy mean:{a_m:.5f}")
        print(f"Loss mean for {self.cv} models:{l_m:.5f}")
        print(f"95 % Confidence Interval for Accuracy mean:({low_a:.5f},{high_a:.5f})")
        print(f"95 % Confidence Interval for Loss mean:({low_l:.5f},{high_l:.5f})")
        self.accuracy_per_class(self.X_val,self.y_val,ax)


    @staticmethod
    def test_model(model,X_train,y_train,X_val,y_val,X_test,y_test,epochs=200,axes=np.zeros((1,2))):
        if not np.all(axes):
            fig,axes = plt.subplots(1,2,figsize=(8,6))
        
        model.labels = np.unique(y_train)
        model.n,model.m = X_train.shape
        model._initialize_weights(X_train,y_train)
        train_loss=[]
        val_loss = []
        test_loss = []

        train_acc=[]
        val_acc = []
        test_acc = []
        
        labels = np.unique(y_train)
        Y_train = one_hot_encoder(y_train,labels)
        Y_val = one_hot_encoder(y_val,labels)
        Y_test = one_hot_encoder(y_test,labels)
        for j in range(epochs):
            
            A = model._forward(X_train)
            y_train_pred = model._predict(X_train) 
            train_loss.append(model._calculate_loss(Y_train,y_train_pred))
            train_acc.append(accuracy(model.predict(X_train),y_train))
            
            y_val_pred = model._predict(X_val)
            val_loss.append(model._calculate_loss(Y_val,y_val_pred))
            val_acc.append(accuracy(model.predict(X_val),y_val))
            
            y_test_pred = model._predict(X_test)
            test_loss.append(model._calculate_loss(Y_test,y_test_pred))
            test_acc.append(accuracy(model.predict(X_test),y_test))
            
            model.back_propagation(X_train,Y_train)
            model._optimize()
        
        arr = np.hstack([train_loss,val_loss])
    
        # plt.ylim(0,np.quantile(arr,0.95)*1.1)
        # plt.yticks(np.linspace(arr.min(),np.quantile(arr,0.95),10))
        acc_train = accuracy(model.predict(X_train),y_train)
        acc_val = accuracy(model.predict(X_val),y_val)
        acc_test=accuracy(model.predict(X_test),y_test)
        
        text = f"Train Accuracy:{acc_train:.4f}\nValidation Accuracy:{acc_val:.4f}\nTest Accuracy:{acc_test:.4f}\n"
        axes[0].set_title(f"Loss {model.__class__.__name__}")
        axes[0].text(0.5,0.5,text,color="black",fontsize=14,ha="center",transform=axes[0].transAxes,bbox=dict(facecolor='white', alpha=0.7))
        axes[0].plot(range(len(train_loss)),train_loss,label="Train Loss")
        axes[0].plot(range(len(val_loss)),val_loss,label="Validation Loss")
        axes[0].plot(range(len(test_loss)),test_loss,label="Test Loss")
        axes[0].legend()
        axes[0].grid()

       
        axes[1].set_title(f"Accuracy {model.__class__.__name__}")
        axes[1].plot(range(len(train_acc)),train_acc,label="Train Accuracy")
        axes[1].plot(range(len(val_acc)),val_acc,label="Validation Accuracy")
        axes[1].plot(range(len(test_acc)),test_acc,label="Test Accuracy")
        axes[1].set_yticks(np.linspace(0,1,11))
        axes[1].legend()
        axes[1].grid()
        
        
    def accuracy_per_class(self,X,y,ax=None):
        y_pred = self.models[np.argmax(self.results)].predict(X)
        result = y_pred==y
        labels = np.unique(y)
        acc_class = []
        acc = accuracy(y_pred,y)
        for l in labels:
            idx = y==l
            ac = result[idx].sum()/idx.sum()
            acc_class.append(ac)
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,6))
        bars = ax.bar(labels, acc_class, color="darkorange", edgecolor="k")
        
        for bar, prec in zip(bars, acc_class):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{prec:.2f}",
                    ha="center", fontsize=8)
        
        ax.set_title(f"{self.estimator.__name__} Per-Class Accuracy")
        ax.set_xlabel("Class")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(labels)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=acc, color="red", linestyle="--",
                   label=f"Overall Vaildation Accuracy: {acc:.4f}")
        ax.legend()
        
    
        
        
    
        