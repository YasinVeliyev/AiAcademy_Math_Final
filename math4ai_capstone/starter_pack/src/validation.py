import numpy as np
import copy
from scipy import stats
import matplotlib.pyplot as plt

from utils import accuracy






class Validation():
    def __init__(self,estimator,X,y,X_val,y_val,params = {},cv=5):
        self.estimator=estimator
        self.cv = cv
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.results = []
        self.models = []
        self.params = params
        
    def fit(self):
        self.results = []
        self.models = []
        for i in range(self.cv):
            np.random.seed(i)
            model = self.estimator(**copy.deepcopy(self.params));
            model.fit(self.X,self.y)
            y_pred = model.predict(self.X_val)
            self.results.append(accuracy(y_pred,self.y_val))
            self.models.append(model)
        return max(self.results),self.models[np.argmax(self.results)]

    def report(self):
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
        print(f"--- Statistics for {self.cv} Seeds ---")
        print(f"Accuracy mean:{a_m:.5f}")
        print(f"Loss mean for {self.cv} models:{l_m:.5f}")
        print(f"95 % Confidence Interval for Accuracy mean:({low_a:.5f},{high_a:.5f})")
        print(f"95 % Confidence Interval for Loss mean:({low_l:.5f},{high_l:.5f})")
        self._precision(self.X_val,self.y_val)

    
    def _precision(self,X,y):
        y_pred = self.models[np.argmax(self.results)].predict(X)
        result = y_pred==y
        labels = np.unique(y)
        precisions = []
        acc = accuracy(y_pred,y)
        for l in labels:
            idx = np.where(y_pred==l)
            precision = result[idx].sum()/(y_pred==l).sum()
            precisions.append(precision)
            
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, precisions, color="darkorange", edgecolor="k")
        
        for bar, prec in zip(bars, precisions):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{prec:.2f}",
                    ha="center", fontsize=9)
        
        ax.set_title("Per-Class Precision")
        ax.set_xlabel("Class")
        ax.set_ylabel("Precision")
        ax.set_xticks(labels)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=acc, color="red", linestyle="--",
                   label=f"Overall Accuracy: {acc:.4f}")
        ax.legend()
        ax.grid(axis="y")
        plt.tight_layout()
        plt.show()
    
        