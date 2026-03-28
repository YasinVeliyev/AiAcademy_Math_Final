import numpy as np
import copy






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
            self.results.append(self.accuracy(y_pred))
            self.models.append(model)
        return max(self.results),self.models[np.argmax(self.results)]

    def report(self):
        loss = [m.loss[-1] for m in self.models]
        l_m = np.mean(loss)
        a_m = np.mean(self.results)
        l_s = np.std(loss,ddof=1)
        a_s = np.std(self.results,ddof=1)
        confidence_interval_for_loss = (l_m - 2.776*l_s/np.sqrt(self.cv),l_m + 2.776*l_s/np.sqrt(self.cv))
        confidence_interval_for_accuracy = (a_m - 2.776*a_s/np.sqrt(self.cv),a_m + 2.776*a_s/np.sqrt(self.cv))
        low_a, high_a = confidence_interval_for_accuracy
        low_l, high_l = confidence_interval_for_loss
        print(f"--- Statistics for {self.cv} Seeds ---")
        print(f"Accuracy mean:{a_m:.5f}")
        print(f"Loss mean for {self.cv} models:{l_m:.5f}")
        print(f"95 % Confidence Interval for Accuracy mean:({low_a:.5f},{high_a:.5f})")
        print(f"95 % Confidence Interval for Loss mean:({low_l:.5f},{high_l:.5f})")

    


        
    def accuracy(self,y_pred):
        return y_pred[y_pred==self.y_val].size/self.y_val.size
            