import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
matplotlib.use("TkAgg")

from validation import Validation
from nn import NeuralNetwork
from softmax_classfication import SoftMaxClassification
from utils import *
from pca import pca_dimensions,plot_scree

data_digits = np.load(r"math4ai_capstone/starter_pack/data/digits_data.npz")
digits_split_indices = np.load(r"math4ai_capstone/starter_pack/data/digits_split_indices.npz")
data_linear = np.load(r"math4ai_capstone/starter_pack/data/linear_gaussian.npz")
data_moons = np.load(r"math4ai_capstone/starter_pack/data/moons.npz")


X_d_train = data_digits["X"][digits_split_indices["train_idx"]]
y_d_train = data_digits["y"][digits_split_indices["train_idx"]]
X_d_test = data_digits["X"][digits_split_indices["test_idx"]]
y_d_test = data_digits["y"][digits_split_indices["test_idx"]]
X_d_val = data_digits["X"][digits_split_indices["val_idx"]]
y_d_val = data_digits["y"][digits_split_indices["val_idx"]]


X_l_train=data_linear['X_train']
y_l_train= data_linear['y_train']
X_l_val=data_linear['X_val']
y_l_val= data_linear['y_val']
X_l_test=data_linear['X_test'] 
y_l_test=data_linear['y_test']


X_m_train=data_moons['X_train']
y_m_train= data_moons['y_train']
X_m_val=data_moons['X_val']
y_m_val= data_moons['y_val']
X_m_test=data_moons['X_test'] 
y_m_test=data_moons['y_test']




data = [(X_d_train,y_d_train,X_d_val,y_d_val),(X_l_train,y_l_train,X_l_val,y_l_val),(X_m_train,y_m_train,X_m_val,y_m_val)]
data_name = ["Digits","Linear gaussian","Moons"]

for i,d in enumerate(data):
    print(f"Data set {data_name[i]}")        
    validation_s = Validation(SoftMaxClassification,*d)
    acc,model_s = validation_s.fit()
    validation_s.report()



for d in data[1:]:
    fig, axs = plt.subplots(1, 2,figsize=(10,4),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    for i,(lr,e) in enumerate(itertools.product([0.05],[200])):
        ss = SoftMaxClassification(learning_rate=lr,max_iter=e)
        ss.fit(*d[0:2])
        ss.plot_loss(ax=axs[0])
        ss.plot_decison_boundary(*d[0:2],ax=axs[1])



for i,d in enumerate(data):
    print(f"Data set {data_name[i]}")
    validation_nn = Validation(NeuralNetwork,X_d_train,y_d_train,X_d_val,y_d_val,params={"size":[32],"optimizer":"momentum","epochs":200,"learning_rate":0.05})
    acc_m,model_nn = validation_nn.fit()
    validation_nn.report()




for j,d in enumerate(data):
    print(f"Data set {data_name[j]}")
    fig, axs = plt.subplots(2, 3,figsize=(18, 12),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    for i,(o,s,lr) in enumerate(itertools.product(["sgd","adam","momentum"],[32],[0.001,0.05])):
        nn = NeuralNetwork(size=[s],optimizer=o,learning_rate=lr)
        nn.fit(*d[:2])
        ax = axs[i//3,i%3]
        nn.plot_loss(ax)



for j,d in enumerate(data):
    print(f"Data set {data_name[j]}")
    fig, axs = plt.subplots(2, 3,figsize=(18, 12),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    for i,(o,s,lr) in enumerate(itertools.product(["sgd","adam","momentum"],[32],[0.001,0.05])):
        nn = NeuralNetwork(size=[s],optimizer=o,learning_rate=lr)
        nn.fit(*d[:2])
        ax = axs[i//3,i%3]
        nn.plot_decision_boundary(*d[:2],ax)




idxs = make_bias(y_d_train,1)
plot_hist(y_d_train,idxs)


nn = NeuralNetwork(optimizer="adam",learning_rate=0.2,size=[8])
test_model(nn,X_d_train[idxs],y_d_train[idxs],X_d_test,y_d_test,epochs=200)


idxs = make_bias(y_l_train,1)
plot_hist(y_l_train,idxs)

ss = SoftMaxClassification(learning_rate=0.05,batch_size=32)
test_model(ss,X_l_train[idxs],y_l_train[idxs],X_l_val,y_l_val)


nn = NeuralNetwork(optimizer="adam",learning_rate=0.001,size=[64])
test_model(nn,X_d_train[:30],y_d_train[:30],X_d_val,y_d_val,epochs=200)



def compare_at_fixed_pca_dimension(dimensions:list,estimator,X_Train,y_train,X_val,y_val):
    X_mean,X_centered,Vt = pca_dimensions(X_Train)
    X_d_val_centered = X_val - X_mean
    for d in dimensions:
        print(f"PCA dimension {d}")
        X_d_t = X_centered @ Vt[:d].T 
        X_d_v = X_d_val_centered @ Vt[:d].T
        validation_pca_s = Validation(estimator,X_d_t,y_train,X_d_v,y_val)
        acc,model_pca = validation_pca_s.fit()
        model_pca.plot_loss()
        validation_pca_s.report()

plot_scree(X_d_train)
compare_at_fixed_pca_dimension([10,20,40],SoftMaxClassification,X_d_train,y_d_train,X_d_val,y_d_val)