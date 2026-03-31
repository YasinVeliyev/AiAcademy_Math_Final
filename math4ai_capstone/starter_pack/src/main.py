import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import argparse
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
test_data = [(X_d_test,y_d_test),(X_l_test,y_l_test),(X_m_test,y_m_test)]
data_name = ["Digits","Linear gaussian","Moons"]


def gradient_sanity_check():
    SoftMaxClassification.gradient_check(X_d_train[:30],y_d_train[:30])
    NeuralNetwork.gradient_check(X_d_train[:30],y_d_train[:30])




def one_failure_case_analysis():
    fig,axs = plt.subplots(1,2,figsize=(18,6))
    ss = SoftMaxClassification()
    Validation.test_model(ss,X_d_train[:30],y_d_train[:30],X_d_val,y_d_val,X_d_test,y_d_test,ax = axs[0])

    nn = NeuralNetwork(optimizer="adam",learning_rate=0.001,size=[64])
    Validation.test_model(nn,X_d_train[:30],y_d_train[:30],X_d_val,y_d_val,X_d_test,y_d_test,ax = axs[1])
    plt.show()
    
    fig, axs = plt.subplots(1, 3,figsize=(21, 6),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    idxs = make_bias(y_d_train,1)
    plot_hist(y_d_train,idxs,axs[:2])
    nn = NeuralNetwork(optimizer="adam",learning_rate=0.3,size=[8])
    Validation.test_model(nn,X_d_train[idxs],y_d_train[idxs],X_d_val,y_d_val,X_d_test,y_d_test,ax=axs[2])
    plt.show()






def compare_at_fixed_pca_dimension(dimensions:list,estimator,X_Train,y_train,X_val,y_val):
    X_mean,X_centered,Vt = pca_dimensions(X_Train)
    X_d_val_centered = X_val - X_mean
    fig, axs = plt.subplots(1, 3,figsize=(18, 6),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    for i,d in enumerate(dimensions):
        X_d_t = X_centered @ Vt[:d].T 
        X_d_v = X_d_val_centered @ Vt[:d].T
        model_pca = estimator()
        model_pca.fit(X_d_t,y_train)
        acc = accuracy(model_pca.predict(X_d_v),y_val)
        text = f"Data set {data_name[0]}\nDimension: {d}\nAccuracy :{acc:.4f}"
        axs[i].text(0.5,0.5,text,color="black",fontsize=14,ha="center",transform=axs[i].transAxes)
        model_pca.plot_loss(axs[i])
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ofca", action="store_true", help="One failure-case analysis.")
    parser.add_argument("--gsc", action="store_true", help="Gradient sanity check")
    parser.add_argument("--cafpd", action="store_true", help="Softmax comparison at fixed PCA dimensions m ∈ {10, 20, 40}")
    parser.add_argument("--psp", action="store_true", help="Plot scree plot")

    args = parser.parse_args()
    if args.ofca:
        one_failure_case_analysis()
    if args.gsc:
        gradient_sanity_check()
    
    if args.cafpd:
        compare_at_fixed_pca_dimension([10,20,40],SoftMaxClassification,X_d_train,y_d_train,X_d_val,y_d_val)
    if args.psp:
        plot_scree(X_d_train)
