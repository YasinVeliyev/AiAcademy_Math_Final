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
    fig,axs = plt.subplots(2,2,figsize=(18,12))
    ss = SoftMaxClassification()
    Validation.test_model(ss,X_d_train[:30],y_d_train[:30],X_d_val,y_d_val,X_d_test,y_d_test,axes = axs[0])

    nn = NeuralNetwork(optimizer="adam",learning_rate=0.001,size=[64])
    Validation.test_model(nn,X_d_train[:30],y_d_train[:30],X_d_val,y_d_val,X_d_test,y_d_test,axes = axs[1])
    plt.show()
    
    fig, axs = plt.subplots(2, 2,figsize=(18, 12),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    idxs = make_bias(y_d_train,1)
    plot_hist(y_d_train,idxs,axs[0])
    nn = NeuralNetwork(optimizer="adam",learning_rate=0.3,size=[8])
    Validation.test_model(nn,X_d_train[idxs],y_d_train[idxs],X_d_val,y_d_val,X_d_test,y_d_test,axes=axs[1])
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
    parser.add_argument("--cpad", action="store_true", help="Checkpoint policy across datasets")
    parser.add_argument("--ra", action="store_true", help="Required ablations")
    parser.add_argument("--pca", action="store_true", help="Track A: PCA/SVD and input geometry")
    parser.add_argument("--rss", action="store_true", help="Repeated-seed statistics")
    parser.add_argument("--isc", action="store_true", help="Implementation sanity checks")
    parser.add_argument("--rce", action="store_true", help="Required core experiments")
   

    args = parser.parse_args()
    if args.ra:
        one_failure_case_analysis()
        fig,axs =  plt.subplots(1,3,figsize=(18,6),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
        
        # Optimizer study on digits.
        for i,o in enumerate(["adam","momentum","sgd"]):
            text = f"Data set {data_name[0]};Optimizer: {o.capitalize()}"
            model_nn = NeuralNetwork(optimizer=o)
            model_nn.fit(X_d_train,y_d_train)
            axs[i].text(0.5,0.5,text,color="black",fontsize=14,ha="center",transform=axs[i].transAxes,bbox=dict(facecolor='white', alpha=0.7))
            axs[i].text(0.5,0.5,text,color="black",fontsize=14,ha="center",transform=axs[i].transAxes,bbox=dict(facecolor='white', alpha=0.7))
            model_nn.plot_loss(axs[i])
        plt.show()
        
        #Capacity ablation on moons. Use hidden widths {2, 8, 32}. Interpret what changes in the learned decision boundary.
        fig, axs = plt.subplots(3, 3,figsize=(21, 18),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
        for i,(o,s) in enumerate(itertools.product(["sgd","adam","momentum"],[2,8,32])):
            nn = NeuralNetwork(size=[s],optimizer=o)
            nn.fit(X_m_train,y_m_train)
            ax = axs[i//3,i%3]
            nn.plot_decision_boundary(X_m_train,y_m_train,ax)
        plt.show()
    
    if args.isc:
        gradient_sanity_check()

        # successful overfitting of a very small subset of training examples,
        # evidence that the loss decreases on a tiny subset after a few updates,
        fig,axs = plt.subplots(2,2,figsize=(18,12))
        ss = SoftMaxClassification()
        Validation.test_model(ss,X_d_train[:30],y_d_train[:30],X_d_val,y_d_val,X_d_test,y_d_test,epochs=200,axes = axs[0])
        nn = NeuralNetwork(optimizer="adam",learning_rate=0.001,size=[64])
        Validation.test_model(nn,X_d_train[:30],y_d_train[:30],X_d_val,y_d_val,X_d_test,y_d_test,epochs=200,axes = axs[1])
        plt.show()
        
        #confirmation that predicted class probabilities sum to one,
        s = X_d_test.shape[0]
        ss = SoftMaxClassification(learning_rate=0.05,batch_size=32)
        ss.fit(X_d_train,y_d_train)
        print(ss._predict(X_d_test[np.random.randint(0,s)]).sum(axis=1),nn._predict(X_d_test[np.random.randint(0,s)]).sum(axis=1))
    
    if args.pca:
        # one scree plot
        plot_scree(X_d_train)
        #  one small softmax comparison at fixed PCA dimensions m ∈ {10, 20, 40},
        compare_at_fixed_pca_dimension([10,20,40],SoftMaxClassification,X_d_train,y_d_train,X_d_val,y_d_val)

        # one 2D PCA visualization of the digits data,
        X_mean,X_centered,Vt = pca_dimensions(X_d_train)
        X_2d = X_centered @ Vt[:2].T
        plt.scatter(X_2d[:,0], X_2d[:,1], c=y_d_train, cmap='tab10')
        plt.title("2D PCA visualization of the digits data")
        plt.show()

    if args.rss or args.cpad:
        fig,axs =  plt.subplots(1,2,figsize=(18,6))
        print(f"--- Statistics for 5 Seeds ---")
        text = f"Data set {data_name[0]}\nOptimizer: SGD"
        validation_s = Validation(SoftMaxClassification,*data[0],*test_data[0])
        acc,model_s = validation_s.fit()
        validation_s.report(axs[0])
        model_s.plot_loss(axs[1])
        axs[0].text(0.5,0.5,text,color="black",fontsize=14,ha="center",transform=axs[0].transAxes,bbox=dict(facecolor='white', alpha=0.7))
        axs[1].text(0.5,0.5,text,color="black",fontsize=14,ha="center",transform=axs[1].transAxes,bbox=dict(facecolor='white', alpha=0.7))
        plt.show()
    if args.rce:
        # Train and compare both models on the linear Gaussian task. Include decision-boundary plots.
        # Train and compare both models on the moons task. Include decision-boundary plots.
        for i,d in enumerate(data[1:],1):
            text = f"Data set:{data_name[i]}"
            fig, axs = plt.subplots(2, 2,figsize=(16,10),gridspec_kw={"hspace": 0.4, "wspace": 0.3})
            for ax in axs.flat:
                ax.text(0.5,0.5,text,color="black",fontsize=14,ha="center",transform=ax.transAxes,bbox=dict(facecolor='white', alpha=0.7))
                
            ss = SoftMaxClassification()
            ss.fit(*d[0:2])
            axs[0,0].text(0.5,0.4,f"Accuracy :{accuracy(ss.predict(test_data[i][0]),test_data[i][1]):.4f}",color="black",fontsize=14,ha="center",transform=axs[0,0].transAxes,bbox=dict(facecolor='white', alpha=0.7))
            ss.plot_loss(ax=axs[0,0])
            ss.plot_decision_boundary(*d[0:2],ax=axs[0,1])
            
            nn = NeuralNetwork()
            nn.fit(*d[0:2])
            axs[1,0].text(0.5,0.4,f"Accuracy :{accuracy(nn.predict(test_data[i][0]),test_data[i][1]):.4f}",color="black",fontsize=14,ha="center",transform=axs[1,0].transAxes,bbox=dict(facecolor='white', alpha=0.7))
            nn.plot_loss(ax=axs[1,0])
            nn.plot_decision_boundary(*d[0:2],ax=axs[1,1])
            plt.show()

        # Train and compare both models on the fixed digits benchmark using the same preprocessingand split.
        text = f"Data set:{data_name[0]}"
        fig, axs = plt.subplots(1, 2,figsize=(16,6),gridspec_kw={"hspace": 0.4, "wspace": 0.3})

        ss = SoftMaxClassification()
        ss.fit(X_d_train,y_d_train)
        axs[0].text(0.5,0.5,f"Data set:{data_name[0]}\nAccuracy {accuracy(ss.predict(X_d_test),y_d_test):.4f}",color="black",fontsize=14,ha="center",transform=axs[0].transAxes,bbox=dict(facecolor='white', alpha=0.7))
        ss.plot_loss(ax=axs[0])


        nn = NeuralNetwork()
        nn.fit(X_d_train,y_d_train)
        axs[1].text(0.5,0.5,f"Data set:{data_name[0]}\nAccuracy {accuracy(nn.predict(X_d_test),y_d_test):.4f}",color="black",fontsize=14,ha="center",transform=axs[1].transAxes,bbox=dict(facecolor='white', alpha=0.7))
        nn.plot_loss(ax=axs[1])
        plt.show()