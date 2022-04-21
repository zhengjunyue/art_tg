import kaldiio
from kaldiio import ReadHelper
from kaldiio import WriteHelper
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import fftpack

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from pandas import Series,DataFrame
import torch
import torchvision
import sys 
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import itertools
def dim_reduction(feature_mat, labels=None, method='t-sne', target_dim=2,
                  standardise_data=False, return_comp_model=False):
    """ supervised (LDA) [labels required!] and unsupervised (PCA, t-SNE and SVD)
            dimentionality reduction techniques
        
        NOTE1: It is recommended to standardise the data before dim reduction
        NOTE2: If the dim of the data is too high, say above 50 and tsne is 
               intended to be used, it is recommended to do an initial dim 
               redunction to 50 through PCA and then apply t-SNE
    """

    if isinstance(feature_mat, list):
	    feature_mat = np.vstack(feature_mat)
    N, D = feature_mat.shape  # N: number of frames, D: feature vector dimension
        
    if labels is not None:
        if isinstance(labels, list):
            labels = np.hstack(labels)
        label_len = len(labels)
        if N == label_len:
            raise ValueError(f"Number of frames '{N}' and labels '{label_len}' "\
                             "do not match!")

    if target_dim in ['', None]:
        target_dim = D
    elif target_dim <= 0:
        target_dim += D

    if standardise_data:
        feat_mat_mean = np.mean(feature_mat, axis=0)
        feat_mat_std = np.std(feature_mat, axis=0)
        feature_mat = (feature_mat - feat_mat_mean) / feat_mat_std


    method = method.upper()
    
    # 0. DCT
    if method == "DCT":
        comp_feature_mat = dct(feature_mat, target_dim)
    
    # 1. t-SNE (t-distributed Stochastic Neighbor Embedding)
    # NOTE: For t-SNE there is no particular model, each dataset has its own transform
    #       e.g. what is found for train data is not applicable for test data
    if method in ["TSNE","T-SNE"]:
	    comp_model = TSNE(n_components=target_dim, random_state=0)
	    comp_feature_mat = comp_model.fit_transform(feature_mat)
        
    # 2. PCA (Principle Component Analysis)
    elif method == "PCA": 	    
	    comp_model = PCA(n_components=target_dim)
	    comp_feature_mat = comp_model.fit_transform(feature_mat)

    # 3. KPCA (Kernel Principle Component Analysis)
    elif method in ["KPCA", "KERNELPCA", "KERNEL"]:
        kernel = 'rbf' # Default: 'linear', 'poly', 'rbf', 'sigmoid', 'cosine' 	   
        comp_model = KernelPCA(n_components=target_dim, kernel=kernel)
        comp_feature_mat = comp_model.fit_transform(feature_mat)

    # 4. SPCA (Sparse Principle Component Analysis) ...
    elif method in ["SPCA", "SPARSEPCA", "SPARSE"]:
        alpha = 1 # Sparsity control, higher alpha -> higher sparsity    
        comp_model = SparsePCA(n_components=target_dim, alpha=alpha,
                               normalise_components=True)
        comp_feature_mat = comp_model.fit_transform(feature_mat)        

    # 5. TruncatedSVD 
    elif method in ["SVD","TRUNCATEDSVD"]:
	    comp_model = TruncatedSVD(n_components=target_dim, random_state=0)
	    comp_feature_mat = comp_model.fit_transform(feature_mat)
        
	# 6. LDA (Linear Discriminant Analysis)
    elif method == "LDA":
        solver="svd" # "svd", "lsqr", "eigen"
        comp_model = LDA(n_components=target_dim, solver=solver)
        comp_feature_mat = comp_model.fit_transform(feature_mat, labels)

    # 7. Nearest Component Analysis
    elif method == "NCA":
        comp_model = NeighborhoodComponentsAnalysis(n_components=target_dim)
        comp_feature_mat = comp_model.fit_transform(feature_mat, labels)

    else:
        raise ValueError(f"The compression method '{method}' is not supported! "\
                         "ONLY PCA, LDA, TSNE and SVD, NCA")
        
    if return_comp_model:
        return comp_feature_mat, comp_model

    return comp_feature_mat

def dict_generator(generator):
    dict_g = {}
    for key, data in generator:
        dict_g[key] = data
    return dict_g
def ark_file(feat,dnn,ep,fold="source_filter"):
    base_path = "/data/ac1zy/pytorch-kaldi-new/pytorch-kaldi/exp/"+fold+"/"+feat+"/exp_files"
    if fold=="source_filter":        
        file = os.path.join(base_path,"forward_test_ep"+ep+"_ck04_out_"+dnn+".ark")
    else:
        file = os.path.join(base_path,"forward_test_ep"+ep+"_ck4_out_"+dnn+".ark")
    d1=dict_generator(kaldiio.load_ark(file))
    return d1

def concat(arr1,arr2):
    return np.concatenate((arr1, arr2), axis=0)
def k_means_cluster(data):
    pca = PCA(2) 
    #Transform the data
    df = pca.fit_transform(data)
    #Initialize the class object
    kmeans = KMeans(n_clusters= 2)
 
    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
    
    #Getting the Centroids
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)

    #plotting the results:

    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend(loc=[1, 0])
    plt.show()
    
def concat_data(dict1):
    i=0
    label = []
    label_dys = []
    for key in dict1:
        i+=1
        if i==1:
            data = dict1[key]
            label += dict1[key].shape[0] * [key.split("-")[0]]
            if 'C' in key.split("-")[0]:
                label_dys += dict1[key].shape[0] * ['typ']
            else:
                label_dys += dict1[key].shape[0] * ['dys']
        else:
            data = concat(data,dict1[key])
            label += dict1[key].shape[0] * [key.split("-")[0]]
            if 'C' in key.split("-")[0]:
                label_dys += dict1[key].shape[0] * ['typ']
            else:
                label_dys += dict1[key].shape[0] * ['dys']
    label=np.array(label)
    label_dys=np.array(label_dys)
    return data,label,label_dys


def plot_2d(data):
    df1 = dim_reduction(data,method='pca', standardise_data=True, target_dim=50)
    #Transform the data
    #df = pca.fit_transform(data)
    df = dim_reduction(df1,standardise_data=True)
#     u_labels = np.unique(label)
#     for i in u_labels:
#         plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i, alpha=0.05)
#     plt.legend(loc=[1, 0])
#     plt.show()
    return df




def plot_2d_1(data):
    pca = PCA(2) 
    #Transform the data
    #df = pca.fit_transform(data)
    df = dim_reduction(data,method='pca', standardise_data=True, target_dim=2)
#     u_labels = np.unique(label)
#     for i in u_labels:
#         plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i, alpha=0.05)
#     plt.legend(loc=[1, 0])
#     plt.show()
    return df

#dnn4 = ark_file("fd1_mag_sp_nomlp","dnn4","10")
#data,label,label_dys=concat_data(dnn4)
#df4_mag_sp_10 = plot_2d(data)
#with open('pickle/mag_sp_nomlp_dnn4_ep10.pickle', 'wb') as handle:
 #   pickle.dump(df4_mag_sp_10, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#dnn4 = ark_file("fd1_mag_nomlp","dnn4","10")
#data,label,label_dys=concat_data(dnn4)
#df4_mag_10 = plot_2d(data)
#with open('pickle/mag_nomlp_dnn4_ep10.pickle', 'wb') as handle:
 #   pickle.dump(df4_mag_10, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#dnn4 = ark_file("fd1_raw_imag_real_abs_sp_nomlp","dnn4","10")
#data,label,label_dys=concat_data(dnn4)
#df4_real_imag_sp_10 = plot_2d(data)
#with open('pickle/real_imag_abs_sp_nomlp_dnn4_ep10.pickle', 'wb') as handle:
 #   pickle.dump(df4_real_imag_sp_10, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#dnn4 = ark_file("fd1_raw_imag_real_abs_nomlp","dnn4","10")
#data,label,label_dys=concat_data(dnn4)
#df4_real_imag_10 = plot_2d(data)
#with open('pickle/real_imag_abs_nomlp_dnn4_ep10.pickle', 'wb') as handle:
 #   pickle.dump(df4_real_imag_10, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#dnn4 = ark_file("fd1_mag_nomlp","dnn4","39")
#data,label,label_dys=concat_data(dnn4)
#df4_mag_50 = plot_2d(data)
#with open('pickle/mag_nomlp_dnn4_ep40.pickle', 'wb') as handle:
 #   pickle.dump(df4_mag_50, handle, protocol=pickle.HIGHEST_PROTOCOL)

dnn31 = ark_file("fd1_mag_sp_nomlp","dnn31","39")
data,label,label_dys=concat_data(dnn31)
df31_mag_sp = plot_2d(data)
with open('pickle/mag_sp_nomlp_dnn31_ep40.pickle', 'wb') as handle:
    pickle.dump(df31_mag_sp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

dnn32 = ark_file("fd1_mag_sp_nomlp","dnn32","39")
data,label,label_dys=concat_data(dnn32)
df32_mag_sp = plot_2d(data)
with open('pickle/mag_sp_nomlp_dnn32_ep40.pickle', 'wb') as handle:
    pickle.dump(df32_mag_sp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

dnn31 = ark_file("fd1_raw_imag_real_abs_sp_nomlp","dnn31","39")
data,label,label_dys=concat_data(dnn31)
df31_real_imag_sp = plot_2d(data)
with open('pickle/real_imag_abs_sp_nomlp_dnn31_ep40.pickle', 'wb') as handle:
    pickle.dump(df31_mag_sp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

dnn32 = ark_file("fd1_raw_imag_real_abs_sp_nomlp","dnn32","39")
data,label,label_dys=concat_data(dnn32)
df32_real_imag_sp = plot_2d(data)
with open('pickle/real_imag_abs_sp_nomlp_dnn32_ep40.pickle', 'wb') as handle:
    pickle.dump(df32_real_imag_sp, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
dnn31 = ark_file("fd1_mag_nomlp","dnn31","39")
data,label,label_dys=concat_data(dnn31)
df31_mag = plot_2d(data)
with open('pickle/mag_nomlp_dnn31_ep40.pickle', 'wb') as handle:
    pickle.dump(df31_mag, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

dnn32 = ark_file("fd1_mag_nomlp","dnn32","39")
data,label,label_dys=concat_data(dnn32)
df32_mag = plot_2d(data)
with open('pickle/mag_nomlp_dnn32_ep40.pickle', 'wb') as handle:
    pickle.dump(df32_mag, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    

dnn31 = ark_file("fd1_raw_imag_real_abs_nomlp","dnn31","44")
data,label,label_dys=concat_data(dnn31)
df31_real_imag = plot_2d(data)
with open('pickle/real_imag_abs_nomlp_dnn31_ep45.pickle', 'wb') as handle:
    pickle.dump(df31_mag, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

dnn32 = ark_file("fd1_raw_imag_real_abs_nomlp","dnn32","44")
data,label,label_dys=concat_data(dnn32)
df32_real_imag = plot_2d(data)
with open('pickle/real_imag_abs_nomlp_dnn32_ep45.pickle', 'wb') as handle:
    pickle.dump(df32_real_imag, handle, protocol=pickle.HIGHEST_PROTOCOL) 
