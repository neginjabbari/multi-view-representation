import scipy
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def purity(y_true, y_pred):
    # contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    pu = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    return pu


def eval(real_labels, labels):
    permutation = []
    n_clusters = len(np.unique(real_labels))
    labels = np.unique(labels, return_inverse=True)[1]
    for i in range(n_clusters):
        idx = labels == i
        if np.sum(idx) != 0:
            new_label = scipy.stats.mode(real_labels[idx])[0]
            permutation.append(new_label)
    new_labels = [permutation[label] for label in labels]
    
    
    nmi = normalized_mutual_info_score(real_labels, labels)
    ari = adjusted_rand_score(real_labels, labels)
    acc = accuracy_score(real_labels, new_labels)
    f1we = f1_score(real_labels, new_labels, average='weighted')
    f1ma = f1_score(real_labels, new_labels, average='macro')
    f1mi = f1_score(real_labels, new_labels, average='micro')
    pur = purity(real_labels, labels)
    return nmi, acc, ari, f1mi, f1ma, f1we, pur
    


def loadDataset(num):
    datasets = ['3Sources.npy',
'BBCSport.npy',
'Caltech101.npy',
'Caltech_2.npy',
'Citeseer.npy',
'Coil100.npy',
'Cora.npy',
'EYaleB10.npy',
'Handwritten.npy',
'MNIST10.npy',
'UCIdigit.npy',
'Umist.npy',
'Yale32.npy',
'Yeast.npy',
'Cora2.npy',
'texas.npy',
'wisconsin.npy',
'washington.npy',
'cornell.npy',
'digit2.npy']
    
    dataset = datasets[num]

    data = torch.load(f'../Datasets/{dataset}')
    V = len(data)-1
    Y = data[-1].flatten()
    c = len(np.unique(Y))
    X0 = data[:-1]

    X = []
    if dataset =='BBCSport.npy' or dataset=='texas.npy' or dataset=='wisconsin.npy' or dataset=='washington.npy' or dataset=='cornell.npy':
        for x in X0:
            X.append(torch.tensor(x.toarray()).type(torch.float32))
    elif dataset =='Caltech_2.npy' or dataset == 'Handwritten.npy' or dataset == 'UCIdigit.npy' or dataset == 'digit2.npy':
        for x in X0:
            X.append(torch.tensor(x.astype(np.float32)).type(torch.float32))
        
    else:
        for x in X0:
            X.append(torch.tensor(x).type(torch.float32))

    return X, Y, V, c
    
    
def normalization(X, norm):
    if norm=='L1':
        Xn = X/torch.sum(X)
    elif norm=='L2col':
        Xn = X/torch.norm(X, dim=0)
    elif norm=='L2row':
        Xn = (X.T/torch.norm(X, dim=1)).T
    elif norm=='L1col':
        Xn = X/torch.norm(X, dim=0, p=1)
    elif norm=='L1row':
        Xn = (X.T/torch.norm(X, dim=1, p=1)).T
    return Xn
    
    
def kmeansInitialization(X, c):
    g = KMeans(n_clusters=c, n_init='auto').fit(X.T).labels_
    H0 = F.one_hot(torch.tensor(g).type(torch.long), c).T + 0.2

    return H0
    
    
def nnmf(X, r, iter):
    eps = torch.tensor(10 ** -10)
    d, n = X.shape

    W = torch.rand(d, r)
    H = torch.rand(r, n)

    for t in range(iter):

        Wn = X @ H.T
        Wd = W @ (H @ H.T)
        W  = W * (Wn/torch.maximum(Wd, eps))

        Hn = W.T @ X
        Hd = (W.T @ W) @ H 
        H  = H * (Hn/torch.maximum(Hd, eps))

    return W, H