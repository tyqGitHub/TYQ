import numpy as np
import random
from datadeal import *
from GAT import *
from utils import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import pandas as pd




A=np.loadtxt("./data/MDAD/drug_microbe_matrix.txt")
Smf=np.loadtxt("./data/MDAD/microbe_function_sim.txt")
Src=np.loadtxt("./data/MDAD/drug_structure_sim.txt")

Fr=np.hstack((Src,A))
Fm=np.hstack((A.T,Smf))
scores=[]
labels=[]

score=torch.sigmoid(torch.FloatTensor(np.dot(Fr, Fm.T)))
score=score.numpy()
for i in range(1373):
    for j in range(173):
        scores.append(score[i][j])
        labels.append(A[i][j])
fpr, tpr, threshold = roc_curve(labels, scores)
auc_val = auc(fpr, tpr)
print(auc_val)