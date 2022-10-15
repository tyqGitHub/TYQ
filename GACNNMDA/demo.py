import numpy as np
import random
from datadeal import *
from GAT import *
from utils import *
from sklearn.metrics import roc_curve,auc
import pandas as pd
from CNN import *


A=np.loadtxt("./data/MDAD/drug_microbe_matrix.txt")  #邻接矩阵
A1=np.loadtxt("./data/MDAD/drug_dmicrobe_matrix.txt") #用drug-microbe-disease填补了的关联矩阵
known=np.loadtxt("./data/MDAD/known.txt") #已知关联索引
unknown=np.loadtxt("./data/MDAD/unknown.txt")
known1=np.loadtxt("./data/MDAD/known1.txt")
unknown1=np.loadtxt("./data/MDAD/unknown1.txt")
Smf=np.loadtxt("./data/MDAD/microbe_function_sim.txt")
Src=np.loadtxt("./data/MDAD/drug_structure_sim.txt")
dd=pd.read_excel("./data/MDAD/drug_drug_interactions.xlsx")
dd=dd.values
mm=pd.read_excel("./data/MDAD/microbe_microbe_interactions.xlsx")
mm=mm.values
def kflod_5(known,unknown,A):
    scores = []
    tlabels = []
    k = []
    unk = []
    b=np.zeros((1,2))
    lk = len(known)  # 已知关联数
    luk = len(unknown)  # 未知关联数
    for j in range(lk):
        k.append(j)
    for j in range(luk):
        unk.append(j)
    random.shuffle(k)  # 打乱顺序
    random.shuffle(unk)
    for cv in range(1, 6):
        interaction = np.array(list(A))
        if cv < 5:
            B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]  # 1/5的1的索引
            B2 = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]  # 1/5的0的索引
            for i in range(lk // 5):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        else:
            B1 = known[k[(cv - 1) * (lk // 5):lk], :]
            B2 = unknown[unk[(cv - 1) * (luk // 5):luk], :]
            for i in range(lk - (lk // 5) * 4):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        B=np.vstack((B1,B2))
        b=np.vstack((b,B))
        Srg=GIP_Calculate1(interaction)
        Smg=GIP_Calculate(interaction)
        Srh=HIP_Calculate(interaction)
        Smh=HIP_Calculate(interaction.T)
        Sr=(Srg+Srh)/2
        Sm=(Smg+Smh)/2
        for i in range(len(dd)):
            Sr[int(dd[i][0])-1][int(dd[i][2])-1]=1
        for i in range(len(mm)):
            Sm[int(mm[i][0])-1][int(mm[i][2])-1]=1
        N1 = np.hstack((Sr, interaction))
        N2 = np.hstack((interaction.T, Sm))
        Net = np.vstack((N1, N2))  # 异构网络1
        Srr1 = RWR(Sr)
        Smm1 = RWR(Sm)
        Fr1 = np.hstack((Src, A))
        Fr1 = np.hstack((Fr1, Srr1))
        Fr1 = np.hstack((Fr1, A))
        Fm1 = np.hstack((A.T, Smf))
        Fm1 = np.hstack((Fm1, A.T))
        Fm1 = np.hstack((Fm1, Smm1))
        F = np.vstack((Fr1, Fm1))
        train33(Net,F)
        Feature = np.loadtxt("./embedding.txt")
        #构造特征矩阵

        Fr = Feature[0:1373, :]
        Fr = np.hstack((Fr, Src))
        Fr = np.hstack((Fr, A))
        Fr = np.hstack((Fr, Srr1))
        Fr = np.hstack((Fr, A))

        Fm = Feature[1373:, :]
        Fm=np.hstack((Fm,A.T))
        Fm=np.hstack((Fm,Smf))
        Fm=np.hstack((Fm,A.T))
        Fm=np.hstack((Fm,Smm1))

        F = Nodepairs(Fr, Fm)
        label = []
        for i in range(np.size(interaction, axis=0)):
            for j in range(np.size(interaction, axis=1)):
                label.append(interaction[i][j])
        label = np.array(label)
        F_test = []  # 测试集数据
        label_test = []  # 测试集标签
        tlabel_test = []  # 真实测试集标签
        index_test = []
        for i in range(len(B1)):
            index = int((B1[i, 0] - 1) * 173 + B1[i, 1] - 1)
            index_test.append(index)
            F_test.append(F[index])
            # label_test.append(label[int(index)])
            tlabel_test.append(A[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1])
        for i in range(len(B2)):
            index = int((B2[i, 0] - 1) * 173 + B2[i, 1] - 1)
            index_test.append(index)
            F_test.append(F[index])
            # label_test.append(label[int(index)])
            tlabel_test.append(A[int(B2[i, 0]) - 1, int(B2[i, 1]) - 1])
        F_test = np.array(F_test)
        tlabel_test = np.array(tlabel_test)
        F_train = []
        label_train = []
        index_train = []
        for i in range(173 * 1373):
            index_train.append(i)
        for i in range(len(index_test)):
            index_train.remove(index_test[i])
        for i in range(len(index_train)):
            F_train.append(F[i])
            label_train.append(label[i])
        score, tlabel = train1(F_train, label_train, F_test, tlabel_test)
        for i in range(len(score)):
            scores.append(score[i])
            tlabels.append(tlabel[i])

        # score = torch.sigmoid(torch.FloatTensor(np.dot(Fr, Fm.T)))
        # score = np.array(score)
        # for i in range(len(B1)):
        #     index1 = int(B1[i, 0] - 1)
        #     index2 = int(B1[i, 1] - 1)
        #     scores.append(score[index1, index2])
        #     tlabels.append(A[index1, index2])
        # for i in range(len(B2)):
        #     index1 = int(B2[i, 0] - 1)
        #     index2 = int(B2[i, 1] - 1)
        #     scores.append(score[index1, index2])
        #     tlabels.append(A[index1, index2])
        print("fold cv--{}".format(cv))
    return scores,tlabels,b
scores1,tlabels1,blabel1=kflod_5(known,unknown,A)
np.savetxt("./data/scores1.txt",scores1)
np.savetxt("./data/tlabels1.txt",tlabels1)
np.savetxt("./data/blabel1.txt",blabel1)

scores2,tlabels2,blabel2=kflod_5(known1,unknown1,A1)
np.savetxt("./data/scores2.txt",scores2)
np.savetxt("./data/tlabels2.txt",tlabels2)
np.savetxt("./data/blabel2.txt",blabel2)


scores1=np.loadtxt("./data/scores1.txt")
scores2=np.loadtxt("./data/scores2.txt")
blabel1=np.loadtxt("./data/blabel1.txt")
blabel2=np.loadtxt("./data/blabel2.txt")
# tlabels1=np.loadtxt("./data/tlabels1.txt")
# tlabels2=np.loadtxt("./data/tlabels2.txt")
s1=np.zeros((1373,173))
s2=np.zeros((1373,173))
tlabels=[]
for i in range(len(A)):
    for j in range(np.size(A,axis=1)):
        tlabels.append(A[i][j])
tmp=0
for i in range(1,len(blabel1)):
    s1[int(blabel1[i][0] - 1)][int(blabel1[i][1] - 1)] = scores1[tmp]
    tmp += 1
tmp=0
for i in range(1,len(blabel2)):
    s2[int(blabel2[i][0] - 1)][int(blabel2[i][1] - 1)] = scores2[tmp]
    tmp += 1
scores=[]
for i in range(len(s1)):
    for j in range(np.size(s1,axis=1)):
        scores.append((s1[i][j]+s2[i][j])/2)
fpr, tpr, threshold = roc_curve(tlabels, scores)
auc_val = auc(fpr, tpr)
print(auc_val)
