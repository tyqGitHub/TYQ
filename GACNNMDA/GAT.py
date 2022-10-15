import torch
from torch import nn
from layers import GraphConvolution,GraphConvSparse,GraphAttention
import torch.nn.functional as F
import numpy as np
from datadeal import *
import torch.optim as optim
import time




def Graph_update(G,adj):
    adj=adj.detach().numpy()
    adj=adj[0:1373,1373:]
    Sr=adj[0:1373,0:1373]
    Sm=adj[1373:,1373:]
    fr=G[0:1373,:]
    fm=G[1373:,:]
    fr=fr.detach().numpy()
    fm = fm.detach().numpy()
    Sgfr=GIP_Calculate1(fr)
    Sgfm=GIP_Calculate(fm.T)
    Sr1=Sgfr
    Sm1=Sgfm
    for i in range(len(Sr)):
        for j in range(np.size(Sr,axis=1)):
            if Sr[i][j]==1:
                Sr1[i][j]=1
    for i in range(len(Sm)):
        for j in range(np.size(Sm, axis=1)):
            if Sm[i][j] == 1:
                Sm1[i][j] = 1
    N1 = np.hstack((Sr1, adj))
    N2 = np.hstack((adj.T, Sm1))
    Net = np.vstack((N1, N2))
    Net=torch.FloatTensor(Net)
    return Net
class GAT(nn.Module):
    def __init__(self,nfeat,ndim,nclass,dropout,alpha):
        super(GAT, self).__init__()
        self.gal1=GraphAttention(nfeat,ndim, dropout,alpha)
        self.gal2=GraphAttention(ndim,nclass,dropout,alpha)
    def forward(self,x,adj):
        Z=self.gal1(x,adj)
        #nadj=Graph_update(Z,adj)
        Z=self.gal2(Z,adj)
        a=Z.detach().numpy()
        np.savetxt("./embedding.txt",a)
        ZZ=torch.sigmoid(torch.matmul(Z,Z.T))
        return ZZ

def train33(Net,Feature):
    adj=torch.FloatTensor(Net)
    x=torch.FloatTensor(Feature)
    idx_train = range(1300)
    idx_test = range(1300, 1500)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    model=GAT(x.shape[1],256,128,0.4,0.2)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.01, weight_decay=5e-4)
    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(x, adj)
        loss_train = F.mse_loss(output[idx_train,:], adj[idx_train,:])
        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.5f}'.format(loss_train.item()),
              'time: {:.4f}s'.format(time.time() - t))
        return loss_train

    def test():
        model.eval()
        output = model(x, adj)
        loss_test = F.mse_loss(output[idx_test,:], adj[idx_test,:])
        print("Test set results:",
              "loss= {:.5f}".format(loss_test.item()))

    t_total = time.time()
    for epoch in range(10):
        loss = train(epoch)
        # if loss < 0.1:
        #     break

    print("Optimization Finished!")
    print("Total time elapsed: {:.5f}s".format(time.time() - t_total))

    test()
