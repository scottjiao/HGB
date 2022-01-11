import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb
from torch_geometric.utils import dense_to_sparse, f1_score
from gcn import GCNConv
from torch_scatter import scatter_add
import torch_sparse
import torch_sparse as torch_sparse_old
import time
#import torch_sparse_old
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops, add_self_loops

class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers,dropout):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.dropout=dropout
        layers = []
        """for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=False))"""


        oneDimConvs=[]
        for i in range(num_layers):
            oneDimConvs.append(GTConv(num_edge, num_channels, num_nodes))
        


        self.oneDimConvs = nn.ModuleList(oneDimConvs)
        #self.layers = nn.ModuleList(layers)
        self.loss = nn.CrossEntropyLoss()
        self.gcn = GCNConv(in_channels=self.w_in, out_channels=w_out)
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            edge, value=H[i]
            edge, value = remove_self_loops(edge, value)
            deg_row, deg_col = self.norm(edge.detach(), self.num_nodes, value)
            value = deg_col * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                    dtype=dtype,
                                    device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        row, col = edge_index
        deg = scatter_add(edge_weight.clone(), col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, A, X, target_x, target):
        Hs=[]
        Ws=[]
        for i in range(self.num_layers):
            H = self.oneDimConvs[i](A)
            H = self.normalization(H)
            Hs.append(H)
            Ws.append((F.softmax(self.oneDimConvs[i].weight, dim=1)).detach())



        for i in range(self.num_channels):
            X_temp=X
            for j in range(len(Hs)):
                edge_index, edge_weight = Hs[j][i][0], Hs[j][i][1]
                if j==0:
                    param_=True
                else:
                    param_=False
                X_temp=self.gcn(X_temp,edge_index=edge_index.detach(), edge_weight=edge_weight,param_=param_)

            if i==0:
                X_ = F.relu(X_temp)
                #X_ = X_
            else:
                #edge_index, edge_weight = H[i][0], H[i][1]
                X_ = torch.cat((X_,F.relu(X_temp)), dim=1)
        X_=F.dropout(X_,p=self.dropout,training=self.training)
        X_ = self.linear1(X_)
        X_=F.dropout(X_,p=self.dropout,training=self.training)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws



class history_GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers):
        super(history_GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        layers = []
        """for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=False))"""


        oneDimConvs=[]
        for i in range(num_layers):
            oneDimConvs.append(GTConv(num_edge, num_channels, num_nodes))
        


        self.oneDimConvs = nn.ModuleList(oneDimConvs)
        #self.layers = nn.ModuleList(layers)
        self.loss = nn.CrossEntropyLoss()
        self.gcn = GCNConv(in_channels=self.w_in, out_channels=w_out)
        self.linear1 = nn.Linear(self.w_in, self.w_out)
        self.linear2 = nn.Linear(self.w_out*self.num_channels, self.num_class)
        self.proj=[]
        for i in range(self.num_channels):
            self.proj.append( nn.Linear(self.w_out, 1))
        self.proj=nn.ModuleList(self.proj)


    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            edge, value=H[i]
            edge, value = remove_self_loops(edge, value)
            deg_row, deg_col = self.norm(edge.detach(), self.num_nodes, value)
            value = deg_col * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                    dtype=dtype,
                                    device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        row, col = edge_index
        deg = scatter_add(edge_weight.clone(), col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, A, X, target_x, target):
        Hs=[]
        Ws=[]
        for i in range(self.num_layers):
            H = self.oneDimConvs[i](A)
            H = self.normalization(H)
            Hs.append(H)
            Ws.append((F.softmax(self.oneDimConvs[i].weight, dim=1)).detach())


        X=self.linear1(X)
        for i in range(self.num_channels):
            
            intermed=[]
            X_temp=X
            for j in range(len(Hs)):
                edge_index, edge_weight = Hs[j][i][0], Hs[j][i][1]
                param_=False
                X_temp=self.gcn(X_temp,edge_index=edge_index.detach(), edge_weight=edge_weight,param_=param_)
                intermed.append(X_temp)
            intermed_stacked=torch.stack(intermed, dim=1)
            retain_score = self.proj[i](intermed_stacked)
            retain_score = retain_score.squeeze()
            retain_score = torch.sigmoid(retain_score)
            retain_score = retain_score.unsqueeze(1)
            X_prop = torch.matmul(retain_score, intermed_stacked).squeeze()
            

            if i==0:
                X_ = F.relu(X_prop)
                #X_ = X_
            else:
                #edge_index, edge_weight = H[i][0], H[i][1]
                X_ = torch.cat((X_,F.relu(X_prop)), dim=1)
        #X_ = self.linear2(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws
        

class oldGTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=False))

        GTConv(num_edge, num_channels, num_nodes)


        
        self.layers = nn.ModuleList(layers)
        self.loss = nn.CrossEntropyLoss()
        self.gcn = GCNConv(in_channels=self.w_in, out_channels=w_out)
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            edge, value=H[i]
            edge, value = remove_self_loops(edge, value)
            deg_row, deg_col = self.norm(edge.detach(), self.num_nodes, value)
            value = deg_col * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                    dtype=dtype,
                                    device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        row, col = edge_index
        deg = scatter_add(edge_weight.clone(), col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, A, X, target_x, target):
        Ws = []
        Hs=[]
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:                
                H, W = self.layers[i](A, H)
            H = self.normalization(H)
            Hs.append(H)
            Ws.append(W)
        for i in range(self.num_channels):
            X_temp=X
            for H in Hs:
                edge_index, edge_weight = H[i][0], H[i][1]

                X_temp=F.relu(self.gcn(X,edge_index=edge_index.detach(), edge_weight=edge_weight))

            if i==0:
                X_ = X_temp
                #X_ = X_
            else:
                #edge_index, edge_weight = H[i][0], H[i][1]
                X_ = torch.cat((X_,X_temp), dim=1)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws



class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
            self.conv2 = GTConv(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
    
    def forward(self, A, H_=None):
        t0=time.time()
        if self.first == True:
            result_A = self.conv1(A)
            result_B = self.conv2(A)                
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        t1=time.time()
        print(f"subgraph convolutions use {t1-t0:.4f}s")
        H = []

        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]
            edges, values = torch_sparse_old.spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes, self.num_nodes)
            #edges,values=a_edge, a_value #test the efficiency
            H.append((edges, values))
            t2=time.time()
            print(f"random walk matrix for channel {i+1} mul use {t2-t1:.4f}s")
            t1=t2
        return H, W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        self.bias = None
        self.num_nodes = num_nodes
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        filter = F.softmax(self.weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index,edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value*filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=self.num_nodes, n=self.num_nodes)
            results.append((index, value))
        return results
