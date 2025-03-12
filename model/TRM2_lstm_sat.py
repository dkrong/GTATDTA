import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, global_add_pool, GATv2Conv
from torch_geometric.nn import TransformerConv #
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import math
# GCN-CNN based model 
# D: TransformerConv +TransformerConv
# T : LSTM +sat

class TRM2_lstm_sat(torch.nn.Module): ##### TransformerConv
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2,hidden_dim=128):

        super(TRM2_lstm_sat, self).__init__()##### TransformerConv
        
        

        self.n_output = n_output
        self.conv1 = TransformerConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = TransformerConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.W_Q = nn.Linear(hidden_dim,hidden_dim,bias =False)
        self.W_K = nn.Linear(hidden_dim,hidden_dim,bias =False)
        self.W_V = nn.Linear(hidden_dim,hidden_dim,bias =False)

        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, 64, 1, dropout=0.2, bidirectional=True)
        #self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        #self.fc1_xt = nn.Linear(32*121, output_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)
        #self.dropout = nn.Dropout(0.5)
        self.layernorm = nn.LayerNorm(hidden_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def attention(self,Q,K,V):
        d_k = K.size(-1)
        scores = torch.matmul(Q,K.transpose(1,2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores,dim=-1)
        context = torch.matmul(alpha_n,V)
        output = context.sum(1)
        
        return output,alpha_n

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        
        embedded_xt = self.embedding_xt(target)
        output_xt, _ = self.bilstm(embedded_xt)
        
        Q = self.W_Q(output_xt)     
        K = self.W_K(output_xt)
        V = self.W_V(output_xt)

        #conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        #xt = conv_xt.view(-1, 32 * 121)
       # xt = self.fc1_xt(xt)
        attn_output,alpha_n = self.attention(Q, K, V)
        out_ln = self.layernorm(attn_output)

        xt = self.fc(out_ln)  

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
