import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoModel, BertTokenizer, RobertaTokenizer
from transformers import BertConfig, BertModel


from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, global_add_pool, GATv2Conv
from torch_geometric.nn import TransformerConv #
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import math
from math import sqrt
from scipy import stats
import re
import numpy as np
from okvqa.gumbel_softmax import gumbel_softmax
# GCN-CNN based model 
# D: TransformerConv +TransformerConv
# T : LSTM +sat +
# I : am("affinity matrix")

class GATA(torch.nn.Module): ##### TransformerConv
    def __init__(self, 
                 n_output=2,  # 分类，模型的输出维度是2
                 num_features_xd=78, 
                 num_features_xt=25,
                 n_filters=32, 
                 embed_dim=128, 
                 output_dim=128, 
                 dropout=0.2,
                 hidden_dim=128):

        super(GATA, self).__init__()##### TransformerConv
        
        

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
        
        #self.is_bidirectional = True
        self.bilstm = nn.LSTM(embed_dim,  #input
                              64, # hidden
                                2,  # LSTM层的数量
                                dropout=0.2, 
                                bidirectional=True)  # 双向lstm
        
        #self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        #self.fc1_xt = nn.Linear(32*121, output_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)
        #self.dropout = nn.Dropout(0.5)
        self.layernorm = nn.LayerNorm(hidden_dim)
        
        ## affinity matrix 
        self.v_att_proj = nn.Linear(128, 1024) # 
        self.l_att_proj = nn.Linear(128, 1024)
        self.linear_300 = nn.Sequential(nn.Linear(128, 1024), nn.ReLU(), nn.Linear(1024, output_dim)) # 修改维度
        
        ##

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task
    
    # self-attention for Target extraction
    def attention(self,Q,K,V):
        d_k = K.size(-1)
        scores = torch.matmul(Q,K.transpose(1,2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores,dim=-1)
        context = torch.matmul(alpha_n,V)
        output = context.sum(1)
        
        return output,alpha_n
    #

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        # print('x shape = ', x.shape)
        
        ### drug 
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)  # 'drug output' 

        ### target
        embedded_xt = self.embedding_xt(target)
        output_xt, _ = self.bilstm(embedded_xt)
        
        # self-at 
        Q = self.W_Q(output_xt)     
        K = self.W_K(output_xt)
        V = self.W_V(output_xt)
        attn_output,alpha_n = self.attention(Q, K, V)
        out_ln = self.layernorm(attn_output)
        xt = self.fc(out_ln)  # 'target output' 
        #print("xt:",xt.shape)
        
        ### interaction 
        #_________________________________________________
        # affinity matrix 
        l_att = self.l_att_proj(x)  # 'drug output'    # language_output
        v_att = self.v_att_proj(xt) # 'target output'  # vision_output
        sim_matrix_v2l = torch.matmul(v_att, l_att.transpose(0,1))  # b * v_length * l_length   
                                                                    # sim_matrix_v2l = affinity matrix 
                                                                   # l_att.transpose(1,2)
        kg_output, k = torch.topk(sim_matrix_v2l, dim=-1, k=1)  # ‘Row-Wise Max-Pooling’ 
        # #normalize(abandon)
        # kg_output = F.log_softmax(kg_output,dim=-1)   
        
        # hard attention
        hard_attention_value = gumbel_softmax(kg_output.squeeze())
        head = (xt * hard_attention_value.unsqueeze(-1))# .sum(-2) # xt · (attention value) 

        # soft attention
        # kg_output = F.softmax(kg_output.squeeze(), dim=-1)
        # head = (vision_output * kg_output.unsqueeze(-1)).sum(-2)
        xt_hat = self.linear_300(head)
        xt_sum = xt + xt_hat
        
        #print("x:",x.shape)
        #print("xt_hat:",xt_hat.shape)
       
        #_________________________________________________
        
        # concat
        xc = torch.cat((x, xt_sum), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out, hard_attention_value



def define_mol_encoder(is_freeze=True):
    #mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    #mol_encoder = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    mol_tokenizer = RobertaTokenizer.from_pretrained("model/seyonec_local/ChemBERTa-zinc-base-v1",
                                                     local_files_only=True,
                                                     truncation=True, 
                                                     max_length=552) 
    mol_encoder = AutoModel.from_pretrained("model/seyonec_local/ChemBERTa-zinc-base-v1", local_files_only=True)                                 


    if is_freeze:
        for param in mol_encoder.embeddings.parameters():
            param.requires_grad = False

        for layer in mol_encoder.encoder.layer[:6]: # 固定所有的参数
            for param in layer.parameters():
                param.requires_grad = False

    return mol_tokenizer, mol_encoder

def define_prot_encoder(is_freeze=True):
    #prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False) #do_lower_case = true 忽略大小写 
    #prot_encoder = AutoModel.from_pretrained("Rostlab/prot_bert")
    prot_tokenizer = BertTokenizer.from_pretrained("model/Rostlab_local/prot_bert_bfd", 
                                                   do_lower_case=False,
                                                   local_files_only=True) #do_lower_case = true 忽略大小写 
    prot_encoder = AutoModel.from_pretrained("model/Rostlab_local/prot_bert_bfd",
                                             local_files_only=True)

    if is_freeze:
        for param in prot_encoder.embeddings.parameters():
            param.requires_grad = False

        for layer in prot_encoder.encoder.layer[:6]: # 固定所有的参数
            for param in layer.parameters():
                param.requires_grad = False
       
    return prot_tokenizer, prot_encoder

    

class DTI(nn.Module):
    def __init__(
        self,
        mol_tokenizer,
        prot_tokenizer,
        mol_encoder,
        prot_encoder,
        hidden_dim=512,
        mol_dim=768,
        prot_dim=1024,
        device_no=0,
        theta=0.5
    ):
        super().__init__()
        
        self.theta = theta
        
        # Pretrained block
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.mol_encoder = mol_encoder
        self.prot_encoder = prot_encoder
        

        # 初始化lambda （0-1之间的随机数）
        #self.is_learnable_lambda = is_learnable_lambda

        #if self.is_learnable_lambda and fixed_lambda == -1:
        #    self.lambda_ = torch.nn.Parameter(torch.rand(1).to(f"cuda:{device_no}"), requires_grad=True)
        #elif self.is_learnable_lambda == False and ((fixed_lambda >= 0) and (fixed_lambda <= 1)):
        #    lambda_ = torch.ones(1) * fixed_lambda
        #    self.lambda_ = lambda_.to(f"cuda:{device_no}")
        #print(f"Initial lambda parameter: {self.lambda_}")
        
        # GATA block
        self.GATA = GATA()
        
        
        #
        self.molecule_align = nn.Sequential(nn.LayerNorm(mol_dim), 
                                            nn.Linear(mol_dim, hidden_dim, bias=False) )
        #self.protein_align_teacher = nn.Sequential( nn.LayerNorm(1024), nn.Linear(1024, hidden_dim, bias=False))
        self.protein_align = nn.Sequential(nn.LayerNorm(prot_dim),
                                            nn.Linear(prot_dim, hidden_dim, bias=False))

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cls_out = nn.Linear(hidden_dim, 2)  # 分类，输出维度是2 

    def forward(self,data):
        
        #pretrained block
        SMILES= data.SMILES
        FASTAs = data.FASTA 
        FASTA_list=[]       
        for FASTA in FASTAs:
            FASTA = re.sub(r"[UZOB]", "X", FASTA)
            FASTA_list.append(FASTA)
            
        # 对 SMILES 序列进行截断
        SMILE_list=[]  
        for SMILE in SMILES:
            max_length = 512
            if len(SMILE) > max_length:
                SMILE = SMILE[:max_length]
            SMILE_list.append(SMILE)

        FASTA_input = self.prot_tokenizer(FASTA_list, 
                                          return_tensors='pt')# PyTorch 张量类型
        SMILES_input = self.mol_tokenizer(SMILE_list, 
                                          return_tensors='pt',
                                          padding=True)

        
        mol_feat = self.mol_encoder(**SMILES_input.to("cuda:0")).last_hidden_state[:, 0]# Pretrained drug features
        #print("mol_feat",type(data.mol_feat))
        prot_feat = self.prot_encoder(**FASTA_input.to("cuda:0")).last_hidden_state[:, 0] # Pretrained protein features
        mol_feat = self.molecule_align(mol_feat)
        prot_feat = self.protein_align(prot_feat)
        #prot_feat_teacher = self.protein_align_teacher(prot_feat_teacher).squeeze(1)
        #if self.is_learnable_lambda == True:
        #    lambda_ = torch.sigmoid(self.lambda_)
        #elif self.is_learnable_lambda == False:
        #    lambda_ = self.lambda_.detach()
        #merged_prot_feat = lambda_ * prot_feat + (1 - lambda_) * prot_feat_teacher

        x = torch.cat([mol_feat, prot_feat], dim=1)    
        x = F.dropout(F.gelu(self.fc1(x)), 0.1)
        x = F.dropout(F.gelu(self.fc2(x)), 0.1)
        x = F.dropout(F.gelu(self.fc3(x)), 0.1)
        pre_out = self.cls_out(x) # Pretrained out # .squeeze(-1)
        
        #GATA block
        g_out,hard_attention_value = self.GATA(data) # GATA out
        #print("g_out_type",type(g_out))
        #print("g_out",g_out)
        #print("pre_out_type",type(pre_out))
        #print("pre_out",pre_out)
        
        #回归模型的输出
        #out= g_out* self.theta + pre_out.view(-1, 1) * (1-self.theta)
        
        #分类模型的输出
        #print('pre_out',pre_out)
        #print('g_out',g_out)
        out= g_out + pre_out
        #print('out',out)
        #out = torch.sigmoid(g_out+pre_out.view(-1, 1)) 


        return out


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
