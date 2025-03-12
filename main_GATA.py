import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from model.model_ import *
import argparse

datasets = ['davis']#[int(sys.argv[1])]]
modeling = GCNNet 
model_st = 'GCNNet'

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000
seed_torch(seed)
seed_torch(seed)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    
    parser = argparse.ArgumentParser(description="GATA for DTI prediction") # start 
     # Add argument
    parser.add_argument('--config', required=True, help="path to config file", type=str)
    parser.add_argument('--data', required=True, type=str, metavar='TASK',help='dataset')
    args = parser.parse_args() # end 




#_________________________________________

#-------------------------------------------


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def main():
    


mol_tokenizer, mol_encoder = define_mol_encoder(is_freeze=True)
prot_tokenizer, prot_encoder = define_prot_encoder(is_freeze=True)

datasets = ['davis']#[int(sys.argv[1])]]  #,'kiba'
modeling = GCNNet #[int(sys.argv[2])] 
        # [GINConvNet, GATNet, GAT_GCN, GCNNet]
model_st = 'GCNNet'

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)