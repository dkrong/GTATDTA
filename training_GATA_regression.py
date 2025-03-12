import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
import wandb
from preprocessing_GATA import *
from torch_geometric.data import InMemoryDataset #, DataLoader
from torch_geometric.loader import DataLoader
from model.TRM2_cnn import TRM2_cnn
from model.TRM2_lstm_cnn import TRM2_lstm_cnn
from model.TRM2_lstm_sat import TRM2_lstm_sat
from model.TRM2_lstm_sat_am import TRM2_lstm_sat_am
from model.model_ import * # our model 
import re
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

cuda_name = "cuda:0"
datasets = ['davis']#[int(sys.argv[0])]
# 初始化wandb
wandb.init(project="GASA_test",
           entity="rongdk2020") # username
# 配置wandb 
wandb.config = {'seed':0, 
                'modeling':DTI,       # 传入模型
                'model_st':'DTI',     # 模型名字
                'cuda_name':'cuda:0', #  配置
    'Learning rate':0.0005,
    'Epochs':1000,
    'LOG_INTERVAL':20,
    'TRAIN_BATCH_SIZE':64,
    'TEST_BATCH_SIZE':64,
}

#----------------------------
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)
        #print("output_type",type(output)) # tuple
        #print("data.y",data.y.view(-1, 1).float()) #tensor
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % wandb.config['LOG_INTERVAL'] == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            wandb.log({'Train epoch':epoch,'loss': loss.item()})

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data).to(device)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

####


mol_tokenizer, mol_encoder = define_mol_encoder(is_freeze=True)
prot_tokenizer, prot_encoder = define_prot_encoder(is_freeze=True)
####
task= 'regression'# cold_drug\cold_all
# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', wandb.config['model_st'] + '_' + dataset )
    processed_data_file_train = 'data_raw/' + dataset+'/cold_target/processed/processed_data_train.pt'
    processed_data_file_test = 'data_raw/' + dataset +'/cold_target/processed/processed_data_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = GATADataset(root='data_raw/davis/cold_target')
        test_data = GATADataset(root='data_raw/davis/cold_target')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, 
                                  batch_size=wandb.config['TRAIN_BATCH_SIZE'], 
                                  shuffle=True)
        test_loader = DataLoader(test_data, 
                                 batch_size=wandb.config['TEST_BATCH_SIZE'], 
                                 shuffle=False)
             #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print('Train size:', len(train_loader))
        #print('Test size:', len(val_loader))
        print('Test size:', len(test_loader))


        # training the model
        device = torch.device(wandb.config['cuda_name'] if torch.cuda.is_available() else "cpu")
        model = wandb.config['modeling'](mol_tokenizer,
                         prot_tokenizer,
                         mol_encoder,
                         prot_encoder).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['Learning rate'])
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        result_files='results_GASA/'
        model_file_name = result_files+task+'model_' + wandb.config['model_st'] + '_' + dataset +  '.model'
        result_file_name = result_files+task+'result_' + wandb.config['model_st'] + '_' + dataset +  '.csv'
        for epoch in range(wandb.config['Epochs']):
            train(model, device, train_loader, optimizer, epoch+1)
            G,P = predicting(model, device, test_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P),get_rm2(G,P)]
            if ret[1]<best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret)))
                best_epoch = epoch+1
                best_mse = ret[1]
                best_ci = ret[-2]
                best_rm2 = ret[-1]
                print('rmse improved at epoch ',
                      best_epoch, 
                      '; best_mse,best_ci:', 
                      best_mse,best_ci,
                      'get_rm2',
                      best_rm2,
                      wandb.config['model_st'],dataset)
            else:
                print(ret[1],'No improvement since epoch ', best_epoch, 
                      '; best_mse,best_ci:', best_mse,best_ci,
                      'get_rm2',best_rm2,
                      wandb.config['model_st'],dataset)

