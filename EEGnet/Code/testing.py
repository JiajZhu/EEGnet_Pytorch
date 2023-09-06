import os, glob
import sys
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import CustomDataset
from model import EEGNet

def main(config):
    print('##############################################################')
    print("Configuration:")
    print(config)
    print('##############################################################')
    

    exp_path = config.root+'/experiment/' 
    save_dir = "test_sub_up"
    if not os.path.exists(exp_path + save_dir):
        os.makedirs(exp_path + save_dir)
    LossPath = os.path.join(exp_path + save_dir)


    epoch_start = 0
    max_epoch = config.max_epoch #max traning epoch
    '''
    If continue from previous training
    '''
    # config.dropout=.0
    print('Using checkpoint: ', config.checkpoint_path)
    model = EEGNet(config).to(config.device)


    ##############################################################################
    # Initialize dataloader
    ##############################################################################
    print("loading data")
    dataset_test = CustomDataset(config,train=False)
    dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False,pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    if config.checkpoint_path:
      checkpoint = torch.load(config.checkpoint_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      print('load checkpoint from',config.checkpoint_path)
    else:
      print("Error in main:no checkpoint provided!")
      sys.exit()
    criterion = nn.CrossEntropyLoss()   

 

    ##############################################################################
    # Learning curve
    ##############################################################################
    
    print('LossPath',LossPath)
    with open(os.path.join(LossPath + '/test_results.csv'), 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Test loss','Test acc'])   
        f.close()
    ##################################################################################################################################
    ########################################################### Validation ###########################################################
    ##################################################################################################################################
    with torch.no_grad():
        model.eval()
        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        epoch_train_acc = []
        for (data,label) in tqdm(dataloader_test):
            step_start_time = time.time()
            prediction = model(data.float().to(config.device))
            loss = criterion(prediction,label.to(config.device))
            acc = (prediction.argmax(dim=-1) == label.to(config.device)).float().mean()
            epoch_loss.append(loss)
            epoch_total_loss.append(loss.item())
            epoch_train_acc.append(acc)
            epoch_step_time.append(time.time() - step_start_time)

        test_loss = sum(epoch_total_loss) / len(epoch_total_loss)
        test_acc = sum(epoch_train_acc) / len(epoch_train_acc)
        print(f"loss = {test_loss:.5f},acc = {test_acc:.5f}")

    ##############################################################################
    # Learning curve
    ##############################################################################
    with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['%.15f' % test_loss,'%.15f' % test_acc])
        f.close()
  
  
