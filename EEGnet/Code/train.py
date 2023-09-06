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
    
    # define saving path
    exp_path = config.root+'/experiment/' 
    save_dir = 'classification'
    if not os.path.exists(exp_path + save_dir):
        os.makedirs(exp_path + save_dir)
    LossPath = os.path.join(exp_path + save_dir)

    '''
    If continue from previous training
    '''
    print('Using checkpoint: ', config.checkpoint_path)
    # Initialize the model
    model = EEGNet(config).to(config.device)
    epoch_start = 0
    max_epoch = config.max_epoch #max traning epoch

    ##############################################################################
    # Initialize dataloader
    ##############################################################################
    print("loading data")
    dataset_train = CustomDataset(config,train=True)
    dataset_test = CustomDataset(config,train=False)
    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    
    if config.checkpoint_path:
      checkpoint = torch.load(config.checkpoint_path)
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      epoch_start = checkpoint['epoch']+1
      model.load_state_dict(checkpoint['model_state_dict'])
      print('load checkpoint from',config.checkpoint_path)
    else:
      print("No chekcpoint loaded")
    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=config.lr,weight_decay=config.weight_decay)
    stop_num = 20
    early_stop = stop_num
    ##############################################################################
    # Setup Learning curve file
    ##############################################################################
    
    print('LossPath',LossPath)
    with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Current Epoch','Train loss','Train acc','Test loss','Test acc'])   
        f.close()
    
    print('Start training ...')
    torch.cuda.empty_cache()
    best_loss = np.inf
    ################################################################################################################################
    ########################################################### Training ###########################################################
    ################################################################################################################################
    for epoch in range(epoch_start, max_epoch):
        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        epoch_train_acc = []
        step = 0
        for (data,label) in tqdm(dataloader_train):
            step += 1
            step_start_time = time.time()
            model.train()
            optimizer.zero_grad()
            prediction = model(data.float().to(config.device))
            loss = criterion(prediction,label.to(config.device))
            # print("data",data.shape)
            # print("prediction",prediction)
            # print("label",label)
            epoch_loss.append(loss)
            
            loss.backward()
            optimizer.step()
            acc = (prediction.argmax(dim=-1) == label.to(config.device)).float().mean()
            # if step%1 == 0:
            #     grad_norms = [torch.norm(p.grad) for p in model.parameters()]
            #     print(f"Gradient norms at iteration {step}: {grad_norms}")
            #     print('steps {0:d} - training loss {1:.4f}- acc {2:.4f}'.format(step,loss.item(),acc))
                

            epoch_train_acc.append(acc)
            epoch_step_time.append(time.time() - step_start_time)
            epoch_total_loss.append(loss.item())
        train_loss = sum(epoch_total_loss) / len(epoch_total_loss)
        train_acc = sum(epoch_train_acc) / len(epoch_train_acc)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{max_epoch:03d} ] loss = {train_loss:.5f},acc = {train_acc:.5f}")

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
            print(f"[ Valid | {epoch + 1:03d}/{max_epoch:03d} ] loss = {test_loss:.5f},acc = {test_acc:.5f}")

        ##############################################################################
        # Record Learning curve
        ##############################################################################
        with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%d' %(epoch + 1), '%.15f' % train_loss,'%.15f' % train_acc, '%.15f' % test_loss,'%.15f' % test_acc])
            f.close()
            

        #############################################################################
        # Save best model and early stop
        #############################################################################

        if test_loss < best_loss:
            best_epoch = epoch
            best_loss = test_loss
            # model.save(os.path.join(LossPath, 'Best_model.pt'))
            early_stop = stop_num
        else:
            early_stop = early_stop - 1
        if epoch % 1 == 0:    ### update
            # model.save(os.path.join(LossPath, 'Best_{}_model.pt'.format(epoch)))  
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, os.path.join(LossPath, 'CheckPoint_{}_model.pt'.format(epoch)))
        if early_stop and epoch < max_epoch - 1:  
            continue  
        else:
          print('early stop at'+str(epoch)+'epoch')
          break  
        # model.save(os.path.join(LossPath, 'model.pt'))  
