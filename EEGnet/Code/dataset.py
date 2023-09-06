import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os,glob
import numpy as np
import random
def make_dataset(root,dir_list):
  filepath=[]
  for dir_i in dir_list:
    filepath_tmp = sorted(glob.glob(root+dir_i + '/*.csv'))
    filepath.extend(filepath_tmp)
  return filepath
class CustomDataset(Dataset):
    def __init__(self,config,train):
        self.augment = config.augment
        self.label0 = config.label_0
        self.label1 = config.label_1
        self.datatype = "train" if train else "test"
        self.input_size = config.input_size
        self.time_indices_end =  int((config.time_end)*config.sampling_rate)
        self.time_indices_start = self.time_indices_end - self.input_size
        self.data_root = config.root + '/Data/' + self.datatype +'/'
        self.file_paths = []
        self.file_paths_0 = make_dataset(self.data_root,self.label0)
        self.file_paths_1 = make_dataset(self.data_root,self.label1)
        self.sample_num_0 = len(self.file_paths_0)
        self.sample_num_1 = len(self.file_paths_1)
        self.file_paths.extend(self.file_paths_0)
        self.file_paths.extend(self.file_paths_1)
        # self.label0_dir = self.data_root + self.label0
        # self.label1_dir = self.data_root + self.label1
        # self.file_paths_0=sorted(glob.glob(self.label0_dir + '/*.csv'))
        # self.file_paths_1=sorted(glob.glob(self.label1_dir + '/*.csv'))
        # self.file_paths = []
        # self.file_paths.extend(self.file_paths_0)
        # self.file_paths.extend(self.file_paths_1)
        # print('loading label0 from {}, label1 from {}'.format(self.file_paths_0,self.file_paths_1))
        if self.augment and self.datatype == "train":
          print("Augmenting data")
        else:
          print("Data not augmented")
    def __getitem__(self, index):
        if index < self.sample_num_0:
          name = self.file_paths_0[index]
          label = 0
        else:
          name = self.file_paths_1[index-self.sample_num_0]
          label = 1
        # print("name",name)
        dataframe = pd.read_csv(name,header=None)
        data = dataframe.values[:,self.time_indices_start:self.time_indices_end]
        data = data[None,:,:]
        # data= np.delete(data, [39,40], axis=0)
        ##Data augmenting
        if self.augment and self.datatype == "train":
          low = -3.0  
          high = 3.0  
          std = 0.5
          random_number = np.random.uniform(low, high)
          data += np.random.randn(*data.shape)*std
          data += random_number
        # if name.split("__")[-1].split('.')[0] in self.label0:
        #   label = 0
        # elif name.split("__")[-1].split('.')[0] in self.label1:
        #   label = 1
        # else:
        #   print("Error in dataset, unknow filename suffix, can not determine label,filename is \n {}".format(name))
        #   label = np.random.randint(2)

        return data,label

    def __len__(self):
        # Return the total number of items in the dataset
        sample_num_all = self.sample_num_0 + self.sample_num_1
        return sample_num_all