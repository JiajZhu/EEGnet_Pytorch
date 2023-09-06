import ml_collections
import torch.nn as nn
import torch
import numpy as np

def get_config(checkpoint_path=False):
  config = ml_collections.ConfigDict()
  config.checkpoint_path = checkpoint_path
  config.label_0 = ['class0']
  config.label_1 = ['class1']

  config.sampling_rate = 1000 #in Hz
  config.time_end = 1.2 # the list time point (in seconds) of the desired input timepoints to the model
  config.input_size= 1024 # the size of the input to model, 
  # input size 1024 with sampling rate 1000 Hz correspondes to 1.024 seconds
  config.output_size= 2
  config.T1=256
  config.T2=128
  config.F1=32
  config.F2=128
  config.Depth=4
  config.electrodes=58 # the number of electrodes
  config.dropout=.0
  config.Softmax=False

  config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

  config.max_epoch = 30
  config.lr = 1e-5
  config.weight_decay = 5e-4
  config.batch_size = 1 # you could set batch_size to larger values for your own data
  
  config.root = '/content/drive/MyDrive/EEGnet'
  config.augment= False
  return config