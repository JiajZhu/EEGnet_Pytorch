import torch.nn as nn
import torch
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from modelio import LoadableModel, store_config_args


##configuration settings are in file options.py

class EEGNet(LoadableModel):
    @store_config_args
    # def __init__(self,input_size=1024,output_size=2,T1=250,T2=125,F1=32,F2=512,Depth=4,electrodes=58,dropout=.25,Softmax=False):
    def __init__(self,config):
        super(EEGNet, self).__init__()
        self.layer1 = nn.Sequential(
          nn.Conv2d(1,config.F1,(1,config.T1),padding=(0,config.T1//2-1)),
          nn.BatchNorm2d(config.F1),
        )
        self.layer2 = nn.Sequential(
          nn.Conv2d(config.F1,config.Depth*config.F1,(config.electrodes,1)),
          nn.BatchNorm2d(config.Depth*config.F1),
          nn.LeakyReLU(),
          nn.Dropout(config.dropout),
          nn.AvgPool2d(1,4),
        )
        self.layer3 = nn.Sequential(
          nn.Conv2d(config.Depth*config.F1,config.F2,(1,config.T2),padding=(0,config.T2//2-1)),
          nn.Conv2d(config.F2,config.F2,(1,1)),
          nn.BatchNorm2d(config.F2),
          nn.LeakyReLU(),
          nn.AvgPool2d(1,8),
          nn.Dropout(config.dropout)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.F2*config.input_size//32,config.output_size),
            # nn.Linear(1024,config.output_size),
        )
        self.out = nn.Softmax if config.Softmax else nn.Identity()
        # self.fc.apply(self._init_weights)

    def forward(self, x):
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.fc(x)
      x = self.out(x)
      return x

    # def _init_weights(self, m):
    #   if isinstance(m, nn.Linear):
    #       trunc_normal_(m.weight, std=.02)
    #       if isinstance(m, nn.Linear) and m.bias is not None:
    #           nn.init.constant_(m.bias, 0)
    #   elif isinstance(m, nn.LayerNorm):
    #       nn.init.constant_(m.bias, 0)
    #       nn.init.constant_(m.weight, 1.0)



