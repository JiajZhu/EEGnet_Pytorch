# EEGnet_Pytorch
This is a pytorch implementation of [EEGnet](https://arxiv.org/abs/1611.08024) that could easily run on google colab 


To run this code, simply upload it to google drive, then run the script main_script.ipynb, which will train the EEGnet with default hyperparmeters on our sample data. The learning information and checkpoints will be sotred in experiment\classification. 

For more information about running deep learning code with colab, see [this](https://neptune.ai/blog/how-to-use-google-colab-for-deep-learning-complete-tutorial) 


The EEG model is defined in EEGnet/Code/model.py. Dataset is defined in EEGnet/Code/dataset.py. 
All hyperparameters, traning settings and other configurational settings are in EEGnet/Code/options.py. You could change these settings when training on your own data, and upload your own data to folder Data\class0 (which is labeled as 0 when traning and testing) and Data\class1 (labeled as 1).
