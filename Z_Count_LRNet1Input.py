# PyTorch
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr

# Models
from Z_LRNet_1Input import LRNet_1Input
from LossFunction import HybridCDLoss

# Other
import os
import numpy as np
import random
from skimage import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import floor, ceil, sqrt, exp
from IPython import display
from itertools import chain
import time
import warnings
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from Method.Utils.ChangeDetectionUtils import get_current_date,check_dir_or_create
from Method.Utils.SaveIndicatorsDuringTraining import save_indicators
from TOOL import *
from Parameters4LRNet import Parameters
parameters=Parameters()
cuda_device_id=parameters.cuda_device_id
print("torch version       ：",torch.__version__)
print("torch cuda available：",torch.cuda.is_available())
print("torch device count  ：",torch.cuda.device_count())
torch.cuda.set_device(cuda_device_id)
print("torch device current：",torch.cuda.current_device())
print('IMPORTS OK')


# Global Variables
TYPE_DATASET = parameters.TYPE_DATASET # 1:DSIFN-Dataset | 2:WHU-Building-Dataset | 3:LEVIR-CD | 4:S2Looking | 5:WHU-BCD
dict_dataset = {1: "DSIFN-Dataset", 2: "WHU-Building-Dataset", 3: "LEVIR-CD", 4: "S2Looking", 5: "WHU-BCD"}
N_CHANNEL = 3
PATH_DATASET = f'../../../../DataRepo/{dict_dataset[TYPE_DATASET]}/'
PATH_TRAIN_DATASET = PATH_DATASET + 'train/'
PATH_VAL_DATASET = PATH_DATASET + 'val/'
PATH_TEST_DATASET = PATH_DATASET + 'test/'
FILE_TRAIN = 'train.txt'   #test_train.txt | train.txt
FILE_VAL = 'val.txt'
FILE_TEST = 'test.txt' #test_test.txt | test.txt
NORMALISE_IMGS = parameters.NORMALISE_IMGS
MODE_NORMALISE = parameters.MODE_NORMALISE
DATA_AUG = parameters.DATA_AUG
VAL_OR_NOT = parameters.ValOrNot

N_EPOCHS = parameters.N_EPOCHS   #50
BATCH_SIZE = parameters.BATCH_SIZE
PATCH_SIDE = parameters.PATCH_SIDE #128 512
LOAD_TRAINED = parameters.LOAD_TRAINED

OUTPUT_RESULT_DIR = f"../../../../ResultRepo/{dict_dataset[TYPE_DATASET]}/LRNet_1Input"
CUR_DATE=get_current_date()

print('DEFINITIONS OK')


# Dataset
if DATA_AUG:
    data_transform = tr.Compose([RandomFlip(), RandomRot()])
else:
    data_transform = None

train_dataset = ChangeDetectionDataset(TYPE_DATASET, path=PATH_TRAIN_DATASET, train_val_test ='train', patch_side = PATCH_SIDE, NORMALISE=NORMALISE_IMGS, MODE_NORMALISE=MODE_NORMALISE, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0, pin_memory=True)   #num_workers = 4
test_dataset = ChangeDetectionDataset(TYPE_DATASET, path=PATH_TEST_DATASET, train_val_test ='test', patch_side = PATCH_SIDE, NORMALISE=NORMALISE_IMGS, MODE_NORMALISE=MODE_NORMALISE, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0, pin_memory=True)   #num_workers = 4
if VAL_OR_NOT==True:
    valid_dataset = ChangeDetectionDataset(TYPE_DATASET, path=PATH_VAL_DATASET, train_val_test ='val', patch_side = PATCH_SIDE, NORMALISE=NORMALISE_IMGS, MODE_NORMALISE=MODE_NORMALISE, transform=data_transform)
    valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0, pin_memory=True)    #num_workers = 4

print(f'DATASETS {dict_dataset[TYPE_DATASET]} OK')


# NETWORK
net, net_name = LRNet_1Input(lrnet_cos_sim_threshold=parameters.lrnet_cos_sim_threshold, lrnet_label_threshold=parameters.lrnet_label_threshold), 'LRNet_1Input'
# print(net)

net.cuda()
criterion = HybridCDLoss(label_smoothing_para_beta=parameters.beta, hard_ratio_para_theta=parameters.theta)

print('NETWORK ' + net_name + ' OK')
NumOfTrainableParameters=count_parameters(net)
print('Number of trainable parameters:', NumOfTrainableParameters)

I1, I2, cm, cm32 = train_dataset.get_img(train_dataset.names[0])
I1 = torch.unsqueeze(I1, 0).float().cuda()
I2 = torch.unsqueeze(I2, 0).float().cuda()
tensor = torch.cat([I1, I2], dim=1)

flops = FlopCountAnalysis(net, tensor)
print("FLOPs: ", flops.total())
print("Paras: ",parameter_count_table(net))