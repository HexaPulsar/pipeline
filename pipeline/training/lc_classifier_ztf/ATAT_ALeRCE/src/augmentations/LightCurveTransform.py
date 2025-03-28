from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy
from .submodules import *


class WindowSelect:
    def __init__(self,num_bands:int,window_size:int):
        self.num_bands = num_bands
        self.window_size = window_size
    def __call__(self,sample:dict): 
        
        
        for i in range(self.num_bands):
             
            nonzero_measures = torch.count_nonzero(sample['data'][:,i], dim = 0)
            intersection_check = nonzero_measures- self.window_size
            if  ((sample['data'][:,i]!=0 ).sum()<= self.window_size):
                #print('triggered interesection check')
                continue
            else:
                start = torch.randint(0,intersection_check ,size = (1,)) 
                end = start + self.window_size
                new_mask = torch.zeros_like(sample['data'][:,i])
                new_mask[start:end] = 1
                sample['mask'][:,i]= new_mask
                sample["data"][:,i] = sample['mask'][:,i] * sample["data"][:,i]
                sample["time"][:,i] = sample['mask'][:,i] * sample["time"][:,i]
                
        assert sample['mask'].sum() !=0
        return sample

    