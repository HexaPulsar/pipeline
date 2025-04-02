
from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy
from .WindowApply import WindowApply
from scipy.ndimage import gaussian_filter1d
class ShiftData(WindowApply):
    def __init__(self,num_bands,window = -1):
        super().__init__()
        self.window = window
        self.num_bands = num_bands
    def __call__(self, sample):
        for i in range(self.num_bands):
            band = sample['data'][:,i]
            band_mask = (sample['data'][:,i]!=0).bool()
            max_sample_n = band_mask.sum(axis = 0)
            sum_exp = band + torch.rand(band.shape) * band.mean() - torch.rand(band.shape) * band.mean()
            band = (band + sum_exp) * band_mask
            if self.window > 0:
                sample['data'][:,i] = self.apply_to_window(band, sample['data'][:,i], self.window, max_sample_n)
            else:
                sample["data"][:,i] = band
        return sample
    
    
class GaussianFilter:
    def __init__(self, num_bands,filter_std):
        self.num_bands =num_bands
        self.filter_std = filter_std
    def __call__(self,sample):
        
        for i in range(self.num_bands):
            filtered_signal = gaussian_filter1d(sample['data'][:,i], self.filter_std)
            filtered_signal = torch.tensor(filtered_signal)* (sample['data'] != 0)[:,i]
            sample['data'][:,i] = filtered_signal
        return sample
        
    
class Factor:
    def __init__(self, factor = 0.5):
        self.factor = factor
          
    def __call__(self, sample):
        
        if isinstance(self.factor,list):
            return self.random_factor(sample)
        
        data = sample['data']
        sample['data'] = (data * self.factor)
        return sample
    
    def random_factor(self,sample):
        
        data = sample['data']
        choose = np.random.choice(self.factor)
        sample['data'] =  (data * choose)
        return sample
    


class CosData(WindowApply):
    def __init__(self,num_bands,window = -1,Tmax = 1000.0):
        super().__init__()
        self.window = window
        self.num_bands = num_bands
        
    def __call__(self, sample):
        for i in range(self.num_bands):
            band = sample['data'][:,i]
            band_mask = (sample['data'][:,i]!=0).bool()
            max_sample_n = band_mask.sum(axis = 0)
            sum_exp = torch.cos(band/1000.0) 
            band = (band + sum_exp) * band_mask
            if self.window > 0:
                sample['data'][:,i] = self.apply_to_window(band, sample['data'][:,i], self.window, max_sample_n)
            else:
                sample["data"][:,i] = band
        return sample


class SinData(WindowApply):
    def __init__(self,num_bands,window = -1,Tmax = 1000.0):
        super().__init__()
        self.window = window
        self.num_bands = num_bands
        
    def __call__(self, sample):
        for i in range(self.num_bands):
            band = sample['data'][:,i]
            band_mask = (sample['data'][:,i]!=0).bool()
            max_sample_n = band_mask.sum(axis = 0)
            sum_exp = torch.sin(band/1000.0) 
            band = (band + sum_exp) * band_mask
            if self.window > 0:
                sample['data'][:,i] = self.apply_to_window(band, sample['data'][:,i], self.window, max_sample_n)
            else:
                sample["data"][:,i] = band
        return sample



class GaussianNoise(WindowApply):
    def  __init__(self, num_bands, window = 10):
        super().__init__()
        self.num_bands = num_bands
        self.window = window
       
    def __call__(self, sample):
        for i in range(self.num_bands):
            band_data = sample['data'][:,i]
            band_mask = band_data!=0
            max_sample_n = band_mask.sum(axis = 0)
            noise = torch.normal(0,1, size=(band_data.size(0),)).to(device=band_data.device, non_blocking=True) 
            band_data = (band_data + noise* band_data.mean()) * band_mask
            if self.window > 0:
                sample['data'][:,i] = self.apply_to_window(band_data, sample['data'][:,i], self.window, max_sample_n)
            else:
                sample["data"][:,i] = band_data
        return sample
