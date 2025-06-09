
from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy
from .WindowApply import WindowApply
from scipy.ndimage import gaussian_filter1d

class InverseLC:
    def __call__(self,sample):
        sample['data'] = -sample['data']
        return sample


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
    
    
class CutBand:
    def __init__(self, num_bands):
        self.num_bands =num_bands

    def __call__(self, sample):
        nonzero = torch.count_nonzero(sample['data'], dim = 1)
        if all([nonzero[0] < 6, nonzero[1] < 6]):
            return sample
        elif nonzero[0] < 6 and nonzero[1] >=6:
            sample['data'][:,0] = 0
            sample['time'][:,0] = 0
            sample['mask'][:,0] = 0
            return sample
        elif nonzero[0] >= 6 and nonzero[1] <6:
            sample['data'][:,1] = 0
            sample['time'][:,1] = 0
            sample['mask'][:,1] = 0
            return sample
        else:
            band = np.random.choice(list(range(self.num_bands)))
            sample['data'][:, band] = 0
            sample['time'][:, band] = 0
            sample['mask'][:, band] = 0
            return sample
    
class GaussianFilter:
    def __init__(self, num_bands,filter_std:list):
        self.num_bands =num_bands
        self.filter_std = filter_std
    def __call__(self,sample):
        for i in range(self.num_bands):
            choose_filter_std = np.random.choice(self.filter_std)
            nonzero = torch.count_nonzero(sample['data'][:,i])
            filtered_signal = gaussian_filter1d(sample['data'][:nonzero, i], choose_filter_std)
            sample['data'][:nonzero,i] = torch.tensor(filtered_signal, dtype = torch.float)
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



class GaussianNoise:
    def  __init__(self, num_bands, window = 10,std = 0.1):
        super().__init__()
        self.num_bands = num_bands
        self.window = window
        self.std = std
       
    def __call__(self, sample):
        for i in range(self.num_bands):
            band_data = sample['data'][:,i]
            band_mask = band_data!=0
            noise = torch.normal(0,self.std, size=(band_data.size(0),)).to(device=band_data.device, non_blocking=True) 
            band_data = band_data + noise * band_mask

            sample["data"][:,i] = band_data
        return sample



class Shuffle:
    def  __init__(self, num_bands):
        super().__init__()
        self.num_bands = num_bands
       
    def __call__(self, sample):
        for i in range(self.num_bands):
            sample["data"][:,i] = torch.permute(sample['data'][:,i],dims = (0,))
        return sample


class ClipSignal:
    def  __init__(self, num_bands, min = -10, max = 10):
        super().__init__()
        self.num_bands = num_bands
        self.min = min
        self.max = max
       
    def __call__(self, sample):
        for i in range(self.num_bands):
            sample["data"][:,i] = torch.clip(sample['data'][:,i],min = self.min, max = self.max)
        return sample