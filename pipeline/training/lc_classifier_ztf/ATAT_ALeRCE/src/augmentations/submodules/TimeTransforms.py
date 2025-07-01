from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy
from .WindowApply import WindowApply

class TimeFactor:
    def __init__(self,num_bands = 2,factor =0.5, apply_to_classes: list = None):
        self.factor = factor
        self.num_bands = num_bands
        self.apply_to_classes = apply_to_classes
    def __call__(self, sample):
        if self.apply_to_classes is None:
            if isinstance(self.factor,list):
                return self.random_factor(sample)
            else:
                for i in range(self.num_bands): 
                    band_time = sample['time'][:,i] * self.factor
                    sample['time'][:, i] = band_time
            
        else:
            if sample['labels'] in self.apply_to_classes:
                if isinstance(self.factor,list):
                    return self.random_factor(sample)
                for i in range(self.num_bands): 
                    band_time = sample['time'][:,i] * self.factor
                    sample['time'][:, i] = band_time
        return sample

    def random_factor(self,sample):
        factor = np.random.choice(self.factor)
        for i in range(self.num_bands):
            band_time = sample['time'][:,i] * factor
            sample['time'][:, i] = band_time
        return sample
    
from scipy.ndimage import gaussian_filter1d

class TimeGaussianFilter:
    def __init__(self, num_bands,filter_std:list, apply_to_classes:list = None):
        self.num_bands =num_bands
        self.filter_std = filter_std
        self.apply_to_classes = apply_to_classes
    def __call__(self,sample):
        if self.apply_to_classes is None:
            self.gauss_filter(sample)
        else: #apply only if correct class
            if sample['labels'] in self.apply_to_classes:
                 self.gauss_filter(sample)
        return sample
    
    def gauss_filter(self, sample):
        for i in range(self.num_bands):
                choose_filter_std = np.random.choice(self.filter_std)
                if choose_filter_std == -1:
                    return sample
                nonzero = torch.count_nonzero(sample['time'][:,i])
                filtered_signal = gaussian_filter1d(sample['time'][:nonzero, i], choose_filter_std)
                sample['time'][:nonzero,i] = torch.tensor(filtered_signal, dtype = torch.float)




class TimeGaussianNoise:
    def  __init__(self, num_bands, apply_to_classes: list = None):
        super().__init__()
        self.num_bands = num_bands
        self.apply_to_classes = apply_to_classes
    def __call__(self, sample):
        if self.apply_to_classes is None:
            for i in range(self.num_bands):
                if torch.count_nonzero(sample['time'][:,i], dim = -1)  > 0:
                    band_time = sample['time'][:,i]
                    nonzero = torch.nonzero(band_time)
                    band_mask = band_time!=0
                    noise = torch.normal(0,abs(band_time[nonzero].mean())*(1e-3), size=(band_time.size(0),)).to(device=band_time.device, non_blocking=True) 
                    noise.sort()
                    band_time = band_time + noise * band_mask
                    sample["time"][:,i] = band_time
                else:
                    continue
        else:
            if sample['labels'] in self.apply_to_classes:
                for i in range(self.num_bands):
                    if torch.count_nonzero(sample['time'][:,i], dim = -1)  > 0:
                        band_time = sample['time'][:,i]
                        nonzero = torch.nonzero(band_time)
                        band_mask = band_time!=0
                        noise = torch.normal(0,abs(band_time[nonzero].mean())*(1e-4), size=(band_time.size(0),)).to(device=band_time.device, non_blocking=True) 
                        noise.sort()
                        band_time = band_time + noise * band_mask
                        sample["time"][:,i] = band_time
                    else:
                        continue
        return sample
    
