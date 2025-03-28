from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy
from .WindowApply import WindowApply

class TimeFactor(WindowApply):
    def __init__(self,num_bands = 2,window = 10, factor =0.5):
        self.factor = factor
        self.num_bands = num_bands
        self.window = window
        
    def __call__(self, sample):

        if isinstance(self.factor,list):
            return self.random_factor(sample)
        for i in range(self.num_bands): 
            band_time = sample['time'][:,i] *self.factor
            band_mask = (sample['data'][:,i]!=0).bool()
            if band_mask.dim() > 1:
                max_sample_n = int(band_mask[i].sum().item())  # Convert tensor to int
            else:
                max_sample_n = int(band_mask.sum().item())  # Convert tensor to int
            if self.window > 0:
                
                sample['time'][:, i] = self.apply_to_window(
                    band_time, sample['time'][:, i], 
                    self.window, max_sample_n
                )
            else:
                sample['time'][:, i] = band_time
        return sample

    def random_factor(self,sample):
        factor = np.random.choice(self.factor)
        for i in range(self.num_bands):
            band_time = sample['time'][:,i] * factor
            band_mask = (sample['data'][:,i]!=0).bool()
            if band_mask.dim() > 1:
                max_sample_n = int(band_mask[i].sum().item())  # Convert tensor to int
            else:
                max_sample_n = int(band_mask.sum().item())  # Convert tensor to int
            if self.window > 0:
                
                sample['time'][:, i] = self.apply_to_window(
                    band_time, sample['time'][:, i], 
                    self.window, max_sample_n
                )
            else:
                sample['time'][:, i] = band_time
        return sample
  
class TimePoissonNoise(WindowApply):
    def __init__(self, num_bands, rate, window=10):
        super().__init__()
        self.num_bands = num_bands
        self.rate = rate
        self.window = window
        self.Tmax = 1000
    
    def __call__(self, sample):
        
        new_time = sample['time'].clone()
        data_mask = (sample['data'] != 0).bool()
        for i in range(self.num_bands):
            band_time = new_time[:, i]
            band_mask = data_mask[:, i]
            
            noise = torch.poisson(torch.full_like(band_time, self.rate)) * torch.randint(0,self.Tmax,size = (1,))
            noise[0] = 0
            noise_max = noise.max()
            normalized_noise = noise / torch.clamp(noise_max, min=1e-10)
            torch.sort(normalized_noise)
            
            noisy_time = band_time + normalized_noise #* band_time.mean()
            max_sample_n = int(band_mask.sum().item())
            if self.window > 0:
                new_time[:, i] = self.apply_to_window(
                    noisy_time, sample['time'][:, i], 
                    self.window, max_sample_n
                )
            else:
                new_time[:, i] = noisy_time
        sample['time'] = new_time
        sample.update({'time':new_time})
        #print(sample['time'])
        return sample
    
    
    

class Exptime(WindowApply):
    def __init__(self,num_bands,window = 10,Tmax = 1000.0):
        super().__init__()
        self.window = window
        self.num_bands = num_bands
        
    def __call__(self, sample):
        for i in range(self.num_bands):
            band_time = sample['time'][:,i]
            band_mask = (sample['data'][:,i]!=0).bool()
            max_sample_n = band_mask.sum(axis = 0)
            sum_exp = torch.exp(band_time/1000.0) 
            band_time = (band_time + sum_exp) * band_mask
            if self.window > 0:
                sample['time'][:,i] = self.apply_to_window(band_time, sample['time'][:,i], self.window, max_sample_n)
            else:
                sample["time"][:,i] = band_time
        return sample



class TimeGaussianNoise(WindowApply):
    def  __init__(self, num_bands, mean = 0,std = 1,window = 10):
        self.num_bands = num_bands
        self.mean = mean
        self.std = std
        self.window = window
    def __call__(self, sample):
        for i in range(self.num_bands):
            band_time = sample['time'][:,i]
            band_mask = (sample['time'][:,i]!=0).bool()
            if band_mask.dim() > 1:
                max_sample_n = int(band_mask[i].sum().item())  # Convert tensor to int
            else:
                max_sample_n = int(band_mask.sum().item())  # Convert tensor to int
            
            noise = noise = torch.normal(self.mean,self.std, size=(sample['time'].shape[0],)).to(device=sample['time'].device, non_blocking=True) 
             
            noise.sort()
            band_time = (band_time + noise*band_time.mean()) * band_mask
            
            if self.window > 0:
                sample['time'][:,i] = self.apply_to_window(band_time, sample['time'][:,i], self.window, max_sample_n)
            else:
                sample["time"][:,i] = band_time
        return sample

class TimeShift(WindowApply):
    def __init__(self,num_bands = 2,window = 10, min_scale=0, max_scale=10, Tmax = 1000.0):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.window = window
        self.Tmax = Tmax
        self.num_bands = num_bands
        
    def __call__(self, sample):
        for i in range(self.num_bands): 
            band_time = sample['time'][:,i] 
            band_mask = (sample['data'][:,i]!=0).bool()
            factor = torch.FloatTensor(1).uniform_(2*torch.pi*self.min_scale, 2*torch.pi* self.max_scale) / self.Tmax
            band_time = band_time + factor * band_mask
            if band_mask.dim() > 1:
                max_sample_n = int(band_mask[i].sum().item())  # Convert tensor to int
            else:
                max_sample_n = int(band_mask.sum().item())  # Convert tensor to int
            if self.window > 0:
                sample['time'][:,i] = self.apply_to_window(band_time, sample['time'][:,i], self.window, max_sample_n)
            else:
                sample["time"][:,i] = band_time
        return sample
  