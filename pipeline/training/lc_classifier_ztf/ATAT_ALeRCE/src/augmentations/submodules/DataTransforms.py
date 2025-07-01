
from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy

from scipy.ndimage import gaussian_filter1d

class InverseLC:
    def __call__(self,sample):
        sample['data'] = -sample['data']
        return sample

class CutBand:
    def __init__(self, num_bands, min_samples = 6):
        self.num_bands =num_bands
        self.min_samples = min_samples
    def __call__(self, sample):
        nonzero = torch.count_nonzero(sample['data'], dim = 1)
        if all([nonzero[0] <= self.min_samples, nonzero[1] <= self.min_samples, nonzero[0] + nonzero[1] <= self.min_samples]):
            return sample
        band = np.random.choice(list(range(self.num_bands)))
        sample['data'][:, band] = 0
        sample['time'][:, band] = 0
        sample['mask'][:, band] = 0
        return sample


class GaussianFilter:
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
                nonzero = torch.count_nonzero(sample['data'][:,i])
                filtered_signal = gaussian_filter1d(sample['data'][:nonzero, i], choose_filter_std)
                sample['data'][:nonzero,i] = torch.tensor(filtered_signal, dtype = torch.float)

        
class Factor:
    def __init__(self, factor = 0.5,apply_to_classes:list = None):
        self.factor = factor
        self.apply_to_classes = apply_to_classes

        assert any([isinstance(factor, int), isinstance(factor,list)])
    def __call__(self, sample):
        
        if self.apply_to_classes is None:
            if isinstance(self.factor,list):
                return self.random_factor(sample)
            
            data = sample['data']
            sample['data'] = (data * self.factor)
            
        else:
            if sample['labels'] in self.apply_to_classes:
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
 

class GaussFactor:
    def __init__(self,num_bands = 2,scale = 1e-3, apply_to_classes:list = None):
        self.apply_to_classes = apply_to_classes
        self.num_bands  = num_bands
        self.scale = scale
        assert self.scale <=1, 'scale must be <=1'
    def __call__(self, sample):
        for band in range(self.num_bands):
            band_data = sample['data'][:, band]
            scalars = torch.normal(0,self.scale, size=(band_data.size(0),)).to(device=band_data.device, non_blocking=True) +1
            if self.apply_to_classes is None:
                sample['data'][:,band] = (band_data * scalars)
            else:
                if sample['labels'] in self.apply_to_classes:
                    sample['data'][:,band] = (band_data * scalars)
        return sample 
 
class GaussTimeFactor:
    def __init__(self,num_bands = 2,scale = 1e-3, apply_to_classes:list = None):
        self.apply_to_classes = apply_to_classes
        self.num_bands  = num_bands
        self.scale = scale
        assert self.scale <=1, 'scale must be <=1'
    def __call__(self, sample):
        for band in range(self.num_bands):
            band_data = sample['time'][:, band]
            scalars = torch.normal(0,self.scale, size=(band_data.size(0),)).to(device=band_data.device, non_blocking=True) +1
            scalars.sort()
            if self.apply_to_classes is None:
                sample['time'][:,band] = (band_data * scalars)
            else:
                if sample['labels'] in self.apply_to_classes:
                    sample['time'][:,band] = (band_data * scalars)
        return sample 
 
class GaussianNoise:
    def  __init__(self, num_bands):
        super().__init__()
        self.num_bands = num_bands
       
    def __call__(self, sample):
        for i in range(self.num_bands):
            if torch.count_nonzero(sample['data'][:,i], dim = -1)  > 0:

                band_data = sample['data'][:,i]
                nonzero = torch.nonzero(band_data)
                band_mask = band_data!=0
                noise = torch.normal(0,abs(band_data[nonzero].mean())*(0.01), size=(band_data.size(0),)).to(device=band_data.device, non_blocking=True) 
                band_data = band_data + noise * band_mask
                sample["data"][:,i] = band_data
            else:
                continue
        return sample
 

class RandomSwapAdjacentRows:
    def __init__(self, p = 0.5):
        self.p = p
    def __call__(self, sample):
        """
        Args:
            sample (Tensor): A 2D sample of shape (N, M), where N is even.

        Returns:
            Tensor: Tensor with some adjacent rows randomly swapped.
        """
        if sample['data'].dim() != 2:
            raise ValueError("Input sample['data'] must be 2D.")
        n = sample['data'].size(0)
        if n % 2 != 0:
            raise ValueError("Number of rows must be even.")

        sample['data'] = sample['data'].clone()
        do_swap = torch.rand(n // 2) < self.p

        for i in range(n // 2):
            if do_swap[i]:
                i0, i1 = 2 * i, 2 * i + 1
                sample['data'][[i0, i1]] = sample['data'][[i1, i0]]

        return sample