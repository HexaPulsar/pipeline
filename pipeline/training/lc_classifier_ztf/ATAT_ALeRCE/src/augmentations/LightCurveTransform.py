from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy
from .submodules import *
import torch


class Roll:
    def __init__(self, num_bands, max_roll = 50, apply_to_classes:list = None):
        self.num_bands = num_bands
        self.max_roll = max_roll
        self.apply_to_classes = apply_to_classes

    def __call__(self,sample):
        if self.apply_to_classes is None:
            self.roll(sample)
        else:
            if sample['labels'] in self.apply_to_classes:
                self.roll(sample)
        return sample

    def roll(self,sample):
        for band in range(self.num_bands):
            seq_roll = torch.randint(0,self.max_roll,size = (1,))
            sample['data'][:,band] = torch.roll(sample['data'][:,band],shifts= (seq_roll,), dims = 0)
            sample['time'][:,band] = torch.roll(sample['time'][:,band],shifts= (seq_roll,), dims = 0)
            sample['mask'][:,band] = torch.roll(sample['mask'][:,band],shifts= (seq_roll,), dims = 0)


import torch
class ZScoreUndersample:
    """samplear elementos de la curva excluyendo aquellas observaciones que no estan contenidas en abs(zscore) > thr"""
    
    def __init__(self, thr =None, min_samples = 6, impose_seqlen:int = 200, inject_gauss_noise = False):
        """_summary_

        Args:
            thr (float, optional): El threshold de zscore sobre el cual se deberían seleccionar samples. Defaults to torch.rand(1).
            min_samples (int, optional): El minimo de punto que debe tener la curva. Si no cumple este minimo no bajosamplea, solo retorna el original. 
            De la misma manera si originalmente se tienen mas de 6 muestras y a través del bajosampleo se seleccionan menos de 6, se devuelve el array original.  Defaults to 6.
            impose_seqlen (int, optional): impone un largo de la secuencia de salida. Para cuando la secuencia de entrada tiene ++ puntos que lo que se quiere a la salida. Defaults to 200.
        """        
        self.impose_seqlen = impose_seqlen
        self.thr = thr
        self.min_samples = min_samples
        self.inject_gauss_noise = inject_gauss_noise
        
    def __call__(self,sample):
        data = sample['data'] 
        time = sample['time']
        if torch.count_nonzero(data) <= self.min_samples:
            return sample
        seqlen, channels = data.shape
        if channels > 1:
            new_data = torch.zeros((self.impose_seqlen,channels))
            new_time = torch.zeros(self.impose_seqlen,channels)
            for i in range(channels):
                band_data = data[:,i]
                band_time = time[:,i]
                if self.inject_gauss_noise:
                    band_data = band_data + (band_data!=0).bool()*torch.normal(0,abs(band_data.mean()), size=band_data.shape)
                x = torch.nonzero(band_data, as_tuple=True)
                nonz = band_data[x]
                zscore = (nonz - nonz.mean())/nonz.std()
                final_zscore = torch.zeros_like(band_data)
                final_zscore[x] = zscore
                if self.thr is None:
                    select_mask = (abs(final_zscore)>torch.rand(1))
                else:
                    select_mask = (abs(final_zscore)>self.thr)

                if torch.count_nonzero(select_mask) ==0:
                    return sample
                if torch.count_nonzero(select_mask) >= self.impose_seqlen:
                    rand_idx = torch.randperm(self.impose_seqlen)
                    rand_idx.sort()
                else:
                    rand_idx = None
                new_data[:torch.count_nonzero(select_mask),i] = torch.masked_select(band_data,select_mask)[rand_idx]
                new_time[:torch.count_nonzero(select_mask),i] = torch.masked_select(band_time,select_mask)[rand_idx]
        
        if torch.count_nonzero(new_data) == 0:
            return sample
        sample['data']  = new_data
        sample['time']  = new_time
        sample['mask'] = (new_data!=0).bool()
        return sample

class BandPermute:
    def __init__(self, num_bands, apply_to_classes = None):
        self.num_bands = num_bands
        self.apply_to_classes = apply_to_classes
    def __call__(self,sample):
        if self.apply_to_classes is None:
           self.permute(sample)
        else: 
            if sample['labels'] in self.apply_to_classes:
               self.permute(sample)
        return sample
    
    def permute(self, sample):
        shift_ = torch.randint(0,self.num_bands,size  =(1,))
        sample['data'] = torch.roll(sample['data'], shifts=(shift_,), dims=0)
        sample['time'] = torch.roll(sample['time'], shifts=(shift_,), dims=0)
        sample['mask'] = torch.roll(sample['mask'], shifts=(shift_,), dims=0)

class TimeNormalization:
    def __call__(self,sample):
        
        time = sample['time']
        mask_min = 9999999999.0 * (time == 0).float()
        # Compute minimum over non-zero time values by adding the mask
        t_min = torch.min(time.float() + mask_min)

        # Normalize and keep zeros in place
        sample['time'] = (time.float() - t_min) * (time != 0).float()
        return sample
    
class WindowSelect:
    def __init__(self,num_bands:int,window_size:int,apply_to_classes=None):
        self.num_bands = num_bands
        self.window_size = window_size
        self.apply_to_classes = apply_to_classes

    def __call__(self,sample:dict): 
        if self.apply_to_classes is None:
            self.window_select(sample)
        else:
            self.window_select(sample)
        return sample
    
    def window_select(self,  sample):
        for i in range(self.num_bands):
                nonzero_measures = torch.count_nonzero(sample['data'][:,i], dim = 0)
                intersection_check = nonzero_measures - self.window_size
                if  torch.count_nonzero((sample['data'][:,i])<= self.window_size):
                    continue
                else:
                    if intersection_check == 0:
                        start = 0
                    else:
                        start = torch.randint(0,intersection_check ,size = (1,)) 
                    end = start + self.window_size
                    sample["data"][:start,i] = 0
                    sample["data"][end:,i] = 0
                    sample["time"][:start,i] = 0
                    sample["time"][end:,i] = 0
                    sample['data'] = torch.roll(sample['data'], shifts=(-self.window_size,), dims = 1)
                    sample['time'] = torch.roll(sample['time'], shifts=(-self.window_size,), dims = 1)
                    sample['mask'] = sample['data'] != 0
    

class RandomSubsample:
    def __init__(self,num_bands:int,window_size:int):
        self.num_bands = num_bands
        self.window_size = window_size

    def __call__(self,sample:dict): 
        new_data = torch.zeros_like(sample['data'][:self.window_size,:])
        new_time = torch.zeros_like(sample['data'][:self.window_size,:])
        for band in range(self.num_bands):
            nonzero_measures = torch.count_nonzero(sample['data'][:,band], dim = 0)
            if  torch.count_nonzero((sample['data'][:,band])<= self.window_size):
                sample['data'] = sample['data'][:self.window_size, band]
                sample['time'] = sample['time'][:self.window_size, band]
                sample['mask'] = sample['mask'][:self.window_size, band]
            else:
                indices = torch.randint(0, nonzero_measures, size=(self.window_size,))
                indices.sort()
                new_data = sample['data'][indices, band]
                new_time = sample['time'][indices, band]
                sample['data'] = new_data
                sample['mask'] = (new_data !=0).bool()
                sample['time'] = new_time
        return sample

    
