from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy
from .submodules import *
import torch


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

    