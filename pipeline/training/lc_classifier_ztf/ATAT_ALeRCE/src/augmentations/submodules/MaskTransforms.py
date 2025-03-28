from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy
from .WindowApply import WindowApply

class OnlyMaskPadding:
    def __call__(self,sample:dict): 
        sample['mask'] = (sample['data'] != 0)
        return sample

class MaskFirstN:
    def __init__(self,mask_first = 8):
        self.mask_first = mask_first
    def __call__(self,sample):
        #
        if self.mask_first ==-1:
            return sample
        if isinstance(self.mask_first,list):
            mask_first = np.random.choice(self.mask_first)
            if mask_first ==-1:
                return sample
            if sample['mask'].sum(dim = [0,1]) < mask_first:
                return sample
            else:
                sample['mask'][:,:mask_first] = 0
                return sample 
        else:
            if sample['mask'].sum(dim = [0,1]) < self.mask_first:
                return sample
            else:
                sample['mask'][:,:self.mask_first] = 0
                return sample 


class MaskWindow:
    def __init__(self,num_bands, window = 2):
        self.window = 2
        self.num_bands = num_bands
    def __call__(self,sample):
        for band in range(self.num_bands):
            obs_count  = sample['mask'][:,band].sum()
            if all([obs_count <6 + self.window]):
                return sample
            seq_window = np.random.randint(0, obs_count, size=(2,))
            seq_window.sort()
            start, end = int(seq_window[0]), int(seq_window[1])  # Explicitly convert to Python integers
            sample['mask'][start:end,band] = 0
            return sample

class ThreeTimeMask:
    "Callable implementation of threetimemask function to integrate with the torchvision.transforms"

    def __init__(self,use_features,use_lightcurves,extracted_feat = None):
        self.use_features = use_features
        self.use_lightcurves = use_lightcurves
        self.extracted_feat = extracted_feat
        self.time_eval_list = [2,4,8,16,32,64,128,256,512,1024,2048, float('inf')]

    def __call__(self, sample):
        
        time_eval = np.random.choice(self.time_eval_list)  
        if self.use_lightcurves:
            mask, time = sample["mask"], sample["time"]
            mask_time = (time <= time_eval).bool()
            sample["mask"] = (mask * mask_time).bool()
        
        if self.use_features:
            sample["extracted_feat"] = self.extracted_feat[time_eval][sample['idx']]
        return sample
  
class MaskChannels: 
    def __init__(self, band_to_mask:int):
      
        self.band_to_mask = band_to_mask
    def __call__(self,sample):
        
        band_mask = deepcopy(sample['mask'])
        band_mask[:,self.band_to_mask] = 0 #mask the channel corresponding to band_to_mask
        if (sample['data'] * band_mask).sum() == 0: #if the lc sample only has values in one of the channels the band_mask could remove all values resulting in a zero input. 
            #if sample['data] * band_mask .sum() is zero, transformer doesn't see a thing. we would like to avoid this.
            return sample
        else:
            sample['mask'] = band_mask
            return sample
   
from scipy import ndimage, datasets 
class SobelFilterMask:
    def __init__(self,keep:Literal['below', 'above'] = 'above',threshold = 0.1):
        self.threshold = threshold
        self.keep = keep
    def __call__(self, sample):
        
        
        signal = sample['data']
        sobel_h = ndimage.sobel(signal, 0) # horizontal gradient
        sobel_h = torch.tensor(sobel_h) * (signal!=0)
        sobel_v = ndimage.sobel(signal, 1)    # vertical gradient
        sobel_v = torch.tensor(sobel_v)* (signal!=0)
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)  
         
        if magnitude.max() > 1:
            magnitude = (magnitude/magnitude.max()) * (signal !=0)
        
        if self.keep == 'below': 
            new_mask = (torch.tensor(magnitude)<= self.threshold).bool() & sample['mask'].clone()
        if self.keep == 'above':
            new_mask =  (torch.tensor(magnitude)>= self.threshold).bool() & sample['mask'].clone()
 
        if new_mask.sum().item() < 6:
            return sample
        else:
            sample['mask'] = new_mask
            return sample
    
     
class RangeSobelFilterMask:
    def __init__(self, threshold_range:tuple = (0.01,0.05)):
        self.threshold_range= threshold_range
    def __call__(self, sample):
        
        
        signal = sample['data']
        sobel_h = ndimage.sobel(signal, 0) # horizontal gradient
        sobel_h = torch.tensor(sobel_h) * (signal!=0)
        sobel_v = ndimage.sobel(signal, 1)    # vertical gradient
        sobel_v = torch.tensor(sobel_v)* (signal!=0)
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)   
        if magnitude.max() > 1:
            magnitude = (magnitude/magnitude.max())* (signal !=0) 
        sample['mask'] = torch.logical_and(
            magnitude >= self.threshold_range[0],
            magnitude <= self.threshold_range[1]
        )  
        return sample


class RandomSobelFilterMask:
    def __init__(self,
                 filter_type:Literal['horizontal', 'vertical', 'magnitude'], 
                 keep: Literal['above', 'below'],
                 threshold_range:tuple = (0.01,0.05)):
        self.threshold_range= threshold_range
        self.keep = keep
        self.filter = filter_type
        assert self.filter in ['horizontal','vertical','magnitude']
    
    def __call__(self, sample):
        
        
        signal = sample['data']
        sobel_h = ndimage.sobel(signal, 0) # horizontal gradient
        sobel_h = torch.tensor(sobel_h) * (signal!=0)
        if sobel_h.max() > 1:
            sobel_h = (sobel_h/sobel_h.max())* (signal !=0)
        
        sobel_v = ndimage.sobel(signal, 1)    # vertical gradient
        sobel_v = torch.tensor(sobel_v)* (signal!=0)
        if sobel_v.max() > 1:
            sobel_v = (sobel_v/sobel_v.max())* (signal !=0)
        
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)  
        magnitude = torch.tensor(magnitude)* (signal!=0)
        
        threshold = torch.FloatTensor(1).uniform_(self.threshold_range[0],self.threshold_range[1]).to(signal.device)
        
        if self.keep == 'above':
            if self.filter == 'horizontal':
                new_mask = (abs(sobel_h) >= threshold).bool()  & sample['mask'].clone()
            elif self.filter == 'vertical':
                new_mask = (abs(sobel_v) >= threshold).bool()  & sample['mask'].clone()
            elif self.filter == 'magnitude':
                new_mask = (magnitude >= threshold).bool()  & sample['mask'].clone()
        elif self.keep == 'below':
            if self.filter == 'horizontal':
                new_mask = (abs(sobel_h) <= threshold).bool()  & sample['mask'].clone()
            elif self.filter == 'vertical':
                new_mask = (abs(sobel_v) <= threshold).bool()  & sample['mask'].clone()
            elif self.filter == 'magnitude':
                new_mask = (magnitude <= threshold).bool()  & sample['mask'].clone()
        if (sample['mask'].sum()) <6: 
            return sample
        else:
            sample['mask'] = new_mask
            return sample
    
    
class RandomRangeSobelFilterMask:
    def __init__(self, threshold_range:tuple = (0.01,0.05)):
        self.threshold_range= threshold_range
    def __call__(self, sample):
        
        
        signal = sample['data']
        sobel_h = ndimage.sobel(signal, 0) # horizontal gradient
        sobel_h = torch.tensor(sobel_h) * (signal!=0)
        sobel_v = ndimage.sobel(signal, 1)    # vertical gradient
        sobel_v = torch.tensor(sobel_v)* (signal!=0)
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)   
        if magnitude.max() > 1:
            magnitude = (magnitude/magnitude.max())* (signal !=0)
        threshold = torch.FloatTensor(2).uniform_(self.threshold_range[0],self.threshold_range[1]).to(signal.device)
        sample['mask'] = torch.logical_and(
            magnitude >= self.threshold_range[0],
            magnitude <= threshold[1]
        )  
        return sample