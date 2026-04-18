from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy

class OnlyMaskPadding:
    def __call__(self,sample:dict): 
        sample['mask'] = (sample['data'] != 0)
        return sample



class RandomMask:
    def __init__(self,num_bands = 2, percentage_masked = 0.1):
        self.num_bands = num_bands
        self.percentage_masked = percentage_masked
    def __call__(self,sample):


        mask = (torch.rand(sample['data'].shape) <= self.percentage_masked).bool()
        sample['mask'] = torch.bitwise_and(mask, sample['mask'])
        return sample

class MaskFirstN:
    def __init__(self,num_bands = 2,mask_first = 8):
        self.num_bands = num_bands
        self.mask_first = mask_first
    def __call__(self,sample):
        if self.mask_first == -1:
            return sample
        if isinstance(self.mask_first, list):
            for i in range(self.num_bands):
                mask_first = np.random.choice(self.mask_first)
                if mask_first == -1:
                    return sample
                if torch.count_nonzero(sample['data'][:,i]) < mask_first:
                    continue
                else:
                    sample['mask'][:mask_first,i] = 0
            return sample
        else:
            if self.mask_first == -1:
                return sample
            for i in range(self.num_bands):
                if torch.count_nonzero(sample['data'][:,i]) < self.mask_first:
                    continue
                else:
                    sample['mask'][:self.mask_first,i] = 0
            return sample

class CutFromN:
    def __init__(self,num_bands = 2,cut_from_n = 8):
        self.num_bands = num_bands
        self.mask_first = cut_from_n
    def __call__(self,sample):
        if self.mask_first == -1:
            return sample
        if isinstance(self.mask_first, list):
            for i in range(self.num_bands):
                mask_first = np.random.choice(self.mask_first)
                if mask_first == -1:
                    return sample
                if torch.count_nonzero(sample['data']) < mask_first:
                    return sample
                else:
                    sample['data'][mask_first:,i] = 0
                    sample['time'][mask_first:,i] = 0
                    sample['mask'] = sample['data']!=0
                    return sample
        else:
            if self.mask_first == -1:
                return sample
            for i in range(self.num_bands):
                if torch.count_nonzero(sample['data']) < self.mask_first:
                    return sample
                else:
                    sample['data'][self.mask_first:,i] = 0
                    sample['time'][self.mask_first:,i] = 0
            sample['mask'] = sample['data'] != 0
            return sample 

class CutFirstN:
    def __init__(self,num_bands = 2,mask_first = 8):
        self.num_bands = num_bands
        self.mask_first = mask_first
    def __call__(self,sample):
        if self.mask_first == -1:
            return sample
        if isinstance(self.mask_first, list):
            for i in range(self.num_bands):
                mask_first = np.random.choice(self.mask_first)
                if mask_first == -1:
                    return sample
                if torch.count_nonzero(sample['data']) < mask_first:
                    return sample
                else:
                    sample['data'][:mask_first,i] = 0
                    sample['time'][:mask_first,i] = 0
                    sample['mask'] = sample['data']!=0
                    return sample
        else:
            if self.mask_first == -1:
                return sample
            for i in range(self.num_bands):
                if torch.count_nonzero(sample['data']) < self.mask_first:
                    return sample
                else:
                    sample['data'][:self.mask_first,i] = 0
                    sample['time'][:self.mask_first,i] = 0
            sample['mask'] = sample['data'] != 0
            return sample 

class MaskWindow:
    def __init__(self,num_bands:int,window_size:int):
        self.num_bands = num_bands
        self.window_size = window_size
    def __call__(self, sample: dict):
        for i in range(self.num_bands):
            nonzero_measures = torch.count_nonzero(sample['data'][:, i]).item()
            intersection_check = nonzero_measures - self.window_size
            if nonzero_measures <= self.window_size:
                continue
            else:
                if intersection_check == 0:
                    start = 0
                else:
                    start = torch.randint(0, intersection_check, size=(1,)).item()

                end = start + self.window_size
                new_mask = torch.zeros(sample['data'].shape[0], dtype=torch.bool, device=sample['data'].device)
                new_mask[start:end] = True
                sample['mask'][:, i] = new_mask
        return sample

class MaXMask:
    def __init__(self,num_bands:int,window_size:int, range= 5):
        self.num_bands = num_bands
        self.window_size = window_size
        self.range = range
    def __call__(self, sample: dict):
        for i in range(self.num_bands):
            get_max_idx = torch.argmax(sample['data'][:, i]).item()
            if get_max_idx < self.range:
                sample['mask'][0:get_max_idx + self.range, i] = True
            else:
                sample['mask'][get_max_idx - self.range:get_max_idx + self.range, i] = True
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
   
from scipy import ndimage, datasets 


class SobelFilterMask:
    def __init__(self, keep: Literal['below', 'above'] = 'above', threshold=0.1):
        self.threshold = threshold
        self.keep = keep

    def __call__(self, sample):
        signal = sample['data']
        signal_mask = signal != 0

        sobel_h = torch.from_numpy(ndimage.sobel(signal.cpu().numpy(), 0)).to(signal.device, signal.dtype) * signal_mask
        sobel_v = torch.from_numpy(ndimage.sobel(signal.cpu().numpy(), 1)).to(signal.device, signal.dtype) * signal_mask
        magnitude = torch.sqrt(sobel_h ** 2 + sobel_v ** 2)

        mag_max = magnitude.max()
        if mag_max > 1:
            magnitude = (magnitude / mag_max) * signal_mask

        if self.keep == 'below':
            new_mask = (magnitude <= self.threshold) & sample['mask']
        else:  # 'above'
            new_mask = (magnitude >= self.threshold) & sample['mask']

        if new_mask.sum().item() < 6:
            return sample
        sample['mask'] = new_mask
        return sample
     
class RangeSobelFilterMask:
    def __init__(self, threshold_range: tuple = (0.01, 0.05)):
        self.threshold_range = threshold_range

    def __call__(self, sample):
        signal = sample['data']
        signal_mask = signal != 0

        sobel_h = torch.from_numpy(ndimage.sobel(signal.cpu().numpy(), 0)).to(signal.device, signal.dtype) * signal_mask
        sobel_v = torch.from_numpy(ndimage.sobel(signal.cpu().numpy(), 1)).to(signal.device, signal.dtype) * signal_mask
        magnitude = torch.sqrt(sobel_h ** 2 + sobel_v ** 2)

        mag_max = magnitude.max()
        if mag_max > 1:
            magnitude = (magnitude / mag_max) * signal_mask

        sample['mask'] = torch.logical_and(
            magnitude >= self.threshold_range[0],
            magnitude <= self.threshold_range[1]
        )
        return sample

class RandomSobelFilterMask:
    def __init__(self,
                 filter_type: Literal['horizontal', 'vertical', 'magnitude'],
                 keep: Literal['above', 'below'],
                 threshold_range: tuple = (0.01, 0.05)):
        self.threshold_range = threshold_range
        self.keep = keep
        self.filter = filter_type
        assert self.filter in ['horizontal', 'vertical', 'magnitude']

    def __call__(self, sample):
        signal = sample['data']
        signal_mask = signal != 0

        sobel_h = torch.from_numpy(ndimage.sobel(signal.cpu().numpy(), 0)).to(signal.device, signal.dtype)
        sobel_h = sobel_h * signal_mask
        if sobel_h.max() > 1:
            sobel_h = (sobel_h / sobel_h.max()) * signal_mask

        sobel_v = torch.from_numpy(ndimage.sobel(signal.cpu().numpy(), 1)).to(signal.device, signal.dtype)
        sobel_v = sobel_v * signal_mask
        if sobel_v.max() > 1:
            sobel_v = (sobel_v / sobel_v.max()) * signal_mask

        magnitude = torch.sqrt(sobel_h ** 2 + sobel_v ** 2) * signal_mask

        threshold = torch.rand(1, device=signal.device, dtype=signal.dtype) * (self.threshold_range[1] - self.threshold_range[0]) + self.threshold_range[0]
        threshold = threshold.item()

        if self.keep == 'above':
            if self.filter == 'horizontal':
                new_mask = (torch.abs(sobel_h) >= threshold) & sample['mask']
            elif self.filter == 'vertical':
                new_mask = (torch.abs(sobel_v) >= threshold) & sample['mask']
            else:  # magnitude
                new_mask = (magnitude >= threshold) & sample['mask']
        else:  # below
            if self.filter == 'horizontal':
                new_mask = (torch.abs(sobel_h) <= threshold) & sample['mask']
            elif self.filter == 'vertical':
                new_mask = (torch.abs(sobel_v) <= threshold) & sample['mask']
            else:  # magnitude
                new_mask = (magnitude <= threshold) & sample['mask']

        if new_mask.sum().item() < 6:
            return sample
        sample['mask'] = new_mask
        return sample
    
class RandomRangeSobelFilterMask:
    def __init__(self, threshold_range: tuple = (0.01, 0.05)):
        self.threshold_range = threshold_range

    def __call__(self, sample):
        signal = sample['data']
        signal_mask = signal != 0

        sobel_h = torch.from_numpy(ndimage.sobel(signal.cpu().numpy(), 0)).to(signal.device, signal.dtype) * signal_mask
        sobel_v = torch.from_numpy(ndimage.sobel(signal.cpu().numpy(), 1)).to(signal.device, signal.dtype) * signal_mask
        magnitude = torch.sqrt(sobel_h ** 2 + sobel_v ** 2)

        mag_max = magnitude.max()
        if mag_max > 1:
            magnitude = (magnitude / mag_max) * signal_mask

        threshold_max = torch.rand(1, device=signal.device, dtype=signal.dtype) * (self.threshold_range[1] - self.threshold_range[0]) + self.threshold_range[0]
        threshold_max = threshold_max.item()

        sample['mask'] = torch.logical_and(
            magnitude >= self.threshold_range[0],
            magnitude <= threshold_max
        )
        return sample