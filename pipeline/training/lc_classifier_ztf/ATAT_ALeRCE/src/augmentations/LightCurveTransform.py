from typing import Literal, Union
import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy
 
 
class MaskFirstN:
    def __init__(self,mask_first = 8):
        self.mask_first = mask_first
    def __call__(self,sample):
        sample = deepcopy(sample)
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

class InverseCurve:
    def __call__(self, sample):
        data = sample["data"]
        data[:, :] *= -1
        sample["data"] = data
        return sample

class InverseTime:
    def __call__(self, sample):
        data = sample["data"]
        time = sample["time"]
        mask = sample["mask"]

        data = torch.flip(data, [0])
        time = torch.flip(time, [0])
        mask = torch.flip(mask, [0])

        data = sample["time"]
        data[:, :] *= -1
        sample["time"] = data
        return sample

class FlipLC:
    def __call__(self, sample):
        sample = deepcopy(sample)
        data = sample["data"]
        
        # Get indices of nonzero elements
        nonzero_indices = torch.nonzero(data)
        
        # Get the nonzero values
        nonzero_values = data[nonzero_indices]
        
        # Create a mask of zeros with the same shape as the input
        result = torch.zeros_like(data)
        
        # Flip only the nonzero values
        flipped_values = torch.flip(nonzero_values, [0])
        
        # Place the flipped values back into their corresponding positions
        for i, idx in enumerate(nonzero_indices):
            result[idx] = flipped_values[i]
            
        sample["data"] = result
        return sample

class PermuteChannels:
    def __call__(self, sample):
        seqlength, channels = sample["data"].shape

        permuted_channels = torch.randperm(
            channels
        )  # Generate a random permutation of channels

        sample["data"] = sample["data"][:, permuted_channels]
        sample["time"] = sample["time"][:, permuted_channels]
        sample["mask"] = sample["mask"][:, permuted_channels]

        return sample
    
class Undersample:
    def __init__(self, num_bands, sample_fraction, ignore_samples_with_less_points_than = 6):
        self.num_bands = num_bands
        self.sample_fraction = sample_fraction
        self.ignore = ignore_samples_with_less_points_than = ignore_samples_with_less_points_than
        assert all([self.sample_fraction < 1, sample_fraction > 0])
    def __call__(self, sample):
        sample = deepcopy(sample)
        
        for channel in range(self.num_bands):
            valid_indices = torch.nonzero(sample['data'][:,channel])
            #print(valid_indices)
            #print(valid_indices.shape)
            if len(valid_indices) < self.ignore: # Skip if too few points
                continue
            
            # Ensure first point is included
            if 0 not in valid_indices:
                valid_indices[0] = 0
            
            # Randomly select subset of indices while keeping first point
            num_points = valid_indices.size(0)
            num_to_keep = max(int(num_points * self.sample_fraction), self.ignore)  # Keep 70% or minimum 12 points
            
            # Always keep the first point and randomly select the rest
            keep_indices = torch.cat([
                valid_indices[0:1],
                valid_indices[1:][torch.randperm(num_points-1)[:num_to_keep-1]]
            ])
            keep_indices = torch.sort(keep_indices)[0]  # Sort indices to maintain temporal order
            
            # Create new zeros vector and copy selected data points
            new_data = torch.zeros_like(sample['data'][:, channel])
            new_time = torch.zeros_like(sample['time'][:, channel])
            new_mask = torch.zeros_like(sample['mask'][:, channel])
            new_data[keep_indices] = sample['data'][keep_indices, channel]
            new_time[keep_indices] = sample['time'][keep_indices, channel]
            new_mask[keep_indices] = sample['mask'][keep_indices, channel]
            
            # Replace original data with undersampled version
            sample['data'][:, channel] = new_data
            sample['time'][:, channel] = new_time
            sample['mask'][:, channel] = new_mask

        return sample


class TimeShift:
    def __init__(self, min_scale=0, max_scale=2*torch.pi):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        data = (sample['time']!=0)
        #time = sample['time']
        factor = torch.FloatTensor(1).uniform_(self.min_scale, self.max_scale)
        # Scale each point of sample['data'] with its corresponding scale factor
        sample['time'] = data + factor*data
        #sample['time'] = time * scale_factors

        return sample
    
class RandomRoll:
    def __call__(self, sample):
        roll_1 = torch.randint(low = 0,high = 99,size=(1,))
        roll_2 = torch.randint(low = 0,high =6,size=(1,))
        sample['data'] = torch.roll(sample['data'], shifts=[roll_1,roll_2],dims=[0,1])
        sample['mask'] = torch.roll(sample['mask'], shifts=[roll_1,roll_2],dims=[0,1])
        sample['time'] = torch.roll(sample['time'], shifts=[roll_1,roll_2],dims=[0,1])
        return sample

class SequenceShift:
    def __init__(self, shift_range: tuple):
        self.shift_range = shift_range

    def __call__(self, sample):
        """
        Shift the sequence in the 'data' key of the input dictionary

        Args:
            sample_dict (dict): Dictionary containing 'data' key with tensor
                               of shape [batch_size, seq_len, channels]

        Returns:
            dict: Dictionary with shifted data
        """

        data = sample["data"]
        time = sample["time"]
        shift_amount = random.randint(self.shift_range[0], self.shift_range[1])
        # Create output tensor of same shape
        result_data = torch.zeros_like(data)
        result_time = torch.zeros_like(data)
        #print(result_data.shape)
        if shift_amount > 0:
            # Shift forward (left)
            result_data[: -shift_amount,:] = data[ shift_amount :,:]
            result_data[ -shift_amount :,:] = 0

            result_time[: -shift_amount,:] = time[shift_amount :,:]
            result_time[ -shift_amount :,:] = 0

        elif shift_amount < 0:
            # Shift backward (right)
            result_data[ -shift_amount :,:] = data[: shift_amount,:]
            result_data[: -shift_amount,:] = 0

            result_time[ -shift_amount :,:] = time[: shift_amount,:]
            result_time[: -shift_amount,:] = 0

        else:
            # No shift
            result_data = data
            result_time = time
        # Update the dictionary with shifted data
        sample["data"] = result_data
        sample["time"] = result_time
        sample["mask"] = (result_data != 0)
        return sample



class Jitter:
    def __call__(self, sample):
        x = sample["data"]  # Shape: [bs, seqlen, channels]

        # Create a mask for non-zero values
        mask = x != 0  # Mask with the same shape as x
        max_jitter = torch.rand(1).to(device=x.device,non_blocking=True).item()

        # Generate random jitter for each channel independently within the range [-max_jitter, max_jitter]
        jitter = (
            torch.rand_like(x, device=x.device) * 2 - 1
        ) * max_jitter  # Shape: [bs, seqlen, channels]

        # Apply jitter only to the non-zero values
        x_with_jitter = x + jitter * mask

        sample["data"] = x_with_jitter
        return sample

class Exptime:
    def __call__(self, sample):
        sample = deepcopy(sample)

        x = sample["data"]  # Shape: [bs, seqlen, channels]
        mask = x!=0
          # torch.rand(1).to(device = x.device).item()
        #std = torch.randint(0,5, size = (1,)).to(device=x.device,non_blocking=True).item()
        
        # Generate Gaussian noise for each channel independently
        noise = torch.normal(0, 1, size=x.shape).to(device=x.device, non_blocking=True) 

        x = sample["time"]  # Shape: [bs, seqlen, channels]
        sample["time"] = torch.exp(x/1000)
        return sample


class GaussianNoise:
    def  __init__(self, num_bands, mean,std):
        self.num_bands = num_bands
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        sample = deepcopy(sample)
        for i in range(self.num_bands):
            noise = torch.normal(self.mean,self.std, size=(sample['data'].shape[0],)).to(device=sample['data'].device, non_blocking=True) 
            sample["data"][:,i] = sample['data'][:,i] + noise * (sample['data'][:,i]!=0)
        return sample


class TimeGaussianNoise:
    def  __init__(self, num_bands, mean,std):
        self.num_bands = num_bands
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        sample = deepcopy(sample)
        for i in range(self.num_bands):
            noise = torch.normal(self.mean,self.std, size=(sample['time'].shape[0],)).to(device=sample['time'].device, non_blocking=True) 
            sample["time"][:,i] = sample['time'][:,i] + noise * (sample['time'][:,i]!=0)
        return sample

class OnlyMaskPadding:
    def __call__(self,sample:dict): 
        sample['mask'] = (sample['data'] != 0)
        return sample
    
class SobelFilterTransform:
    def __init__(self,thr=0.01):
        self.thr = thr
    def __call__(self, sample):
        sample = deepcopy(sample)
        time_eval = np.random.choice(self.time_eval_list)  
        if self.use_lightcurves:
            mask, time = sample["mask"], sample["time"]
            mask_time = (time <= time_eval).bool()
            sample["mask"] = (mask * mask_time).bool()
        
        if self.use_features:
            sample["extracted_feat"] = self.extracted_feat[time_eval][sample['idx']]
        return sample
     

class SelectEven:
    def __init__(self,num_bands):
        self.num_bands = num_bands
    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Input tensor of shape [bs, seqlen, channels].

        Returns:
            torch.Tensor: Tensor with a random channel zeroed for each sample in the batch.
        """
        indices = (2*torch.range(0,99, dtype = int)) + 1
        sample['data'] = sample['data'][indices, :]
        sample['time'] = sample['time'][indices, :]
        sample['mask'] = sample['mask'][indices, :]
        return sample

 
class PostMaxMask:
    def __init__(self,num_bands):
        self.num_bands = num_bands
    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Input tensor of shape [bs, seqlen, channels].

        Returns:
            torch.Tensor: Tensor with a random channel zeroed for each sample in the batch.
        """
        values, indices = sample['data'].max(dim = 1)
        for i in range(self.num_bands):
            sample['mask'][indices[i]:,i] = 0
            return sample
  
class OnlyMaskPadding:
    def __call__(self,sample:dict): 
        sample = deepcopy(sample)

        sample['mask'] = (sample['data'] != 0)
        return sample
    
class WindowMask:
    def __init__(self,num_bands, window_size):
        self.num_bands = num_bands
        self.window_size = window_size
    def __call__(self,sample:dict): 
        sample = deepcopy(sample)
        new_mask = torch.zeros_like(sample['mask'])
        for i in range(self.num_bands):
            if (sample['data'][:,i]!=0).sum() == 0:
                continue
            ints = torch.randint(0,200- self.window_size, (1,))
            #if ordered_ints.sum() == 0:
             #   return sample
            new_mask[ints.item():ints.item()+self.window_size,i] = 1
            new_mask = new_mask  & sample['mask']
        if new_mask.sum() < 6:
            return sample
        sample['mask'] = new_mask
        return sample
    

class MaskChannels: 
    def __init__(self, num_bands):
      
        self.num_bands = num_bands
    def __call__(self,sample):
         
        band = torch.randint(low = 0, high = self.num_bands, size= (1,))
            
        sample['mask'][:, band] = 0
        
        return sample
   
from scipy import ndimage, datasets 
class SobelFilterMask:
    def __init__(self,keep:Literal['below', 'above'] = 'above',threshold = 0.1):
        self.threshold = threshold
        self.keep = keep
    def __call__(self, sample):
        sample = deepcopy(sample)
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
        sample = deepcopy(sample)
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
        sample = deepcopy(sample)
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
        sample = deepcopy(sample)
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
    
from scipy.ndimage import gaussian_filter1d

class GaussianFilter:
    def __init__(self, num_bands,filter_std):
        self.num_bands =num_bands
        self.filter_std = filter_std
    def __call__(self,sample):
        sample = deepcopy(sample)
        for i in range(self.num_bands):
            filtered_signal = gaussian_filter1d(sample['data'][:,i], self.filter_std)
            filtered_signal = torch.tensor(filtered_signal)* (sample['data'] != 0)[:,i]
            sample['data'][:,i] = filtered_signal
        return sample
class Scale:
    def __init__(self, min_scale=0.998, max_scale=1.002):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        data = sample['data']
        #time = sample['time']
         
        # Generate a random scale factor for each point in sample['data']
        scale_factors = torch.empty(sample['data'].shape,device = data.device).uniform_(self.min_scale, self.max_scale)

        # Scale each point of sample['data'] with its corresponding scale factor
        sample['data'] = data * scale_factors
        #sample['time'] = time * scale_factors

        return sample

class TimeScale:
    def __init__(self, min_scale=0.998, max_scale=1.002):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        sample = deepcopy(sample)
        time = sample['time']
        #time = sample['time']
         
        # Generate a random scale factor for each point in sample['time']
        scale_factors = torch.empty(sample['time'].shape,device = time.device).uniform_(self.min_scale, self.max_scale)

        # Scale each point of sample['time'] with its corresponding scale factor
        sample['time'] = time * scale_factors
        #sample['time'] = time * scale_factors

        return sample
    
class Factor:
    def __init__(self, factor = 0.5):
        self.factor = factor
          
    def __call__(self, sample):
        if isinstance(self.factor,list):
            return self.random_factor(sample)
        sample = deepcopy(sample)
        data = sample['data']
        sample['data'] = (data * self.factor)
        return sample
    
    def random_factor(self,sample):
        sample = deepcopy(sample)
        data = sample['data']
        choose = np.random.choice(self.factor)
        sample['data'] =  (data * choose)
        return sample
class TimeFactor:
    def __init__(self, factor =0.5):
        self.factor = factor
        
    def __call__(self, sample):
        if isinstance(self.factor,list):
            return self.random_factor(sample)
        sample = deepcopy(sample)
        data = sample['time']
        sample['time'] = (data * self.factor)
        return sample
    
    def random_factor(self,sample):
        sample = deepcopy(sample)
        data = sample['time']
        choose = np.random.choice(self.factor)
        sample['time'] =  (data * choose)
        return sample
  
    
class TimeWarp: 

    def __init__(self, min_scale=0.8, max_scale=1.2):

        self.min_scale = min_scale
        self.max_scale = max_scale
    def __call__(self, sample):
        x = sample['time']
        
        factor =torch.FloatTensor(1).uniform_(self.min_scale,self.max_scale).to(x.device)
        #factor = (r1 - r2) * torch.rand(1) + r2

        #print(factor)
        sample['time'] = x *factor
        return sample
     
class ChannelTimeShift: 
    def __init__(self, min_scale = 0, max_scale=5):

        self.min_scale = min_scale
        self.max_scale = max_scale
    def __call__(self, sample):
        x = sample['time']
        
        for i in range(x.size(1)):
            min_scale = sample['time'][:,i].max()
            factor =torch.FloatTensor(1).uniform_(-min_scale,self.max_scale).to(x.device)
            mask = (sample['time'][:,i]>0) * factor
            sample['time'][:,i] = sample['time'][:,i] + mask
        return sample
    