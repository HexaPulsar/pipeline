import scipy.signal as signal 
import numpy as np
import torch 
import torch.nn.functional as F
from copy import deepcopy


class ChessCurve:
    def __call__(self, data_dict):
        """
        Creates a chess mask and returns one of two possible variations of the lightcurve. (chess or ~chess)

        Parameters:
        - data_dict: Dictionary with 'data' and 'time' keys containing tensors.

        Returns:
        - Dictionary with 'data' and 'time' keys, where the data and time is masked by the chess pattern or the inverse chess pattern.
        """
        data_dict = data_dict #very very very important line do not remov
        data = data_dict['data']
        time = data_dict['time']

        # Create the chessboard mask for a single lightcurve
        mask = torch.zeros((99, 6), dtype=torch.bool)
        mask[::2, 1::2] = 1
        mask[1::2, ::2] = 1

        # Expand the mask along the batch dimension
        mask = mask.unsqueeze(0)  # Shape (1, 99, 6)
        mask = mask.expand(data.size(0), -1, -1).to(device=data.device)  # Shape (batch_size, 99, 6)

        # Apply the mask and its inverse to the lightcurve and time
        lightcurve_masked = data * mask
        lightcurve_inverse_masked = data * (~mask)

        time_masked = time * mask
        time_inverse_masked = time * (~mask)

        # Randomly choose between the two masked outputs
        selected_data, selected_time = (lightcurve_masked, time_masked) if np.random.rand() > 0.5 \
            else (lightcurve_inverse_masked, time_inverse_masked)
        
        data_dict['data'] = selected_data
        data_dict['time'] = selected_time
        data_dict['mask'] = (selected_time>1).int()
        # Return the modified dictionary
        return data_dict
    
class ChessMask:
    def __call__(self, data_dict):
        """
        Creates a chess mask and returns one of two possible variations of the lightcurve. (chess or ~chess)

        Parameters:
        - data_dict: Dictionary with 'data' and 'time' keys containing tensors.

        Returns:
        - Dictionary with 'data' and 'time' keys, where the data and time is masked by the chess pattern or the inverse chess pattern.
        """ 
        data = data_dict['data']
        time = data_dict['time']

        # Create the chessboard mask for a single lightcurve
        mask = torch.zeros((99, 6), dtype=torch.bool)
        mask[::2, 1::2] = 1
        mask[1::2, ::2] = 1

        # Expand the mask along the batch dimension
        mask = mask.unsqueeze(0)  # Shape (1, 99, 6)
        mask = mask.expand(data.size(0), -1, -1).to(device=data.device)  # Shape (batch_size, 99, 6)

        # Apply the mask and its inverse to the lightcurve and time
        lightcurve_masked = data * mask
        lightcurve_inverse_masked = data * (~mask)

        time_masked = time * mask
        time_inverse_masked = time * (~mask)

        # Randomly choose between the two masked outputs
        selected_data, selected_time = (lightcurve_masked, time_masked) if np.random.rand() > 0.5 \
            else (lightcurve_inverse_masked, time_inverse_masked)
        
        #data_dict['data'] = selected_data
        #data_dict['time'] = selected_time
        data_dict['mask'] = (selected_time>1).int()
        # Return the modified dictionary
        return data_dict

class InverseCurve:
    def __call__(self, sample):
        data = sample["data"]
        data[:, :] *= -1
        sample["data"] = data
        return sample

class FlipLC:
    def __call__(self, sample):
        data = sample["data"]
        time = sample["time"]
        mask = sample["mask"]

        data = torch.flip(data, [0])
        time = torch.flip(time, [0])
        mask = torch.flip(mask, [0])

        sample["data"] = data
        sample["time"] = time
        sample["mask"] = mask

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

class Factor:
    def __init__(self, min_scale=0.99, max_scale=1.01):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        data = sample['data']
        #time = sample['time']
        factor = torch.FloatTensor(1).uniform_(0.95, 1.05)
        # Scale each point of sample['data'] with its corresponding scale factor
        sample['data'] = data * factor
        #sample['time'] = time * scale_factors

        return sample

class Shift:
    def __init__(self, min_scale=0.99, max_scale=1.01):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        data = sample['data']
        #time = sample['time']
        factor = torch.FloatTensor(1).uniform_(-0.5, 0.5)
        # Scale each point of sample['data'] with its corresponding scale factor
        sample['data'] = data + factor*data
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


class GaussianNoise:
    def __call__(self, sample):
        x = sample["data"]  # Shape: [bs, seqlen, channels]
        mask = x!=0
          # torch.rand(1).to(device = x.device).item()
        #std = torch.randint(0,5, size = (1,)).to(device=x.device,non_blocking=True).item()
        
        # Generate Gaussian noise for each channel independently
        noise = torch.normal(0, 1, size=x.shape).to(device=x.device, non_blocking=True) 

        sample["data"] = x + noise * x.mean() * mask
        return sample

class RandomMask:
    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Input tensor of shape [bs, seqlen, channels].

        Returns:
            torch.Tensor: Tensor with a random channel zeroed for each sample in the batch.
        """
        mask = torch.bitwise_and(( torch.rand_like(sample['data'])>=0.5).bool() ,sample['mask']  )  
        
        sample['mask'] = mask
        return sample

import random
class CutLC:
    def __init__(self):
        super().__init__()
        self.eval_times = [10,20,30,40,60,80,100]

    def __call__(self, sample: dict):
        # Select a random eval_time
        eval_time = random.choice(self.eval_times)
        
        # Get the batch size, sequence length, and channels
        seqlen, channels = sample['data'].shape
        
        # Create a mask to zero out elements after eval_time
        mask = torch.arange(seqlen).expand( seqlen).unsqueeze(-1).to(sample['data'].device)
        cutoff_mask = mask < eval_time
        
        # Zero out elements in 'time', 'data', and 'mask' after the eval_time
        sample['time'] = sample['time'] * cutoff_mask
        sample['data'] = sample['data'] * cutoff_mask
        sample['mask'] = sample['mask'] * cutoff_mask
        
        return sample

class OnlyMaskPadding:
    def __call__(self,sample:dict): 
        sample['mask'] = (sample['data'] != 0)
        return sample
    
class SobelFilterTransform:
    def __init__(self,thr=0.01):
        self.thr = thr
    def __call__(self, sample):
        input_tensor = sample['data']
        seqlen, channels = input_tensor.shape
        sobel_filter = torch.tensor([-1, 0, 1], dtype=input_tensor.dtype, device=input_tensor.device).view(1, 1, 3)
        output = torch.zeros_like(input_tensor)

        for c in range(channels):
            channel_data = input_tensor[:, :, c]
            filtered_channel = F.conv1d(channel_data.unsqueeze(1), sobel_filter, padding=1)
            filtered_channel = filtered_channel.squeeze(1)

            norm = filtered_channel.norm(p=2, dim=1, keepdim=True)
            filtered_channel_normalized = filtered_channel / (norm + 1e-8)
            output[:, :, c] = filtered_channel_normalized
        sample['data'] = (output > self.thr).float() * input_tensor
                    
        return sample
    

class MaskChannels: 
    def __init__(self, channel_list):
        self.channel_list = channel_list
    def __call__(self,sample):
        """
        Args:
            sample (torch.Tensor): Input tensor of shape [bs, seqlen, channels].

        Returns:
            torch.Tensor: Tensor with a random channel zeroed for each sample in the batch.
        """
        mask = sample['mask']  # Assuming 'mask' is the key for the input tensor
         
        for i in self.channel_list:
             
            mask[:, :, i] = 0
 
        sample['mask'] = mask.bool
        return sample
    
class RandomMaskChannelZero: 

    def __call__(self,sample):
        """
        Args:
            sample (torch.Tensor): Input tensor of shape [bs, seqlen, channels].

        Returns:
            torch.Tensor: Tensor with a random channel zeroed for each sample in the batch.
        """
        mask = sample['mask']  # Assuming 'mask' is the key for the input tensor
        bs, seqlen, channels = mask.shape

        for i in range(bs):
            random_channel = torch.randint(0, channels, (1,)).item()
            mask[i, :, random_channel] = 0
 
        sample['mask'] = mask.float()
        return sample

class SobelFilterMask:
    def __init__(self,combine_mask_with:str = None,threshold = 0.001):
        self.threshold = threshold
        self.cmw = combine_mask_with
    def __call__(self, sample):
        input_tensor = sample['data']
        #time = sample['time']
        og_mask = sample['mask']
        seqlen, channels = input_tensor.shape
        sobel_filter = torch.tensor([-1, 0, 1], dtype=input_tensor.dtype, device=input_tensor.device).view(1, 1, 3)
        output = torch.zeros_like(input_tensor)

        for c in range(channels):
            channel_data = input_tensor[:, :, c]
            filtered_channel = F.conv1d(channel_data.unsqueeze(1), sobel_filter, padding=1)
            filtered_channel = filtered_channel.squeeze(1)

            norm = filtered_channel.norm(p=2, dim=1, keepdim=True)
            filtered_channel_normalized = filtered_channel / (norm + 1e-8)
            output[:, :, c] = filtered_channel_normalized

        thr = output  > self.threshold 
         
        #sample['mask'] =  (input_tensor != 0) & ~thr
        if self.cmw  == "mask":
            sample['mask'] =  og_mask.bool() & (thr!=0)
        elif self.cmw == "datamask":
            sample['mask'] = thr # (input_tensor != 0) & 

             
        else:
            sample['mask'] =  og_mask.bool() & (thr!=0) if torch.rand(1) >0.5 else (input_tensor != 0) & thr
            
        return sample

class Scale:
    def __init__(self, min_scale=0.9, max_scale=1.2):
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

class RandomPointDrop:
    def __call__(self, sample):
        x = sample['data']  # Shape: [bs, seqlen, channels]
        t = sample['time']  # Shape: [bs, seqlen, channels]

        # Iterate over the batch
        for i in range(x.shape[0]):  # Loop over each sample in the batch
            for c in range(x.shape[2]):  # Loop over each channel
                # Randomly select one point in the sequence to drop (set to zero)
                drop_idx = random.randint(0, x.shape[1] - 1)
                x[i, drop_idx, c] = 0  # Set the selected point to zero
                t[i, drop_idx, c] = 0  # Set the selected point to zero


        sample['data'] = x
        sample['time'] = t
        sample['mask'] = (t>1).float()

        return sample

 
class TimeWarp: 
    def __init__(self, min_scale=0.5, max_scale=1.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
    def __call__(self, sample):
        x = sample['time']
        
        factor =torch.FloatTensor(1).uniform_(self.min_scale,self.max_scale).to(x.device)
        #factor = (r1 - r2) * torch.rand(1) + r2

        #print(factor)
        sample['time'] = x *factor
        return sample
    


class TimeShift: 
    def __init__(self, min_scale=0.1, max_scale=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
    def __call__(self, sample):
        x = sample['time']
        
        factor =torch.FloatTensor(1).uniform_(self.min_scale,self.max_scale).to(x.device)
        mask = sample['time'] * factor
        sample['time'] = sample['time'] + mask
        return sample
    