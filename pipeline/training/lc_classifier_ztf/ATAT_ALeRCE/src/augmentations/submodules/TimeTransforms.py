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
            if isinstance(self.factor, list):
                return self.random_factor(sample)
            else:
                sample['time'] = sample['time'] * self.factor
        else:
            if sample['labels'] in self.apply_to_classes:
                if isinstance(self.factor, list):
                    return self.random_factor(sample)
                sample['time'] = sample['time'] * self.factor
        return sample

    def random_factor(self,sample):
        factor = np.random.choice(self.factor)
        for i in range(self.num_bands):
            band_time = sample['time'][:,i] * factor
            sample['time'][:, i] = band_time
        return sample

class TimeDelta:
    def __init__(self,num_bands = 2,delta_range =(0,1000), apply_to_classes: list = None):
        self.delta_range = delta_range
        self.num_bands = num_bands
        self.apply_to_classes = apply_to_classes
    def __call__(self, sample):
        if torch.count_nonzero(sample['time']).item() == 0:
            return sample
        if self.apply_to_classes is None:
            nonzero_times = sample['time'][sample['time'] > 0]
            if len(nonzero_times) == 0:
                return sample
            min_ = nonzero_times.min().item()
            if min_ >= np.inf:
                return sample
            delta = np.random.randint(0, int(min_))
            sample['time'] = sample['time'] - (sample['time'] > 0) * delta
        else:
            if sample['labels'] in self.apply_to_classes:
                nonzero_times = sample['time'][sample['time'] > 0]
                if len(nonzero_times) == 0:
                    return sample
                min_ = nonzero_times.min().item()
                if min_ >= np.inf:
                    return sample
                delta = np.random.randint(0, int(min_))
                sample['time'] = sample['time'] - (sample['time'] > 0) * delta
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
            nonzero = torch.count_nonzero(sample['time'][:, i]).item()
            filtered_signal = gaussian_filter1d(sample['time'][:nonzero, i].cpu().numpy(), choose_filter_std)
            sample['time'][:nonzero, i] = torch.from_numpy(filtered_signal).to(sample['time'].dtype).to(sample['time'].device)

class TimeNormalization:
    def __call__(self,sample):

        time = sample['time']
        mask_min = 9999999999.0 * (time == 0).float()
        # Compute minimum over non-zero time values by adding the mask
        t_min = torch.min(time.float() + mask_min)

        # Normalize and keep zeros in place
        sample['time'] = (time.float() - t_min) * (time != 0).float()
        return sample


class TimeGaussianNoise:
    def  __init__(self, num_bands, apply_to_classes: list = None):
        super().__init__()
        self.num_bands = num_bands
        self.apply_to_classes = apply_to_classes
    def __call__(self, sample):
        if self.apply_to_classes is None:
            for i in range(self.num_bands):
                band_time = sample['time'][:, i]
                if torch.count_nonzero(band_time).item() > 0:
                    band_mask = band_time != 0
                    nonzero_mean = torch.abs(band_time[band_mask].mean())
                    noise = torch.normal(0, nonzero_mean * 1e-3, size=band_time.shape, device=band_time.device, dtype=band_time.dtype)
                    sample['time'][:, i] = band_time + noise * band_mask
        else:
            if sample['labels'] in self.apply_to_classes:
                for i in range(self.num_bands):
                    band_time = sample['time'][:, i]
                    if torch.count_nonzero(band_time).item() > 0:
                        band_mask = band_time != 0
                        nonzero_mean = torch.abs(band_time[band_mask].mean())
                        noise = torch.normal(0, nonzero_mean * 1e-4, size=band_time.shape, device=band_time.device, dtype=band_time.dtype)
                        sample['time'][:, i] = band_time + noise * band_mask
        return sample

