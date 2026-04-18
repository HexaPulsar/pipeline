from typing import Literal, Union
import torch
from .submodules import *


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
            seq_roll = torch.randint(0, self.max_roll, size=(1,)).item()
            sample['data'][:, band] = torch.roll(sample['data'][:, band], shifts=seq_roll, dims=0)
            sample['time'][:, band] = torch.roll(sample['time'][:, band], shifts=seq_roll, dims=0)
            sample['mask'][:, band] = torch.roll(sample['mask'][:, band], shifts=seq_roll, dims=0)


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

    def __call__(self, sample):
        data = sample['data']
        time = sample['time']
        nonzero_count = torch.count_nonzero(data).item()
        if nonzero_count <= self.min_samples:
            return sample
        _, channels = data.shape
        if channels > 1:
            new_data = torch.zeros((self.impose_seqlen, channels))
            new_time = torch.zeros((self.impose_seqlen, channels))
            for i in range(channels):
                band_data = data[:, i]
                band_time = time[:, i]
                if self.inject_gauss_noise:
                    band_data = band_data + (band_data != 0).bool() * torch.normal(0, abs(band_data.mean()), size=band_data.shape)
                nonzero_mask = band_data != 0
                nonz = band_data[nonzero_mask]

                if len(nonz) == 0:
                    return sample

                zscore = (nonz - nonz.mean()) / (nonz.std() + 1e-8)
                final_zscore = torch.zeros_like(band_data)
                final_zscore[nonzero_mask] = zscore

                if self.thr is None:
                    threshold = torch.rand(1).item()
                else:
                    threshold = self.thr
                select_mask = torch.abs(final_zscore) > threshold

                num_selected = torch.count_nonzero(select_mask).item()
                if num_selected == 0:
                    return sample

                selected_data = torch.masked_select(band_data, select_mask)
                selected_time = torch.masked_select(band_time, select_mask)

                if num_selected >= self.impose_seqlen:
                    rand_idx = torch.randperm(num_selected)[:self.impose_seqlen]
                    rand_idx = rand_idx.sort()[0]
                else:
                    rand_idx = torch.arange(num_selected)

                new_data[:num_selected, i] = selected_data[rand_idx]
                new_time[:num_selected, i] = selected_time[rand_idx]
        else:
            return sample

        if torch.count_nonzero(new_data).item() == 0:
            return sample
        sample['data'] = new_data
        sample['time'] = new_time
        sample['mask'] = (new_data != 0).bool()
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
        shift_ = torch.randint(0, self.num_bands, size=(1,)).item()
        sample['data'] = torch.roll(sample['data'], shifts=shift_, dims=1)
        sample['time'] = torch.roll(sample['time'], shifts=shift_, dims=1)
        sample['mask'] = torch.roll(sample['mask'], shifts=shift_, dims=1)




class RandomMaskTimeVector:
    def __init__(self,num_bands:int,p:int,apply_to_classes=None):
        self.num_bands = num_bands
        self.p = p
        self.apply_to_classes = apply_to_classes

    def __call__(self,sample:dict):
        random_mask = torch.rand_like(sample['data']) >= self.p

        sample['time'] = sample['time']*random_mask
        return sample

class RandomMaskDataVector:
    def __init__(self,num_bands:int,p:int,apply_to_classes=None):
        self.num_bands = num_bands
        self.p = p
        self.apply_to_classes = apply_to_classes

    def __call__(self,sample:dict):
        random_mask = torch.rand_like(sample['data']) >= self.p
        sample['data'] = sample['data']*random_mask
        return sample


class RandomMaskMaskVector:
    def __init__(self,num_bands:int,p:int,apply_to_classes=None):
        self.num_bands = num_bands
        self.p = p
        self.apply_to_classes = apply_to_classes

    def __call__(self,sample:dict):
        random_mask = torch.rand_like(sample['mask']) >= self.p
        sample['mask'] = sample['mask']*random_mask
        #sample['time'] = sample['time']*random_mask
        #sample['mask'] = sample['data'] != 0
        return sample

class WindowSelect:
    def __init__(self,num_bands:int,window_size:int,apply_to_classes=None):
        self.num_bands = num_bands
        self.window_size = window_size
        self.apply_to_classes = apply_to_classes

    def __call__(self,sample:dict):
        self.window_select(sample)
        return sample

    def window_select(self, sample):
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
                sample['data'][:start, i] = 0
                sample['data'][end:, i] = 0
                sample['time'][:start, i] = 0
                sample['time'][end:, i] = 0
                sample['data'][:, i] = torch.roll(sample['data'][:, i], shifts=-start, dims=0)
                sample['time'][:, i] = torch.roll(sample['time'][:, i], shifts=-start, dims=0)
        sample['mask'] = (sample['data'] != 0).bool()

class BlockWindow:
    def __init__(self,num_bands:int,window_size:int,apply_to_classes=None):
        self.num_bands = num_bands
        self.window_size = window_size
        self.apply_to_classes = apply_to_classes

    def __call__(self,sample:dict):
        self.window_select(sample)
        return sample

    def window_select(self, sample):
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
                sample['data'][start:end, i] = 0
                sample['time'][start:end, i] = 0
        sample['mask'] = (sample['data'] != 0).bool()

class MAXWindowSelect:
    def __init__(self, num_bands: int, window_size: int, apply_to_classes=None):
        self.num_bands = num_bands
        self.window_size = window_size
        self.apply_to_classes = apply_to_classes

    def __call__(self, sample: dict):
        self.window_select(sample)
        return sample

    def window_select(self, sample):
        data = sample['data']
        time = sample['time']
        total_length = data.shape[0]

        for i in range(self.num_bands):
            signal = data[:, i]
            nonzero_count = torch.count_nonzero(signal).item()

            if nonzero_count < self.window_size:
                continue

            abs_signal = torch.abs(signal)
            max_idx = torch.argmax(abs_signal).item()

            half_window = self.window_size // 2
            start = max(max_idx - half_window, 0)
            end = min(start + self.window_size, total_length)
            if end - start < self.window_size:
                start = max(end - self.window_size, 0)

            # Zero out values outside the window
            data[:start, i] = 0
            data[end:, i] = 0
            time[:start, i] = 0
            time[end:, i] = 0

            if start > 0:
                sample['data'][:, i] = torch.roll(data[:, i], shifts=-start, dims=0)
                sample['time'][:, i] = torch.roll(time[:, i], shifts=-start, dims=0)
            else:
                sample['data'][:, i] = data[:, i]
                sample['time'][:, i] = time[:, i]

        sample['mask'] = sample['data'] != 0

class RandomSubsample:
    def __init__(self,num_bands:int,window_size:int):
        self.num_bands = num_bands
        self.window_size = window_size

    def __call__(self, sample: dict):
        new_data = torch.zeros_like(sample['data'][:self.window_size, :])
        new_time = torch.zeros_like(sample['data'][:self.window_size, :])
        for band in range(self.num_bands):
            nonzero_measures = torch.count_nonzero(sample['data'][:, band]).item()
            if nonzero_measures <= self.window_size:
                new_data[:, band] = sample['data'][:self.window_size, band]
                new_time[:, band] = sample['time'][:self.window_size, band]
            else:
                indices = torch.randint(0, nonzero_measures, size=(self.window_size,))
                indices = indices.sort()[0]
                new_data[:, band] = sample['data'][indices, band]
                new_time[:, band] = sample['time'][indices, band]
        sample['data'] = new_data
        sample['time'] = new_time
        sample['mask'] = (new_data != 0).bool()
        return sample


