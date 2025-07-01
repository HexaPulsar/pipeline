
import logging
import torch
from src.data.handlers.BaseDataset import BaseDataset
from joblib import load
from dataclasses import dataclass
import logging
from .BaseDataset import BaseDataset
from typing import Literal, Union, Optional
from torchvision.transforms import Compose
from copy import deepcopy
from src.augmentations import LightCurveTransform as LC

class Cut200:
    def __init__(self,num_bands:int,seqlen:int, sampling_type:str = 'overlap_100'):
        self.num_bands = num_bands
        self.seqlen = seqlen
        self.sampling_type = sampling_type
    def __call__(self,sample:dict): 
        sample = deepcopy(sample)
        if ((sample['data']!=0).sum() < 400) or self.sampling_type =='static':
            sample['data']=sample['data'][:200,:]
            sample['mask']= sample['mask'][:200,:]
            sample['time']= sample['time'][:200,:]
        elif self.sampling_type == 'random':
            random_indices = torch.randint(0,200, size = (200,))
            random_indices.sort()
            sample['data']=sample['data'][random_indices,:]
            sample['mask']= sample['mask'][random_indices,:]
            sample['time']= sample['time'][random_indices,:]
        elif self.sampling_type == 'random_window':
            max_seqlen = (sample['data']!=0).sum(dim  = 0).max().item()
            higher = max_seqlen-200
            if higher <= 0:
                return sample
            start = torch.randint(0,higher-200,size = (1,)).item()
            sample['data']=sample['data'][start:start+200,:]
            sample['mask']= sample['data']!=0
            sample['time']= sample['time'][start:start+200,:]
        elif self.sampling_type == 'undersample':
            n = 30
            indices = torch.randint(0,1500-n,size = (n,)).item()
            data = torch.zeros(size = (200,2))
            time = torch.zeros(size = (200,2))
            data[0:n, : ] = sample['data'][indices,:]
            time[0:n, : ] = sample['time'][indices,:]
            sample['data']= data
            sample['mask']= (data!=0).bool()
            sample['time']= time
        elif self.sampling_type == 'overlap_100':
            i = np.random.choice(list(np.linspace(0, 7, 15)[:14]))

            start = (i*200).astype(int)
            end = ((i+1)*200).astype(int)
            #print(start,end)
            sample['data']= sample['data'][start:end,:]
            sample['mask']= sample['mask'][start:end,:]
            sample['time']= sample['time'][start:end,:]
        else:
            sample['data']= sample['data'][:self.seqlen,:]
            sample['mask']= sample['mask'][:self.seqlen,:]
            sample['time']= sample['time'][:self.seqlen,:]
        return sample

import numpy as np
class NormalizeTime:
    def __call__(self,sample):
        time = sample['time']
        new =time - torch.min(time[time != 0]) if torch.any(time != 0) else time
        sample['time'] = np.where(new < 0, 0,new)
        return sample
#cut_lc = Cut200(2,200, 'static')

def select_window(array, num_bands = 2,window_size=200, start_index=None):
   

    band_len = max(np.count_nonzero(array, axis = 0))
    if band_len - window_size > 0:
        start = torch.randint(0, band_len - window_size, size = (1,))
        return start, start+ window_size
    if band_len - window_size == 0:
        return 0, 200
    if band_len - window_size < 0:
        return 0, 200

@dataclass
class ATATDataset(BaseDataset):
    data_root:str
    set_type:str
    experiment_type:str
    seed:int
    train_apply_transform:bool
    validation_apply_transform:bool
    train_key: str
    validation_key:str
    test_key:str
    observation_key:str 
    observation_err_key: str 
    mask_key :str
    mask_photometry_key:str
    mask_detection_key:str
    time_key :str
    time_alert_key :str
    label_key :str
    feature_key:  str
    metadata_key: str
    train_transforms: Optional[list] = None
    val_transforms: Optional[list] = None
    list_time_to_eval = [2048]

    def __post_init__(self):
        super().__init__(**{key:value for key,value in self.__dict__.items() if key not in ['train_transforms', 'val_transforms']})
        self.train_transforms = Compose(self.train_transforms) if self.train_transforms is not None else None
        self.val_transforms = Compose(self.val_transforms) if self.val_transforms is not None else None
        self.use_lightcurves  = True if 'LC' in self.experiment_type else False
        self.use_metadata  = True if 'MD' in self.experiment_type else False
        self.use_features  = True if 'FEAT' in self.experiment_type else False
        self.use_lightcurves_err  = True if 'ERR' in self.experiment_type else False
        logging.debug(f'{self.train_transforms}')
        logging.info(f'Apply train transforms: {self.train_apply_transform}')
        logging.info(f'Apply validation transforms: {self.validation_apply_transform}')
        
        self.window_normalizer = LC.TimeNormalization()
    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """
         
        _idx = self.these_idx[idx]
        data_dict = {
            #"idx": _idx,
            "labels":  self.target[_idx]
        }
        if self.use_lightcurves:
            data_dict.update({"data":torch.tensor(self.data[_idx,:,:],dtype =  torch.float)})
            data_dict.update({"time":torch.tensor(self.time[_idx,:,:],dtype =  torch.float),
                              "mask":torch.tensor(self.mask[_idx,:,:],dtype = bool)})
            if self.mask_photometry_key != '':
                data_dict.update({'mask_photometry':torch.tensor(self.mask_photometry[_idx,:,:],dtype = bool)})
            if self.mask_photometry_key != '':
                data_dict.update({'mask_detection':torch.tensor(self.mask_detection[_idx,:,:],dtype = bool)})

        if self.use_lightcurves_err:
            data_dict.update({"data_err":torch.tensor(self.data_err[_idx,:,:],dtype =  torch.float)})

        if self.use_metadata:
            data_dict.update({"metadata_feat":self.metadata_feat[_idx],})

        if self.use_features:
            data_dict.update(
                {"extracted_feat": self.extracted_feat[_idx]}
            )
        
        tabular_features = []
         
        if self.use_metadata:
            tabular_features.append(data_dict["metadata_feat"].unsqueeze(1))
            data_dict.pop('metadata_feat')
        if self.use_features:
            tabular_features.append(data_dict["extracted_feat"].unsqueeze(1))
        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)


        if all([self.set_type == 'train',self.train_transforms is not None]):
            data_dict = self.train_transforms(data_dict)
        if all([self.set_type == 'validation',self.val_transforms is not None]):
            data_dict = self.val_transforms(data_dict)
        return data_dict

    def __len__(self): 
        return len(self.these_idx)

    