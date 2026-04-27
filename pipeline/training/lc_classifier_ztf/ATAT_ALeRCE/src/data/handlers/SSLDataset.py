
import logging

import numpy as np
import torch
from dataclasses import dataclass
from torchvision.transforms import Compose, RandomApply
from .BaseDataset import BaseDataset
from typing import Union, Optional
from copy import deepcopy


class TimeNormalization:
    def __call__(self, sample):
        time = sample['time']
        mask_min = 9999999999.0 * (time == 0).float()
        t_min = torch.min(time.float() + mask_min)
        sample['time'] = (time.float() - t_min) * (time != 0).float()
        return sample


@dataclass
class SSLDataset(BaseDataset):
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
    transforms_1: Optional[list] = None
    transforms_2: Optional[list] = None
    list_time_to_eval = ['']

    def __post_init__(self):
        super().__init__(**{key:value for key,value in self.__dict__.items() if key not in ['transforms_1','transforms_2']})
        self.use_lightcurves  = True if 'LC' in self.experiment_type else False
        self.use_metadata  = True if 'MD' in self.experiment_type else False
        self.use_features  = True if 'FEAT' in self.experiment_type else False
        self.use_lightcurves_err  = True if 'ERR' in self.experiment_type else False
        #logging.info(f'{self.transforms_1}')
        #logging.info(f'{self.transforms_2}')

        self.transforms_1 =  Compose(self.transforms_1)
        self.transforms_2 =  Compose(self.transforms_2)
        self.time_norm = TimeNormalization()


    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """
        _idx = self.these_idx[idx]
        if all([self.use_lightcurves,self.use_metadata,self.use_features]):
            return  self.get_lc_md_ft(_idx)
        elif all([self.use_metadata,self.use_lightcurves]):
            return self.get_lc_md(_idx)
        elif self.use_lightcurves:
            return self.get_lc(_idx)
        elif self.use_metadata:
            return self.get_md(_idx)
        elif self.use_features:
            return self.get_ft(_idx)
    def __len__(self):
        return len(self.these_idx)


    def get_lc(self, _idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """
        data_dict = {
            "data": torch.from_numpy(self.data[_idx].astype(np.float32, copy=False)),
            "time": torch.from_numpy(self.time[_idx].astype(np.float32, copy=False)),
            "mask": torch.from_numpy(self.mask[_idx].astype(bool, copy=False)),
        }

        if self.mask_photometry_key != '':
            data_dict['mask_photometry'] = torch.from_numpy(self.mask_photometry[_idx].astype(bool, copy=False))
        if self.mask_detection_key != '':
            data_dict['mask_detection'] = torch.from_numpy(self.mask_detection[_idx].astype(bool, copy=False))

        if self.use_metadata:
            md = self.metadata_feat[_idx].squeeze(-1).clone()
            torch.nan_to_num_(md, nan=-1e9)
            data_dict["metadata"] = md
        if self.use_features:
            ft = self.extracted_feat[_idx].squeeze(-1).clone()
            torch.nan_to_num_(ft, nan=-1e9)
            data_dict["features"] = ft

        aug_data_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}

        aug_data_dict = self.transforms_2(aug_data_dict)
        data_dict = self.transforms_1(data_dict)

        aug_data_dict = self.time_norm(deepcopy(aug_data_dict))
        data_dict = self.time_norm(deepcopy(data_dict))

        return (data_dict, aug_data_dict)

    def get_md(self,_idx):
        data_dict = {}
        aug_data_dict = {}
        tabular_features = []
        #aug_tabular_features = []
        data_dict.update({"metadata_feat": self.metadata_feat[_idx]})
        aug_data_dict.update({"metadata_feat": self.metadata_feat[_idx]})

        tabular_features.append(data_dict["metadata_feat"])

        #if self.output_augmented_batch:

        #    aug_data_dict.update({"metadata_feat": self.metadata_feat[_idx].clone()})
        #    aug_tabular_features.append(aug_data_dict["metadata_feat"])

        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
            aug_data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
            #if self.output_augmented_batch:
            #    aug_data_dict["tabular_feat"] = torch.cat(aug_tabular_features, axis=0)

        #data_dict = self.transforms_1(data_dict)
        #if self.output_augmented_batch:
        #    aug_data_dict = self.transforms_2(aug_data_dict)
        #return (data_dict, aug_data_dict) if self.output_augmented_batch else data_dict
        return (data_dict, aug_data_dict)

    def get_lc_md(self, _idx):
        data_dict = {
            "data": torch.from_numpy(self.data[_idx].astype(np.float32, copy=False)),
            "time": torch.from_numpy(self.time[_idx].astype(np.float32, copy=False)),
            "mask": torch.from_numpy(self.mask[_idx].astype(bool, copy=False)),
        }

        md = self.metadata_feat[_idx]
        data_dict["metadata_feat"] = md
        data_dict["tabular_feat"] = md

        aug_data_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}

        data_dict = self.time_norm(deepcopy(data_dict))
        aug_data_dict = self.time_norm(deepcopy(aug_data_dict))
        data_dict = self.transforms_1(data_dict)
        aug_data_dict = self.transforms_2(aug_data_dict)
        return (data_dict, aug_data_dict)

    def get_ft(self,_idx):
        data_dict = {}
        aug_data_dict = {}
        tabular_features = []
        #aug_tabular_features = []
        data_dict.update({"metadata_feat": self.extracted_feat[_idx]})
        aug_data_dict.update({"metadata_feat": self.extracted_feat[_idx]})

        tabular_features.append(data_dict["metadata_feat"])

        #if self.output_augmented_batch:

        #    aug_data_dict.update({"metadata_feat": self.metadata_feat[_idx].clone()})
        #    aug_tabular_features.append(aug_data_dict["metadata_feat"])

        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
            aug_data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
            #if self.output_augmented_batch:
            #    aug_data_dict["tabular_feat"] = torch.cat(aug_tabular_features, axis=0)

        #data_dict = self.transforms_1(data_dict)
        #if self.output_augmented_batch:
        #    aug_data_dict = self.transforms_2(aug_data_dict)
        #return (data_dict, aug_data_dict) if self.output_augmented_batch else data_dict
        return (data_dict, aug_data_dict)
    def get_lc_md_ft(self, _idx):
        data_dict = {
            "data": torch.from_numpy(self.data[_idx].astype(np.float32, copy=False)),
            "time": torch.from_numpy(self.time[_idx].astype(np.float32, copy=False)),
            "mask": torch.from_numpy(self.mask[_idx].astype(bool, copy=False)),
        }

        if self.mask_photometry_key != '':
            data_dict['mask_photometry'] = torch.from_numpy(self.mask_photometry[_idx].astype(bool, copy=False))
        if self.mask_detection_key != '':
            data_dict['mask_detection'] = torch.from_numpy(self.mask_detection[_idx].astype(bool, copy=False))

        md = self.metadata_feat[_idx].squeeze(-1).clone()
        torch.nan_to_num_(md, nan=-1e9)
        data_dict["metadata"] = md

        ft = self.extracted_feat[_idx].squeeze(-1).clone()
        torch.nan_to_num_(ft, nan=-1e9)
        data_dict["features"] = ft
        data_dict["tabular_feat"] = torch.cat([md, ft], dim=-1)

        aug_data_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
        data_dict = self.time_norm(deepcopy(data_dict))
        aug_data_dict = self.time_norm(deepcopy(aug_data_dict))
        data_dict = self.transforms_1(data_dict)
        aug_data_dict = self.transforms_2(aug_data_dict)

        return (data_dict, aug_data_dict)
