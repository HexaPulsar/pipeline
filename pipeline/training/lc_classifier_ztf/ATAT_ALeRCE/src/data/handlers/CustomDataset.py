
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


class TimeNormalization:
    def __call__(self,sample):

        time = sample['time']
        mask_min = 9999999999.0 * (time == 0).float()
        # Compute minimum over non-zero time values by adding the mask
        t_min = torch.min(time.float() + mask_min)

        # Normalize and keep zeros in place
        sample['time'] = (time.float() - t_min) * (time != 0).float()
        return sample

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
        self.time_norm = TimeNormalization()
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
        #'''
        if self.use_metadata:
            md =self.metadata_feat[_idx,:].squeeze(-1)
            md[torch.isnan(md)] = -1e9
            data_dict["metadata"] =md


        if self.use_features:
            ft = self.extracted_feat[_idx,:].squeeze(-1)
            ft[torch.isnan(ft)] = -1e9
            data_dict["features"] =ft

        if self.use_features and self.use_metadata:
            data_dict["tabular_feat"] = torch.cat([md,ft], axis=-1)
        elif self.use_metadata:
            data_dict["tabular_feat"] = data_dict["metadata"]
        elif self.use_features:
            data_dict["tabular_feat"] = data_dict["features"]
        #'''
        data_dict = self.time_norm(deepcopy(data_dict))
        if all([self.set_type == 'train',self.train_transforms is not None]):
            data_dict = self.train_transforms(data_dict)
        if all([self.set_type == 'validation',self.val_transforms is not None]):
            data_dict = self.val_transforms(data_dict)

        return data_dict

    def __len__(self):
        return len(self.these_idx)
