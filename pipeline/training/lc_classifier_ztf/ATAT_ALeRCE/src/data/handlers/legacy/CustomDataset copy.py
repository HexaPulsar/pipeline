import numpy as np
import logging

import h5py
import random
import torch

from torch.utils.data import Dataset
from joblib import load

from torchvision.transforms import Compose, RandomApply, RandomChoice
from .augmentations import SCAugmentation, ThreeTimeMask
from ...augmentations.TabularTransformations import RandomMask
 
import src.augmentations.LightCurveTransform as LC
from dataclasses import dataclass
import logging

@dataclass
class ATATDataset(BaseDataset):
    data_root:str
    set_type:str
    use_lightcurves:bool
    use_lightcurves_err:bool
    use_metadata:bool
    use_features:bool
    seed:int
    train_apply_transform:bool
    validation_apply_transform:bool
    transforms: list

    
    def __post_init__(self):
        super().__init__(BaseDataset,self)
        """loading dataset from H5 file"""
        """ dataset is composed for all samples, where self.these__idx dart to samples for each partition"""
        if self.set_type in ['train','train_step']:
            name = 'training'
        elif self.set_type in ['validation']:
            name = 'validation'
        else:
            name = 'test'

    
        h5_ = h5py.File("{}/dataset.h5".format(self.data_root))

        get_data = (h5_.get("test") if self.set_type == "test" else h5_.get("%s_%s" % (name, self.seed)))
        assert get_data is not None, '{}_{} not a key of the dataset'.format(name,self.seed)
        self.these_idx = get_data[:]
        self.transforms = Compose(self.transforms)
        log_message = (
        f"Dataset Configuration:\n"
        f"{'='*30}\n"
        f"• Set Type         : {self.set_type}\n"
        f"• Total Indices    : {len(self.these_idx)}\n"
        f"• Light Curves     : {'✓' if self.use_lightcurves else '✗'}\n"
        f"• Metadata         : {'✓' if self.use_metadata else '✗'}\n"
        f"• Features         : {'✓' if self.use_features else '✗'}\n"
        f"{'='*30}"
        )
        logging.info(log_message)
        self.data = h5_.get("flux")
        self.data_err = h5_.get("flux_err")
        self.mask = h5_.get("mask")
        self.time = h5_.get("time")
        self.time_alert = h5_.get("time_detection")
        self.target = h5_.get("labels")
        self.labels =  torch.from_numpy(self.target[:][self.these_idx]).long()
        self.use_QT = True
        logging.info(f"Partition : {self.seed} Set Type : {self.set_type}")
        
        if self.use_metadata:
            metadata_feat = h5_.get("metadata_feat")[:]
            path_QT = f"{data_root}/quantiles/metadata/fold_{partition_used}.joblib".format(
                data_root, partition_used
            )
            self.metadata_feat = self.get_tabular_data(
                metadata_feat, path_QT, "metadata"
            )
            
        if self.use_features:
            self.extracted_feat = dict()
            for time_eval in self.list_time_to_eval:
                path_QT = f"{data_root}/quantiles/features/fold_{partition_used}.joblib"
                extracted_feat = h5_.get("extracted_feat_{}".format(time_eval))[:]
                self.extracted_feat.update(
                    {
                        time_eval: self.get_tabular_data(
                            extracted_feat, path_QT, f"features_{time_eval}"    
                        )
                    }
                )
        
    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """
         
        _idx = self.these_idx[idx]
        data_dict = {
            "labels":  self.target[_idx]
        }

        data_dict.update({
                          "time": torch.tensor(self.time[_idx,:,:],dtype =  torch.float),
                            "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool),} ) #if any([self.online_opt_tt,self.force_online_opt]) else None 
        if self.use_lightcurves:
            data_dict.update({"data":  torch.tensor(self.data[_idx,:,:],dtype =  torch.float),
                              })

        if self.use_lightcurves_err:
            data_dict.update({"data_err":  torch.tensor(self.data_err[_idx,:,:],dtype =  torch.float)})

        if self.use_metadata:
            data_dict.update({"metadata_feat":   torch.tensor(self.metadata_feat[_idx],dtype =  torch.float),
                              })

        if self.use_features:
            data_dict.update(
                {"extracted_feat": torch.tensor(self.extracted_feat[self.list_time_to_eval[-1]][_idx]).float()}
            )

        if all([self.train_apply_transform, self.set_type == 'train']):
            data_dict = self.transforms(data_dict)
        if all([self.validation_apply_transform, self.set_type == 'validation']):
            data_dict = self.transforms(data_dict)
        
        tabular_features = []
         
        if self.use_metadata:
            tabular_features.append(data_dict["metadata_feat"].unsqueeze(1))

        if self.use_features:
            tabular_features.append(data_dict["extracted_feat"].unsqueeze(1))

        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
        
        return data_dict

   

    def get_tabular_data(self, tabular_data, path_QT, type_data):
        logging.info(f"Loading and procesing {type_data}. Using QT: {self.use_QT}")
        if self.use_QT:
            QT = load(path_QT)
            tabular_data = QT.transform(tabular_data)
        return torch.from_numpy(tabular_data).float()

    def update_mask(self, sample: dict, timeat: int):
        sample.update(
            {
                "mask": sample["mask"]
                * (sample["time_alert"] - sample["time_alert"][0, :].min() < timeat)
                * (sample["time_photo"] - sample["time_photo"][0, :].min() < timeat)
            }
        )

        return sample



