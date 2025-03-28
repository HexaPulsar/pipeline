from copy import deepcopy
import numpy as np
import logging

import h5py
import random
import torch

from torch.utils.data import Dataset
from joblib import load
import pandas as pd
from .augmentations import SCAugmentation, ThreeTimeMask
from ...augmentations.TabularTransformations import RandomMask
 
import src.augmentations.LightCurveTransform as LC
from dataclasses import dataclass
import logging
from torchvision.transforms import Compose, RandomApply, RandomChoice

@dataclass
class SSLDataset(Dataset):
    data_root:str
    set_type:str
    use_lightcurves:bool
    use_lightcurves_err:bool
    use_metadata:bool
    use_features:bool
    seed:int
    train_apply_transform:bool
    validation_apply_transform:bool
    transforms_1: list
    transforms_2: list

    
    def __post_init__(self):
        """loading dataset from H5 file"""
        """ dataset is composed for all samples, where self.these__idx dart to samples for each partition"""
        

        h5_ = h5py.File("{}".format(self.data_root))
        get_data = (h5_.get("test") if self.set_type == "test" else h5_.get("%s_%s" % (self.set_type, self.seed)))
        assert get_data is not None, '{}_{} not a key of the dataset'.format(self.set_type,self.seed)
        self.these_idx = get_data[:]
        self.transforms_1 = Compose(self.transforms_1)
        self.transforms_2 = Compose(self.transforms_2)
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
        if all([self.use_lightcurves,self.use_metadata,self.use_features]):
            return  self.get_lc_md_ftd(_idx)
        elif all([self.use_metadata,self.use_lightcurves]):
            return self.get_lc_md(_idx)
        elif self.use_lightcurves:
            return self.get_lc(_idx)
        elif self.use_metadata:
            return self.get_md(_idx)
        elif self.use_features:
            return self.get_ft(_idx)
        

    def __len__(self):
        """length of the dataset, is necessary for consistent getitem values"""
        return len(self.these_idx)
    
    def get_tabular_data(self, tabular_data, path_QT, type_data):
        logging.info(f"Loading and procesing {type_data}. Using QT: {self.use_QT}")
        
        QT = load(path_QT)
        df = pd.DataFrame(tabular_data.reshape(tabular_data.shape[0],tabular_data.shape[1]))
        df = QT.transform(df.fillna(12345)) + 0.1
        df = pd.DataFrame(df.reshape(df.shape[0],df.shape[1]) )
        df = df.fillna(0)
        df = df.values.reshape(df.shape[0],df.shape[1],1)
        return torch.Tensor(df).float()

    def update_mask(self, sample: dict, timeat: int):
        sample.update(
            {
                "mask": sample["mask"]
                * (sample["time_alert"] - sample["time_alert"][0, :].min() < timeat)
                * (sample["time_photo"] - sample["time_photo"][0, :].min() < timeat)
            }
        )

        return sample

    def get_lc(self,_idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """

        data_dict = {}
            
        data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                            "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                            "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
        data_dict = self.transforms_1(data_dict)
        

        aug_data_dict = {}
        aug_data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                            "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                            "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
        aug_data_dict = self.transforms_2(aug_data_dict)
        return (data_dict, aug_data_dict)
    
    def get_md(self,_idx):
        data_dict = {}
        aug_data_dict = {}
 
        
        tabular_features = []
        aug_tabular_features = []
        data_dict.update({"metadata_feat": self.metadata_feat[_idx]})
        tabular_features.append(data_dict["metadata_feat"])
        
        if self.output_augmented_batch:
        
            aug_data_dict.update({"metadata_feat": self.metadata_feat[_idx].clone()})
            aug_tabular_features.append(aug_data_dict["metadata_feat"])
         
        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
            if self.output_augmented_batch:
                aug_data_dict["tabular_feat"] = torch.cat(aug_tabular_features, axis=0)
        
        data_dict = self.transforms_1(data_dict)
        if self.output_augmented_batch:
            aug_data_dict = self.transforms_2(aug_data_dict)
        return (data_dict, aug_data_dict) if self.output_augmented_batch else data_dict
    
    def get_lc_md(self,_idx):
        
        data_dict = {}
        aug_data_dict = {}
            
        data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                            "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                            "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
        
        
        if self.output_augmented_batch:
            aug_data_dict = {}
            aug_data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                              "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                                "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
       
        
        tabular_features = []
        aug_tabular_features = []
        data_dict.update({"metadata_feat": self.metadata_feat[_idx]})
        tabular_features.append(data_dict["metadata_feat"])
        
        if self.output_augmented_batch:
        
            aug_data_dict.update({"metadata_feat": self.metadata_feat[_idx].clone()})
            aug_tabular_features.append(aug_data_dict["metadata_feat"])
         
        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
            if self.output_augmented_batch:
                aug_data_dict["tabular_feat"] = torch.cat(aug_tabular_features, axis=0)
        
        data_dict = self.transforms_1(data_dict)
        if self.output_augmented_batch:
            aug_data_dict = self.transforms_2(aug_data_dict)
        return (data_dict, aug_data_dict) if self.output_augmented_batch else data_dict
    
    def get_ft(self,_idx):

        pass
    def get_lc_md_ft(self,_idx):
        
        data_dict = {}
        aug_data_dict = {}
 
        if self.use_lightcurves:
            data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                              "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                                "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
        if self.output_augmented_batch:
            aug_data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                              "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                                "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})

        tabular_features = []
        aug_tabular_features = []
        if self.use_metadata:
            data_dict.update({"metadata_feat": self.metadata_feat[_idx]})
            tabular_features.append(data_dict["metadata_feat"])
            if self.output_augmented_batch:
            
                aug_data_dict.update({"metadata_feat": self.metadata_feat[_idx].clone()})
                aug_tabular_features.append(aug_data_dict["metadata_feat"])
            
        if self.use_features: 
            data_dict.update({"extracted_feat": self.feat_feat[_idx]})
            tabular_features.append(data_dict["extracted_feat"])
            if self.output_augmented_batch:
            
                aug_data_dict.update({"extracted_feat": self.feat_feat[_idx].clone()})
                aug_tabular_features.append(aug_data_dict["extracted_feat"])
        
        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
            if self.output_augmented_batch:
                aug_data_dict["tabular_feat"] = torch.cat(aug_tabular_features, axis=0)
        
        data_dict = self.transforms_1(data_dict)
        if self.output_augmented_batch:
            aug_data_dict = self.transforms_2(aug_data_dict)
        return (data_dict, aug_data_dict) if self.output_augmented_batch else data_dict
