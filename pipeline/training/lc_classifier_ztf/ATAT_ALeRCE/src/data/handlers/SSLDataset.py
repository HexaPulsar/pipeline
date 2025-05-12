
import logging

import torch
from joblib import load
import pandas as pd


 

from dataclasses import dataclass
import logging
from torchvision.transforms import Compose, RandomApply
from .BaseDataset import BaseDataset
from typing import Union, Optional
from torchvision.transforms import Compose
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
    time_key :str
    time_alert_key :str
    label_key :str
    feature_key:  str
    metadata_key: str
    transforms_1: Optional[list] = None
    transforms_2: Optional[list] = None

    def __post_init__(self):
        super().__init__(**{key:value for key,value in self.__dict__.items() if key not in ['transforms_1','transforms_2']})
        self.use_lightcurves  = True if 'LC' in self.experiment_type else False
        self.use_metadata  = True if 'MD' in self.experiment_type else False
        self.use_features  = True if 'FEAT' in self.experiment_type else False
        self.use_lightcurves_err  = True if 'ERR' in self.experiment_type else False       
        logging.info(f'{self.transforms_1}')
        logging.info(f'{self.transforms_2}') 

        self.transforms_1 =  Compose(self.transforms_1)
        self.transforms_2 =  Compose(self.transforms_2)
        

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
 
    def get_lc(self,_idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """

        data_dict = {}
        aug_data_dict = {}
            
        data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                            "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                            "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
        

        aug_data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                            "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                            "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
        
        data_dict = self.transforms_1(data_dict)
        aug_data_dict = self.transforms_2(aug_data_dict)

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
    
    def get_lc_md(self,_idx):
        
        data_dict = {}
        aug_data_dict = {}
            
        data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                            "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                            "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
        
        
        aug_data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                            "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                            "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
       
        tabular_features = []
        aug_tabular_features = []

        data_dict.update({"metadata_feat": torch.tensor(self.metadata_feat[_idx],dtype = torch.float)})
        tabular_features.append(data_dict["metadata_feat"])
        
        
        aug_data_dict.update({"metadata_feat": torch.tensor(self.metadata_feat[_idx],dtype = torch.float).clone()})
        aug_tabular_features.append(aug_data_dict["metadata_feat"])
        
        
        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
            aug_data_dict["tabular_feat"] = torch.cat(aug_tabular_features, axis=0)
        
        #data_dict = self.transforms_1(data_dict)
        #if self.output_augmented_batch:
        #    aug_data_dict = self.transforms_2(aug_data_dict)
        #return (data_dict, aug_data_dict) if self.output_augmented_batch else data_dict
        del data_dict['metadata_feat']
        del aug_data_dict['metadata_feat']
        data_dict = self.transforms_1(data_dict)
        aug_data_dict = self.transforms_2(aug_data_dict)
        return (data_dict, aug_data_dict)
    
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
