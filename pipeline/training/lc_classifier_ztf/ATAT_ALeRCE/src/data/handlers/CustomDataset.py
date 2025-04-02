
import logging
import torch
from src.data.handlers.BaseDataset import BaseDataset
from joblib import load
from dataclasses import dataclass
import logging
from .BaseDataset import BaseDataset
from typing import Union, Optional
from torchvision.transforms import Compose
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
    time_key :str
    time_alert_key :str
    label_key :str
    feature_key:  str
    metadata_key: str
    transforms: Optional[list] = None

    def __post_init__(self):
        super().__init__(**{key:value for key,value in self.__dict__.items() if key != 'transforms'})
        self.transforms = Compose(self.transforms) if self.transforms is not None else None
        self.use_lightcurves  = True if 'LC' in self.experiment_type else False
        self.use_metadata  = True if 'MD' in self.experiment_type else False
        self.use_features  = True if 'FEAT' in self.experiment_type else False
        self.use_lightcurves_err  = True if 'ERR' in self.experiment_type else False
        
    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """
         
        _idx = self.these_idx[idx]
        data_dict = {
            "labels":  self.target[_idx]
        }
        if self.use_lightcurves:
            data_dict.update({"data":torch.tensor(self.data[_idx,:,:],dtype =  torch.float),
                              "time":torch.tensor(self.time[_idx,:,:],dtype =  torch.float),
                              "mask":torch.tensor(self.mask[_idx,:,:],dtype = bool)})
        if self.use_lightcurves_err:
            data_dict.update({"data_err":torch.tensor(self.data_err[_idx,:,:],dtype =  torch.float)})

        if self.use_metadata:
            data_dict.update({"metadata_feat":torch.tensor(self.metadata_feat[_idx],dtype =  torch.float),})

        if self.use_features:
            data_dict.update(
                {"extracted_feat": torch.tensor(self.extracted_feat[self.list_time_to_eval[-1]][_idx], dtype = torch.float)}
            )

        if all([self.train_apply_transform, self.set_type == 'train',self.transforms is not None]):
            data_dict = self.transforms(data_dict)
        if all([self.validation_apply_transform, self.set_type == 'validation',self.transforms is not None]):
            data_dict = self.transforms(data_dict)
        
        tabular_features = []
         
        if self.use_metadata:
            tabular_features.append(data_dict["metadata_feat"].unsqueeze(1))
        if self.use_features:
            tabular_features.append(data_dict["extracted_feat"].unsqueeze(1))
        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
        return data_dict

    def __len__(self): 
        return len(self.these_idx)

    def get_tabular_data(self, tabular_data, path_QT, type_data):
        logging.info(f"Loading and procesing {type_data}. Using QT: {self.use_QT}")
        if self.use_QT:
            QT = load(path_QT)
            tabular_data = QT.transform(tabular_data)
        return torch.from_numpy(tabular_data).float()
 