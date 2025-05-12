import argparse
import yaml
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass,asdict
import logging
from typing import Union, Optional, Any
from src.data.handlers.CustomDataset import ATATDataset
from src.data.handlers.SSLDataset import SSLDataset

@dataclass
class ZTFConfig:
    bands_to_use: list[int]
    classes_to_use: list[str]
    feat_cols: list[str]
    list_time_to_eval: list[int]
    mapping_classes: dict
    max_obs:int
    md_cols: list[str] 
    
@dataclass
class BaseDatasetArgs:
    data_root:str 
    experiment_type:str 
    seed:int = 0
    train_apply_transform:bool = True
    validation_apply_transform:bool  = False
    train_key:str = 'training'
    validation_key:str   = 'validation'
    test_key:str  = 'test'
    observation_key:str = 'flux'
    observation_err_key:str  = 'flux_err'
    time_key:str  = 'time'
    time_alert_key:str = 'time_alert'
    mask_key:str = 'mask'
    feature_key:str  = 'feat_cols'
    metadata_key:str = 'metadata_feat'
    label_key:str= 'labels'
    

@dataclass
class ATATDatasetArgs(BaseDatasetArgs):
    experiment_type: str = ''
    transforms:Optional[list] = None
    
    
    
@dataclass
class SSLDatasetArgs(BaseDatasetArgs):
    experiment_type: str = ''         
    transforms_1: Optional[list] = None
    transforms_2: Optional[list] = None

    
@dataclass 
class DataModuleArgs:
    dataset: Any
    train_use_sampler:bool = True 
    train_shuffle:bool=True
    num_workers:int=8
    pin_memory:bool =True   
    batch_size: int = 32


@dataclass
class TabularArgs:
    embedding_size:int =  128
    embedding_size_sub:int = 256
    num_heads:int = 4
    num_encoders:int = 3
    length_size:int = 0
    dropout: float = 0.01

     
@dataclass
class LightcurveArgs:
    input_size:int =  1
    embedding_size:int =  128
    embedding_size_sub:int =  512
    num_heads:int = 4
    num_encoders:int =  3
    Tmax:float =  1500.0
    pe_type:str = 'tm'
    num_harmonics:int = 64
    num_bands:int = 2
    dropout: float = 0.01

@dataclass
class VICRegArgs:
    layers:str
    inv_coeff: float
    var_coeff: float
    cov_coeff: float
     
 
@dataclass 
class ATATConfig:
    experiment_type: str
    experiment_name: str
    lc: Optional[LightcurveArgs]
    tab: Optional[TabularArgs]
    datamodule: DataModuleArgs
    callbacks: dict
    loggers: dict
    trainer: dict
    learning_rate: float  
    save_dir_path: str
    log_filename: str
    num_classes:int 
    mode: str
    monitor: str 
    checkpoint: Optional[str] = None
