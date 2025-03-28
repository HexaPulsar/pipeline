import argparse
import yaml
import os

from dataclasses import dataclass,asdict
import logging
from torchvision.transforms import Compose, RandomApply, RandomChoice
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
    data_root: Any
    train_key: str
    validation_key: str
    test_key:str
    experiment_type: str = ''
    seed:int=0
    train_apply_transform:bool = False
    validation_apply_transform: bool = False

@dataclass
class ATATDatasetArgs(BaseDatasetArgs):
    experiment_type: str = ''
    transforms:Optional[list] = None
    
    
@dataclass
class SSLDatasetArgs(BaseDatasetArgs):
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
    encoder_type:str ='Linear'
    length_size:int = 0
    list_time_to_eval = None
     
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
    
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
from dataclasses import asdict

class CustomParser:
    def __init__(self, model_config_yaml_path: str):
        """A custom parser implementation for ATAT training arguments using Hydra."""
        
        self.all_args = {}
        self.dataset = {}
        self.config = self._load_config(model_config_yaml_path)
        self._setup_experiment_type(self.config.general.experiment_type)
        self._setup_extra_args(self.config)
        
        self.general = asdict(self.config.general)
        self.lightcurve = asdict(self.config.lightcurve)
        self.tabular = asdict(self.config.tabular)
        self.dataset = asdict(self.config.dataset)
        self.dataloader = asdict(self.config.dataloader)
        self.dataloader.update({'dataset_config_dict': self.dataset})
    
    @staticmethod
    def _load_config(yaml_path: str) -> DictConfig:
        with open(yaml_path, "r") as yaml_file:
            config_dict = yaml.safe_load(yaml_file)
        return OmegaConf.create(config_dict)
    
    def _setup_extra_args(self, config):
        if config.general.use_lightcurves:
            config.lightcurve.input_size = 1
        if config.general.use_lightcurves_err:
            config.lightcurve.input_size = 2
        if config.general.use_metadata:
            config.tabular.length_size += len(config.md_cols)
        if config.general.use_features:
            config.tabular.length_size += len(config.feat_cols)
            config.general.list_time_to_eval = config.list_time_to_eval
    
    def _setup_experiment_type(self, experiment_type_str: str):
        experiment_type_list = experiment_type_str.split('_')
        
        # Reset variables
        self.config.general.use_lightcurves = 'lc' in experiment_type_list
        self.config.general.use_metadata = 'md' in experiment_type_list
        self.config.general.use_features = 'feat' in experiment_type_list
        self.config.general.use_QT = 'feat' in experiment_type_list
        self.config.general.online_opt_tt = 'mta' in experiment_type_list

 