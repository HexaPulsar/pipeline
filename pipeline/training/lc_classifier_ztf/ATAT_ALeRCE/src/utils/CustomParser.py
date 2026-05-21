
from dataclasses import dataclass
from typing import Union, Optional, Any

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
    mask_photometry_key:str = 'mask_photometry'
    mask_detection_key:str = 'mask_detection'
    feature_key:str  = 'feat_cols'
    metadata_key:str = 'metadata_feat'
    metadata_qt_path = ''
    feature_qt_path = ''
    label_key:str= 'labels'

@dataclass
class ATATDatasetArgs(BaseDatasetArgs):
    experiment_type: str = ''
    train_transforms:Optional[list] = None
    val_transforms:Optional[list] = None
    norm_stats_path: Optional[str] = None
    
@dataclass
class SSLDatasetArgs(BaseDatasetArgs):
    experiment_type: str = ''         
    transforms_1: Optional[list] = None
    transforms_2: Optional[list] = None

@dataclass
class VICRegArgs:
    inv_coeff: int = 25
    var_coeff: int = 25
    cov_coeff: int = 1

@dataclass
class DataModuleArgs:
    dataset: Any
    train_use_sampler:bool = True
    val_use_sampler:bool = False
    train_shuffle:bool=True
    num_workers:int=8
    pin_memory:bool =True
    batch_size: int = 32
    eval_probe:bool = False

@dataclass
class TabularArgs:
    embedding_size:int =  128
    embedding_size_sub:int = 256
    num_heads:int = 4
    num_encoders:int = 3
    length_size:int = 0
    dropout: float = 0.01
    checkpoint: Optional[str] = None
    freeze_weights:bool = False
    sequence_norm:bool = True

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
    checkpoint: Optional[str] = None
    freeze_weights:bool = False
    metadata_num_features:int = 6
    features_num_features:int = 181
    use_velocity:bool = False 
    use_acceleration:bool = False 
    use_stats :bool= False 
    use_metadata:bool =False
    use_features:bool =False
    use_sequence_norm:bool = False
    use_timefilm_gelu:bool = False
    use_timefilm_norm: bool= False
    use_exp:bool = False
    use_conv:bool = False
    use_tabular_transformer:bool = False
    use_anomaly_gate:bool = False
    use_causal:bool = False

@dataclass
class VICRegArgs:
    layers:str
    inv_coeff: float
    var_coeff: float
    cov_coeff: float
     
@dataclass
class PretrainArgs:
    warmup_steps: int = 1000
    total_steps: int = 100000
    eta_min_factor: float = 0.1

@dataclass
class ATATConfig:
    experiment_type: str
    experiment_name: str
    online_transforms: Optional[list]
    lc: Optional[LightcurveArgs]
    tab: Optional[TabularArgs]
    datamodule: DataModuleArgs
    vicreg: Optional[VICRegArgs]
    callbacks: dict
    loggers: dict
    trainer: dict
    learning_rate: float
    save_dir_path: str
    log_filename: str
    num_classes:int
    mode: str
    monitor: Optional[str]
    checkpoint: Optional[str] = None
    warmup_steps: int = 1000
    eta_min_factor: float = 1e-2
    scheduler_type: Optional[str] = None
    scheduler_t_max: int = 100
    pretrain: Optional[PretrainArgs] = None
    context_size: int = 1
    normalize_flux: bool = False

 