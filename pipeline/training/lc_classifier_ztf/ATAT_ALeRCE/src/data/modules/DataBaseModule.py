import pytorch_lightning as pl
from dataclasses import dataclass
from src.data.handlers.datasetHandlers import get_dataloader
from src.data.handlers.CustomDataset import ATATDataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torch
from pytorch_metric_learning.samplers import MPerClassSampler


@dataclass
class LitData(pl.LightningDataModule):
    data_root: str = "path/to/dir"
    batch_size: int = 32
    dataset_config_dict: dict
    train_use_sampler = True 
    train_shuffle=True
    num_workers=8
    pin_memory=True
    
    def train_dataloader(self):
        dataset_used=ATATDataset(set_type = 'train',**self.dataset_config_dict)
        if self.train_use_sampler:
            class_sample_count = np.array(
                [
                    len(np.where(dataset_used.labels == t)[0])
                    for t in np.unique(dataset_used.labels)
                ]
            )
            weight = (1.0 / class_sample_count)            
            samples_weight = np.array([weight[t] for t in dataset_used.labels])
            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type("torch.DoubleTensor"), len(samples_weight))
            sampler = MPerClassSampler(dataset_used.labels, m = 64, batch_size=batch_size, length_before_new_iter=len(dataset_used))
            logger.warning(f"USING SAMPLER FOR DATASET {set_type}")
            loader = DataLoader(dataset_used, sampler=sampler, **loader_kwargs)
        else:
            logger.warning(f"NOT USING SAMPLER FOR DATASET {set_type}")
            loader = DataLoader(dataset_used, sampler=None, shuffle=True, drop_last = True,**loader_kwargs)
        return loader
         
    def val_dataloader(self):
        dataset_used=ATATDataset(set_type = 'validation',**self.dataset_config_dict)
        loader = DataLoader(dataset_used, sampler=None, **loader_kwargs)
        return loader
    
    def test_dataloader(self):
        dataset_used=ATATDataset(set_type = 'test',**self.dataset_config_dict)
        loader = DataLoader(dataset_used, sampler=None, **loader_kwargs)
        return loader
     

