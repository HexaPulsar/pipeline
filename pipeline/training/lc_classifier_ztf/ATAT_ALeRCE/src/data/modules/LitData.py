import pytorch_lightning as pl
from dataclasses import dataclass

from src.data.handlers.CustomDataset import ATATDataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torch
from pytorch_metric_learning.samplers import MPerClassSampler
import logging
from src.utils.CustomParser import ATATDatasetArgs

@dataclass
class LitData(pl.LightningDataModule):
    batch_size: int
    dataset: ATATDatasetArgs
    train_use_sampler: bool = True
    train_shuffle: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    def __post_init__(self):
        super().__init__()

    def train_dataloader(self):
        assert self.dataset.experiment_type == 'LC' , cfg.experiment_type
        dataset_used = ATATDataset(set_type="train", **self.dataset)
        if self.train_use_sampler:
            class_sample_count = np.array(
                [
                    len(np.where(dataset_used.labels == t)[0])
                    for t in np.unique(dataset_used.labels)
                ]
            )
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in dataset_used.labels])
            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(
                samples_weight.type("torch.DoubleTensor"), len(samples_weight)
            )
            sampler = MPerClassSampler(
                dataset_used.labels,
                m=64,
                batch_size=self.batch_size,
                length_before_new_iter=len(dataset_used),
            )
            
            loader = DataLoader(
                dataset_used,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=None,
                drop_last=True,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        else:
            
            loader = DataLoader(
                dataset_used,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=self.train_shuffle,
                drop_last=True,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        return loader

    def val_dataloader(self):
        dataset_used = ATATDataset(set_type="validation", **self.dataset)
        loader =loader = DataLoader(
                dataset_used,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=True,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        return loader

    def test_dataloader(self):
        dataset_used = ATATDataset(set_type="test", **self.dataset)
        loader = loader = DataLoader(
                dataset_used,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=True,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        return loader
