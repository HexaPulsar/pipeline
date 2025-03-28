import pytorch_lightning as pl
from src.data.handlers.SSLDataset import SSLDataset
import glob
from torch.utils.data import ConcatDataset
import logging
from torchvision.transforms import Compose, RandomApply, RandomChoice
from typing import Union, Optional
from dataclasses import asdict
from torch.utils.data import DataLoader

from src.utils.CustomParser import SSLDatasetArgs
from dataclasses import dataclass

@dataclass
class LitPretrain(pl.LightningDataModule):
    batch_size: int
    dataset: SSLDatasetArgs
    train_use_sampler: bool = True
    train_shuffle: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    def __post_init__(self):
        super().__init__()

    def train_dataloader(self):
        if isinstance(self.dataset.data_root, str):
            dataset_used = SSLDataset(set_type="train", **self.dataset)
        else:
            datasets = []
            for i in range(len(self.dataset.data_root)):
                dataset_config = {key:value for key,value in self.dataset.items() if key != 'data_root'}
                datasets.append(SSLDataset(set_type='train',data_root = self.dataset.data_root[i], **dataset_config))
            dataset_used = ConcatDataset(datasets)
            print('full dataset size:', len(dataset_used))
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

    def val_dataloader(self):
        if isinstance(self.dataset.data_root, str):
            dataset_used = SSLDataset(set_type="validation", **self.dataset)
        else:
            datasets = []
            for i in range(len(self.dataset.data_root)):
                dataset_config = {key:value for key,value in self.dataset.items() if key != 'data_root'}
                print(dataset_config)
                datasets.append(SSLDataset(set_type='validation',data_root = self.dataset.data_root[i], **dataset_config))
            dataset_used = ConcatDataset(datasets)
            print('full dataset size:', len(dataset_used))
            
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
        if isinstance(self.dataset.data_root, str):
            dataset_used = SSLDataset(set_type="test", **self.dataset)
        else:
            datasets = []
            
            for i in range(len(self.dataset.data_root)):
                dataset_config = {key:value for key,value in self.dataset.items() if key != 'data_root'}
                datasets.append(SSLDataset(set_type='test',data_root = self.dataset.data_root[i], **dataset_config))
            dataset_used = ConcatDataset(datasets)
            print('full dataset size:', len(dataset_used))
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