import pytorch_lightning as pl
from src.data.handlers.SSLDatasetMM import SSLDatasetMM
import glob
from torch.utils.data import ConcatDataset
import logging
from torchvision.transforms import Compose, RandomApply, RandomChoice
from typing import Union, Optional
from dataclasses import asdict
from torch.utils.data import DataLoader

from src.utils.CustomParser import SSLDatasetArgs
from dataclasses import dataclass
from src.data.handlers.CustomDataset import ATATDataset
from torch.utils.data import SequentialSampler
@dataclass
class LitPretrainMM(pl.LightningDataModule):
    batch_size: int
    dataset: SSLDatasetArgs
    train_use_sampler: bool = False
    val_use_sampler: bool = False
    train_shuffle: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    def __post_init__(self):
        super().__init__()

    def train_dataloader(self):
        if isinstance(self.dataset.data_root, str):
            dataset_used = SSLDatasetMM(set_type="train", **self.dataset)
        else:
            datasets = []
            for i in range(len(self.dataset.data_root)):
                dataset_config = {key:value for key,value in self.dataset.items() if key != 'data_root'}
                datasets.append(SSLDatasetMM(set_type='train',data_root = self.dataset.data_root[i], **dataset_config))
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
            dataset_used = SSLDatasetMM(set_type="validation", **self.dataset)
        else:
            datasets = []
            for i in range(len(self.dataset.data_root)):
                dataset_config = {key:value for key,value in self.dataset.items() if key != 'data_root'}
                print(dataset_config)
                datasets.append(SSLDatasetMM(set_type='validation',data_root = self.dataset.data_root[i], **dataset_config))
            dataset_used = ConcatDataset(datasets)
            print('full dataset size:', len(dataset_used))
            
        unlabeled_loader = DataLoader(
                dataset_used,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=True,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        
        labeled_train_dataset = ATATDataset(data_root = '/home/mdelafuente/ZTF_SSL_Dataset/data/H5_files/BY_PARTITION/200_FF.h5',
                                            set_type = 'train',
                                            experiment_type=self.dataset.experiment_type,
                                             train_apply_transform=False,
                                              validation_apply_transform=False,
                                               seed = self.dataset.seed,
                                                transforms = [],  # List of transform modules
                                                train_key =  'training',
                                                validation_key =  'validation',
                                                test_key =  'test',
                                                observation_key =  'flux' , 
                                                observation_err_key =  'flux_err',
                                                time_key =  'time',
                                                time_alert_key =  'time_alert' ,
                                                mask_key =  'mask',
                                                mask_photometry_key =  '',
                                                mask_detection_key =  '',
                                                feature_key =  'extracted_features',
                                                metadata_key =  'metadata_feat',
                                                label_key =  'labels' )
        
        labeled_val_dataset = ATATDataset(data_root = '/home/mdelafuente/ZTF_SSL_Dataset/data/H5_files/BY_PARTITION/200_FF.h5',
                                            set_type = 'validation',
                                            experiment_type=self.dataset.experiment_type,
                                            train_apply_transform=False,
                                            validation_apply_transform=False,
                                            seed = self.dataset.seed,
                                            transforms = [],  # List of transform modules
                                            train_key =  'training',
                                            validation_key =  'validation',
                                            test_key =  'test',
                                            observation_key =  'flux' ,
                                            observation_err_key =  'flux_err',
                                            time_key =  'time',
                                            time_alert_key =  'time_alert' ,
                                            mask_key =  'mask',
                                            mask_photometry_key =  '',
                                            mask_detection_key =  '',
                                            feature_key =  'extracted_features',
                                            metadata_key =  'metadata_feat',
                                            label_key =  'labels' )
        
        labeled_train_loader = DataLoader(
                labeled_train_dataset,
                batch_size=self.batch_size,
                sampler=SequentialSampler(labeled_train_dataset),  # prevents sharding,
                shuffle=False,
                drop_last=False,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        labeled_val_loader = DataLoader(
                labeled_val_dataset,
                batch_size=self.batch_size,
                sampler=SequentialSampler(labeled_val_dataset),
                shuffle=False,
                drop_last=False,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        return [unlabeled_loader, labeled_train_loader, labeled_val_loader]

    def test_dataloader(self):
        if isinstance(self.dataset.data_root, str):
            dataset_used = SSLDatasetMM(set_type="test", **self.dataset)
        else:
            datasets = []
            
            for i in range(len(self.dataset.data_root)):
                dataset_config = {key:value for key,value in self.dataset.items() if key != 'data_root'}
                datasets.append(SSLDatasetMM(set_type='test',data_root = self.dataset.data_root[i], **dataset_config))
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