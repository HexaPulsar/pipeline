import pytorch_lightning as pl
from dataclasses import dataclass
import h5py

from src.data.handlers.CustomDataset import ATATDataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torch
import logging
from src.utils.CustomParser import ATATDatasetArgs

@dataclass
class LitData(pl.LightningDataModule):
    batch_size: int
    dataset: ATATDatasetArgs
    train_use_sampler: bool = True
    val_use_sampler: bool = False
    train_shuffle: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = False
    def __post_init__(self):
        super().__init__()

    def prepare_data(self):
        h5_ = h5py.File("{}".format(self.dataset.data_root))
        assert all([self.dataset.metadata_key in h5_.keys()]), 'metadata_key {} not in dataset keys. dataset keys are {}'.format(self.dataset.metadata_key, h5_.keys())

        get_data = h5_.get("%s_%s" % ('training', self.dataset.seed))
        assert get_data is not None, '{}_{} not a key of the dataset'.format('training',self.dataset.seed)
        train_idx = get_data[:]
        get_data = h5_.get("%s_%s" % ('validation', self.dataset.seed))
        assert get_data is not None, '{}_{} not a key of the dataset'.format('validation',self.dataset.seed)
        val_idx = get_data[:]
        log_message = (
        f"Dataset Configuration:\n"
        f"{'='*30}\n"
        f"• Seed        : {self.dataset.seed}\n"
        f"• Light Curves     : {'✓' if 'LC' in   self.dataset.experiment_type  else '✗'}\n"
        f"• Metadata         : {'✓'  if 'MD' in self.dataset.experiment_type  else '✗'}\n"
        f"• Features         : {'✓' if  'FEAT' in    self.dataset.experiment_type else  '✗'}\n"
        f"• Train samples         : {len(train_idx)}\n"
        f"• Validation samples         : {len(val_idx)}\n"
        f"• Use sampler        : {self.train_use_sampler}\n"
        f"• Batch Size       : {self.batch_size}\n"
        f"{'='*30}"
        )
        logging.info(log_message)

    def get_real_classes_weights(self,labels):

        class_sample_count = np.array(
            [
                len(np.where(labels == t)[0])
                for t in np.unique(labels)
            ]
        )
        weight = 1.0 / class_sample_count**(1/3)

        uniques = np.unique(labels).astype(int)
        d = {key: value for key, value in zip(uniques, weight)}
        samples_weight = np.array([d[labels[i].item()] for i in range(len(labels))])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def setup(self, stage):
        logging.debug(f'{self.dataset.train_transforms}')
        if stage == 'fit':
            logging.info(f'Apply train transforms: {self.dataset.train_apply_transform}')
        if stage == 'validate':
            logging.info(f'Apply validation transforms: {self.dataset.validation_apply_transform}')

        return super().setup(stage)

    def train_dataloader(self):
        dataset_used = ATATDataset(set_type="train", **self.dataset)
        if self.train_use_sampler:

            samples_weight = self.get_real_classes_weights(dataset_used.labels)
            self.samples_weight = samples_weight
            sampler = WeightedRandomSampler(
               samples_weight.type("torch.DoubleTensor"), len(samples_weight)
            )
            #hier_class = map_label_tensor(dataset_used.labels)
            #sampler = MPerClassSampler(
            #    hier_class,
            ##    m=256,
            #    batch_size=self.batch_size,
            #    length_before_new_iter=len(dataset_used),
            #)
            loader = DataLoader(
                dataset_used,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=None,
                drop_last=self.drop_last,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        else:

            samples_weight = self.get_real_classes_weights(dataset_used.labels)
            self.samples_weight = samples_weight
            loader = DataLoader(
                dataset_used,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=self.train_shuffle,
                drop_last=self.drop_last,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        return loader

    def val_dataloader(self):
        dataset_used = ATATDataset(set_type="validation", **self.dataset)
        if self.val_use_sampler:
            samples_weight = self.get_real_classes_weights(dataset_used.labels)
            sampler = WeightedRandomSampler(
               samples_weight.type("torch.DoubleTensor"), len(samples_weight)
            )
            loader = DataLoader(
                dataset_used,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=self.drop_last,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        else:
            loader = DataLoader(
                    dataset_used,
                    batch_size=self.batch_size,
                    sampler=None,
                    shuffle=False,
                    drop_last=self.drop_last,
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
                drop_last=self.drop_last,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory
            )
        return loader
