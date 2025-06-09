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

def map_label_tensor(labels):
    mapping_dict = {
        0: 1, 1: 1, 3: 1, 5: 1, 8: 1,
        2: 2, 6: 2, 7: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2,
        4: 0, 9: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0
    }
    mapping_tensor = torch.tensor([mapping_dict.get(int(label), -1) for label in labels])
    return mapping_tensor

@dataclass
class LitData(pl.LightningDataModule):
    batch_size: int
    dataset: ATATDatasetArgs
    train_use_sampler: bool = True
    train_shuffle: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = False
    def __post_init__(self):
        super().__init__()

    def get_hier_weights(self, labels):
        hier_class = map_label_tensor(labels)
        class_sample_count = np.array(
            [
                len(np.where(hier_class == t)[0])
                for t in np.unique(hier_class)
            ]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in hier_class])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight
    
    def get_real_classes_weights(self,labels):
        class_sample_count = np.array(
            [
                len(np.where(labels == t)[0])
                for t in np.unique(labels)
            ]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def conditional_weights(self,labels):
        hier_class = map_label_tensor(labels)
        H_class_sample_count = np.array(
            [
                len(np.where(hier_class == t)[0])
                for t in np.unique(hier_class)
            ]
        )
        prob_superclass = H_class_sample_count/len(labels)

       
        hierarchy_samples_weight = np.array([prob_superclass[t] for t in hier_class])
        hierarchy_weights = torch.from_numpy(hierarchy_samples_weight)

        class_sample_count = np.array(
            [
                len(np.where(labels == t)[0])
                for t in np.unique(labels)
            ]
        )
       
        prob_subclass = class_sample_count/len(labels)

        subclass_samples_weight = np.array([prob_subclass[t] for t in labels])
        subclass_weights = torch.from_numpy(subclass_samples_weight)

        return hierarchy_weights * subclass_weights
    
    def train_dataloader(self):
        hier_importance = 1.0
        class_importance = 0.0
        assert hier_importance + class_importance == 1.0
        dataset_used = ATATDataset(set_type="train", **self.dataset)
        if self.train_use_sampler:
            print('using sampler')
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
        print('notusingsampler')
        samples_weight = self.get_real_classes_weights(dataset_used.labels)
        self.samples_weight = samples_weight
        print(samples_weight)
        loader =loader = DataLoader(
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
