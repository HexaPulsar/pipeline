import pytorch_lightning as pl
from dataclasses import dataclass, asdict
import h5py

from src.data.handlers.CustomDataset import ATATDataset
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import torch
import logging
from src.utils.CustomParser import ATATDatasetArgs

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = None
    OmegaConf = None


def _to_dict(obj):
    """Convert an object to a dict, handling DictConfig and dataclass objects."""
    if DictConfig is not None and isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    elif hasattr(obj, '__dataclass_fields__'):
        return asdict(obj)
    elif isinstance(obj, dict):
        return obj
    else:
        # Try as a dict-like object
        try:
            return dict(obj)
        except (TypeError, ValueError):
            # Last resort: try __dict__
            return dict(obj.__dict__)


@dataclass
class LitPretrainAR(pl.LightningDataModule):
    batch_size: int
    dataset: ATATDatasetArgs
    train_use_sampler: bool = False
    val_use_sampler: bool = False
    train_shuffle: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = True
    eval_probe: bool = False

    def __post_init__(self):
        super().__init__()
        # Convert data_root from string representation of list to actual list if needed
        if isinstance(self.dataset.data_root, str):
            # Check if it looks like a list (starts with '[')
            if self.dataset.data_root.startswith('['):
                import ast
                self.dataset.data_root = ast.literal_eval(self.dataset.data_root)

    def prepare_data(self):
        # Handle both single file (string) and multiple files (list)
        if isinstance(self.dataset.data_root, str):
            data_roots = [self.dataset.data_root]
        else:
            data_roots = self.dataset.data_root

        total_train = 0
        total_val = 0

        for data_root in data_roots:
            h5_ = h5py.File("{}".format(data_root))
            assert all(
                [self.dataset.metadata_key in h5_.keys()]
            ), "metadata_key {} not in dataset keys. dataset keys are {}".format(
                self.dataset.metadata_key, h5_.keys()
            )

            get_data = h5_.get("%s_%s" % ("training", self.dataset.seed))
            assert (
                get_data is not None
            ), "{}_{} not a key of the dataset".format("training", self.dataset.seed)
            train_idx = get_data[:]
            get_data = h5_.get("%s_%s" % ("validation", self.dataset.seed))
            assert (
                get_data is not None
            ), "{}_{} not a key of the dataset".format("validation", self.dataset.seed)
            val_idx = get_data[:]

            total_train += len(train_idx)
            total_val += len(val_idx)
            h5_.close()

        log_message = (
            f"Dataset Configuration (AR Pretraining):\n"
            f"{'='*30}\n"
            f"• Seed        : {self.dataset.seed}\n"
            f"• Train samples         : {total_train}\n"
            f"• Validation samples         : {total_val}\n"
            f"• Batch Size       : {self.batch_size}\n"
            f"• Eval probe       : {self.eval_probe}\n"
            f"{'='*30}"
        )
        logging.info(log_message)

    def setup(self, stage):
        return super().setup(stage)

    def train_dataloader(self):
        # Convert dataset to dict (handles both dataclass and DictConfig)
        dataset_dict = _to_dict(self.dataset)

        # Filter out internal dataclass fields (those starting with _)
        dataset_dict_filtered = {k: v for k, v in dataset_dict.items() if not k.startswith('_')}

        if isinstance(self.dataset.data_root, str):
            dataset_used = ATATDataset(set_type="train", **dataset_dict_filtered)
        else:
            datasets = []
            for data_root in self.dataset.data_root:
                # Create a copy and override data_root
                config = dict(dataset_dict_filtered)
                config['data_root'] = data_root
                datasets.append(ATATDataset(set_type="train", **config))
            dataset_used = ConcatDataset(datasets)
            logging.info(f"Combined train dataset size: {len(dataset_used)}")

        loader = DataLoader(
            dataset_used,
            batch_size=self.batch_size,
            sampler=None,
            shuffle=self.train_shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        # For AR pretraining with probe eval, return 3 dataloaders:
        # Index 0: validation split (for reconstruction loss)
        # Index 1: training split (for fitting the probe)
        # Index 2: validation split (for scoring the probe)

        # Convert dataset to dict (handles both dataclass and DictConfig)
        dataset_dict = _to_dict(self.dataset)

        # Filter out internal dataclass fields (those starting with _)
        dataset_dict_filtered = {k: v for k, v in dataset_dict.items() if not k.startswith('_')}

        # Handle multiple data_root files
        if isinstance(self.dataset.data_root, str):
            val_dataset = ATATDataset(set_type="validation", **dataset_dict_filtered)
            train_dataset = ATATDataset(set_type="train", **dataset_dict_filtered)
        else:
            val_datasets = []
            train_datasets = []
            for data_root in self.dataset.data_root:
                # Create a copy and override data_root
                config = dict(dataset_dict_filtered)
                config['data_root'] = data_root
                val_datasets.append(ATATDataset(set_type="validation", **config))
                train_datasets.append(ATATDataset(set_type="train", **config))
            val_dataset = ConcatDataset(val_datasets)
            train_dataset = ConcatDataset(train_datasets)
            logging.info(f"Combined val dataset size: {len(val_dataset)}")
            logging.info(f"Combined train dataset size (for probe): {len(train_dataset)}")

        if self.eval_probe:
            val_loader_recon = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            train_loader_probe = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            val_loader_probe = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            return [val_loader_recon, train_loader_probe, val_loader_probe]
        else:
            loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            return loader

    def test_dataloader(self):
        # Convert dataset to dict (handles both dataclass and DictConfig)
        dataset_dict = _to_dict(self.dataset)
        dataset_used = ATATDataset(set_type="test", **dataset_dict)
        loader = DataLoader(
            dataset_used,
            batch_size=self.batch_size,
            sampler=None,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader
