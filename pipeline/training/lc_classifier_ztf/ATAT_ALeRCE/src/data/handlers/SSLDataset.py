from copy import deepcopy
import numpy as np
import logging

import h5py
import random
import torch

from torch.utils.data import Dataset
from joblib import load
import pandas as pd
from torchvision.transforms import Compose, RandomApply, RandomChoice
from ...augmentations import LightCurveTransform as LC
from ...augmentations import TabularTransformations as TAB


class SSLDataset(Dataset):
    def __init__(
        self,
        data_root="data/final/ZTF_ff/LC_MD_FEAT_v2",
        set_type="train",
        use_lightcurves=True,
        use_lightcurves_err=False,
        use_metadata=False,
        use_features=False,
        seed=0,
        eval_metric=None,
        force_online_opt=False,
        per_init_time=0.2,
        online_opt_tt=False,
        same_partition=False,
        use_QT=False,
        list_time_to_eval=[16, 32, 64, 128, 256, 512, 1024, 2048],
        **kwargs,
    ):
        """loading dataset from H5 file"""
        """ dataset is composed for all samples, where self.these_idx dart to samples for each partition"""

        name = (
            "train"
            if set_type == "train" or set_type == "train_step"
            else "validation"
        )
        partition_used = seed if not same_partition else 0

        # only using partition 0
        assert partition_used == 0
        seed_ = 1234
        np.random.seed(seed_)
        random.seed(seed_)
        torch.manual_seed(seed_)

        self.datasets = []
        h5_ = h5py.File(data_root)
        #print(h5_.keys())
        self.these_idx = (
            h5_.get("test")[:]
            if set_type == "test"
            else h5_.get("%s_%s" % (name, partition_used))[:]
        )
        subset_size = None #20000
        if subset_size is not None:
            subset_size = min(subset_size, len(self.these_idx))
            random.seed(seed)  # Use the same seed parameter for consistency
            self.these_idx = random.sample(list(self.these_idx), subset_size)
            print(f"Using subset of {subset_size} samples")
        print(
            f"using set {set_type} total of idx : {len(self.these_idx)}, use_lightcurves {use_lightcurves}, use_metadata {use_metadata}, use_features {use_features},  use MTA {online_opt_tt}"
        )
        
        self.data = h5_.get("flux") # flux
        self.mask = h5_.get("mask")  # mask_alert # mask
        self.time = h5_.get("time") # time_phot # time
        #self.lc_lens = []
        #for i in self.these_idx:
        #    print((self.data[i,:,:] != 0).sum())

        self.eval_time = eval_metric  # must be a number
        self.use_lightcurves = use_lightcurves
        self.use_lightcurves_err = use_lightcurves_err
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.use_QT = use_QT

        self.set_type = set_type
         
        self.per_init_time = per_init_time
        
        #self.len = self.these_idx.shape[0]
        self.list_time_to_eval = list_time_to_eval
       # print("list_time_to_eval: ", list_time_to_eval)

        logging.info(f"Partition : {partition_used} Set Type : {set_type}")
        if self.use_metadata:
            metadata_feat = h5_.get("md_cols")[:]#[self.these_idx]
            path_QT = "{}/quantiles/metadata/md_qt-fold-{}.joblib".format(
                data_root, partition_used
            )
            self.metadata_feat = self.get_tabular_data(
                metadata_feat, path_QT, "metadata"
            )
        #print(h5_.keys())
        if self.use_features:
            feat_feat = h5_.get("ft_cols")[:]#[self.these_idx]
            path_QT = "{}/quantiles/features/ft_qt-fold-{}.joblib".format(
                data_root, partition_used
            )
            self.feat_feat = self.get_tabular_data(
                feat_feat, path_QT, "features"
            )
       

        self.transforms_data_lc = Compose([LC.MaskFirstN([-1,0,1,2]),
                                    RandomChoice([LC.GaussianNoise(num_bands=2,mean = 0,std = 1e-2),
                                    LC.TimeGaussianNoise(num_bands=2,mean = 0, std = 1e-2),
                                    LC.TimeFactor([i/10 for i in range(8,13)])])
                                    
                                    ])
    
        self.transforms_aug_lc =   Compose([LC.MaskFirstN([-1,0,1,2]),
                                    RandomChoice([LC.GaussianNoise(num_bands=2,mean = 0,std = 1e-2),
                                    LC.TimeGaussianNoise(num_bands=2,mean = 0, std = 1e-2),
                                    LC.TimeFactor([i/10 for i in range(8,13)])])
                                    
                                    ])
                       
        '''
        self.transforms_aug_lc = Compose([   
                                            RandomApply([TAB.GaussianNoise()],p = 1),
                                            RandomApply([TAB.RandomMask()],p = 1),

                                            ])
        self.transforms_data_lc = Compose([ 
                                            #RandomApply([TAB.GaussianNoise()],p = 1)

                                            ])
        self.transforms_data_lc = Compose([ 
                                            
                                            RandomApply([TAB.Scale()],p = 0.5),
                                            RandomApply([TAB.Factor()],p = 0.5),
                                            RandomApply([TAB.Shift()],p = 0.5),
                                            RandomApply([TAB.RandomShift()],p = 0.5),
                                            RandomApply([TAB.Jitter()],p = 0.5),
                                            RandomApply([TAB.GaussianNoise()],p = 0.5),
                                            ])
        '''
    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """
        
        
        _idx = self.these_idx[idx]
        data_dict = {}
        aug_data_dict = {}
 
        if self.use_lightcurves:
            data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                              "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                                "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})
            aug_data_dict.update({"data": torch.tensor(self.data[_idx,:,:], dtype= torch.float),
                              "time": torch.tensor(self.time[_idx,:,:], dtype= torch.float),
                                "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool)})

        tabular_features = []
        aug_tabular_features = []
        if self.use_metadata:
            data_dict.update({"metadata_feat": self.metadata_feat[_idx]})
            tabular_features.append(data_dict["metadata_feat"])
            
            aug_data_dict.update({"metadata_feat": self.metadata_feat[_idx].clone()})
            aug_tabular_features.append(aug_data_dict["metadata_feat"])
            
        if self.use_features: 
            data_dict.update({"extracted_feat": self.feat_feat[_idx]})
            tabular_features.append(data_dict["extracted_feat"])
            
            aug_data_dict.update({"extracted_feat": self.feat_feat[_idx].clone()})
            aug_tabular_features.append(aug_data_dict["extracted_feat"])
        
        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
            aug_data_dict["tabular_feat"] = torch.cat(aug_tabular_features, axis=0)
        
        data_dict = self.transforms_data_lc(data_dict)
        aug_data_dict = self.transforms_aug_lc(aug_data_dict)
        return data_dict, aug_data_dict

    def __len__(self):
        """length of the dataset, is necessary for consistent getitem values"""
        return len(self.these_idx)
    
    def get_tabular_data(self, tabular_data, path_QT, type_data):
        logging.info(f"Loading and procesing {type_data}. Using QT: {self.use_QT}")
        if self.use_QT:
            QT = load(path_QT)
            df = pd.DataFrame(tabular_data.reshape(tabular_data.shape[0],tabular_data.shape[1]))
            df = QT.transform(df.fillna(12345)) + 0.1
            df = pd.DataFrame(df.reshape(df.shape[0],df.shape[1]) )
            df = df.fillna(0)
            df = df.values.reshape(df.shape[0],df.shape[1],1)
        return torch.Tensor(df).float()

    def update_mask(self, sample: dict, timeat: int):
        sample.update(
            {
                "mask": sample["mask"]
                * (sample["time_alert"] - sample["time_alert"][0, :].min() < timeat)
                * (sample["time_photo"] - sample["time_photo"][0, :].min() < timeat)
            }
        )

        return sample
