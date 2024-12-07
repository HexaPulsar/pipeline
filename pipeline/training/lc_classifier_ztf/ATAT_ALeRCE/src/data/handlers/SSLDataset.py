from copy import deepcopy
import numpy as np
import logging

import h5py
import random
import torch

from torch.utils.data import Dataset
from joblib import load
import pandas as pd

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

        h5_ = h5py.File("{}/no_contamination.h5".format(data_root))
        print(h5_.keys())
        self.these_idx = (
            h5_.get("test")[:]
            if set_type == "test"
            else h5_.get("%s_%s" % (name, partition_used))[:]
        )

        print(
            f"using set {set_type} total of idx : {len(self.these_idx)}, use_lightcurves {use_lightcurves}, use_metadata {use_metadata}, use_features {use_features},  use MTA {online_opt_tt}"
        )

        self.data = torch.from_numpy(h5_.get("flux")[:][self.these_idx])  # flux
        
        self.mask = torch.from_numpy(
            h5_.get("mask")[:][self.these_idx]
        )  # mask_alert # mask
        self.time = torch.from_numpy(
            h5_.get("time")[:][self.these_idx]
        )  # time_phot # time
         

        self.eval_time = eval_metric  # must be a number
        self.max_time = 1500
        self.use_lightcurves = use_lightcurves
        self.use_lightcurves_err = use_lightcurves_err
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.use_QT = use_QT

        self.set_type = set_type
         
        self.per_init_time = per_init_time
        

        self.list_time_to_eval = list_time_to_eval
        print("list_time_to_eval: ", list_time_to_eval)

        logging.info(f"Partition : {partition_used} Set Type : {set_type}")
        if self.use_metadata:
            metadata_feat = h5_.get("md_cols")[:][self.these_idx]
            path_QT = "{}/quantiles/metadata/md_qt-fold-{}.joblib".format(
                data_root, partition_used
            )
            self.metadata_feat = self.get_tabular_data(
                metadata_feat, path_QT, "metadata"
            )

        if self.use_features:
            self.extracted_feat = dict()
            for time_eval in self.list_time_to_eval:
                path_QT = f"./{data_root}/quantiles/features/fold_{partition_used}.joblib"
                extracted_feat = h5_.get("extracted_feat_{}".format(time_eval))[:][
                    self.these_idx
                ]
                self.extracted_feat.update(
                    {
                        time_eval: self.get_tabular_data(
                            extracted_feat, path_QT, f"features_{time_eval}"
                        )
                    }
                )
                
    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """
        #print('in get item')
        data_dict = {
            "time": self.time[idx],
            "mask": self.mask[idx],
        }

        aug_data_dict = {
            "time": self.time[idx].clone(),
            "mask": self.mask[idx].clone(),
        }

        if self.use_lightcurves:
            data_dict.update({"data": self.data[idx]})
            aug_data_dict.update({"data": self.data[idx].clone()})

        if self.use_metadata:
            data_dict.update({"metadata_feat": self.metadata_feat[idx]})
            aug_data_dict.update({"metadata_feat": self.metadata_feat[idx].clone()})

        if self.use_features:
            data_dict.update({
                "extracted_feat": self.extracted_feat[self.list_time_to_eval[-1]][idx]
            })
            aug_data_dict.update({
                "extracted_feat": self.extracted_feat[self.list_time_to_eval[-1]][idx].clone()
            })
        tabular_features = []
        #print(data_dict['metadata_feat'].shape)
        if self.use_metadata:
            tabular_features.append(data_dict["metadata_feat"].float() )
            #print('post',data_dict['metadata_feat'].unsqueeze(1).shape)

        if self.use_features:
            tabular_features.append(data_dict["extracted_feat"].float())

        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)

            aug_data_dict['tabular_feat'] = data_dict['tabular_feat'].clone()
        return data_dict, aug_data_dict

    def __len__(self):
        """length of the dataset, is necessary for consistent getitem values"""
        return len(self.data)
    
    def get_tabular_data(self, tabular_data, path_QT, type_data):
        logging.info(f"Loading and procesing {type_data}. Using QT: {self.use_QT}")
        if self.use_QT:
            QT = load(path_QT)
            df = pd.DataFrame(tabular_data.reshape(tabular_data.shape[0],tabular_data.shape[1]))
            df = QT.transform(df.fillna(12345)) + 0.1
            df = pd.DataFrame(df.reshape(df.shape[0],df.shape[1]) )
            df = df.fillna(0)
            df = df.values.reshape(df.shape[0],df.shape[1],1) 
        return torch.from_numpy(df)
 

    def update_mask(self, sample: dict, timeat: int):
        sample.update(
            {
                "mask": sample["mask"]
                * (sample["time_alert"] - sample["time_alert"][0, :].min() < timeat)
                * (sample["time_photo"] - sample["time_photo"][0, :].min() < timeat)
            }
        )

        return sample
