import numpy as np
import logging

import h5py
import random
import torch

from torch.utils.data import Dataset
from joblib import load

from torchvision.transforms import Compose, RandomApply
from .augmentations import SCAugmentation, ThreeTimeMask

class ATATDataset(Dataset):
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
        """ dataset is composed for all samples, where self.these__idx dart to samples for each partition"""

        name = (
            "training"
            if set_type == "train" or set_type == "train_step"
            else "validation"
        )
        partition_used = seed if not same_partition else 0

        # only using partition 0
        assert partition_used == 0

        h5_ = h5py.File("{}/dataset.h5".format(data_root))

        self.these_idx = (
            h5_.get("test")[:]
            if set_type == "test"
            else h5_.get("%s_%s" % (name, partition_used))[:]
        )

        print(
            f"using set {set_type} total of idx : {len(self.these_idx)}, \
                use_lightcurves {use_lightcurves}, use_metadata {use_metadata}, use_features {use_features}, \
                    use MTA {online_opt_tt}"
        )

        self.data = h5_.get("flux")
        self.data_err = h5_.get("flux_err")
        self.mask = h5_.get("mask")
        self.time = h5_.get("time")
        self.time_alert = h5_.get("time_detection")
        #self.time_phot = h5_.get("time_photometry")).float()
        self.target = h5_.get("labels")
        self.labels =  torch.from_numpy(self.target[:][self.these_idx]).long()
        
        self.eval_time = eval_metric  # must be a number 
        self.use_lightcurves = use_lightcurves
        self.use_lightcurves_err = use_lightcurves_err
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.use_QT = use_QT

        self.set_type = set_type
        self.force_online_opt = force_online_opt
        self.per_init_time = per_init_time
        self.online_opt_tt = online_opt_tt
        self.len = len(self.these_idx)
        self.list_time_to_eval = list_time_to_eval
        print("list_time_to_eval: ", list_time_to_eval)

        logging.info(f"Partition : {partition_used} Set Type : {set_type}")
        if self.use_metadata:
            metadata_feat = h5_.get("metadata_feat")[:]
            path_QT = f"{data_root}/quantiles/metadata/fold_{partition_used}.joblib".format(
                data_root, partition_used
            )
            self.metadata_feat = self.get_tabular_data(
                metadata_feat, path_QT, "metadata"
            )

        if self.use_features:
            self.extracted_feat = dict()
            for time_eval in self.list_time_to_eval:
                path_QT = f"./{data_root}/quantiles/features/fold_{partition_used}.joblib"
                extracted_feat = h5_.get("extracted_feat_{}".format(time_eval))[:]
                self.extracted_feat.update(
                    {
                        time_eval: self.get_tabular_data(
                            extracted_feat, path_QT, f"features_{time_eval}"
                        )
                    }
                )
        self.transforms = Compose([ThreeTimeMask(self.use_features,self.use_lightcurves,self.extracted_feat if self.use_features else None),
                                    SCAugmentation(self.per_init_time,
                                                list_time_to_eval,self.use_features,self.use_lightcurves,self.extracted_feat if self.use_features else None)
                                    ])
    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """
         
        _idx = self.these_idx[idx]
        data_dict = {
            "labels":  self.target[_idx]
        }

        data_dict.update({'idx':_idx,
                          "time": torch.tensor(self.time[_idx,:,:],dtype =  torch.float),
                            "mask": torch.tensor(self.mask[_idx,:,:],dtype = bool),} ) #if any([self.online_opt_tt,self.force_online_opt]) else None 
        if self.use_lightcurves:
            data_dict.update({"data":  torch.tensor(self.data[_idx,:,:],dtype =  torch.float),
                              })

        if self.use_lightcurves_err:
            data_dict.update({"data_err":  torch.tensor(self.data_err[_idx,:,:],dtype =  torch.float)})

        if self.use_metadata:
            data_dict.update({"metadata_feat":   torch.tensor(self.metadata_feat[_idx],dtype =  torch.float),
                              })

        if self.use_features:
            data_dict.update(
                {"extracted_feat": torch.tensor(self.extracted_feat[self.list_time_to_eval[-1]][_idx]).float()}
            )

        if self.set_type == "train":
            data_dict = self.transforms(data_dict)
         
        tabular_features = []
         
        if self.use_metadata:
            tabular_features.append(data_dict["metadata_feat"].unsqueeze(1))
            #print('post',data_dict['metadata_feat'].unsqueeze(1).shape)

        if self.use_features:
            tabular_features.append(data_dict["extracted_feat"].unsqueeze(1))

        if tabular_features:
            data_dict["tabular_feat"] = torch.cat(tabular_features, axis=0)
        
        #print(data_dict['tabular_feat'].shape)
        return data_dict

    def __len__(self):
        """length of the dataset, is necessary for consistent getitem values"""
        return len(self.these_idx)

    def get_tabular_data(self, tabular_data, path_QT, type_data):
        logging.info(f"Loading and procesing {type_data}. Using QT: {self.use_QT}")
        if self.use_QT:
            QT = load(path_QT)
            tabular_data = QT.transform(tabular_data)
        return torch.from_numpy(tabular_data).float()

    def update_mask(self, sample: dict, timeat: int):
        sample.update(
            {
                "mask": sample["mask"]
                * (sample["time_alert"] - sample["time_alert"][0, :].min() < timeat)
                * (sample["time_photo"] - sample["time_photo"][0, :].min() < timeat)
            }
        )

        return sample

