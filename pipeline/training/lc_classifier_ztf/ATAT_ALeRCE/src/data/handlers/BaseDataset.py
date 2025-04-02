import logging
import h5py
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass


class BaseDataset(Dataset):
    def __init__(self,
        data_root:str,
        set_type:str,
        experiment_type:str,
        seed:int,
        train_apply_transform:bool,
        validation_apply_transform:bool,
        train_key = 'training',
        validation_key = 'validation',
        test_key  = 'test',
        observation_key:str =  'flux',
        observation_err_key: str = 'flux_err',
        mask_key = 'mask',
        time_key = 'time',
        time_alert_key = 'time_alert',
        label_key = 'labels',
        feature_key = '',
        metadata_key = ''):

        """loading dataset from H5 file"""
        """ dataset is composed for all samples, where self.these__idx dart to samples for each partition"""
        if self.set_type in ['train','train_step']:
            name = self.train_key
        elif self.set_type in ['validation']:
            name =  self.validation_key
        else:
            name = self.test_key
        print(self.experiment_type)
        self.use_lightcurves  = True if 'LC' in self.experiment_type else False
        self.use_metadata  = True if 'MD' in self.experiment_type else False
        self.use_features  = True if 'FEAT' in self.experiment_type else False
        self.use_lightcurves_err  = True if 'ERR' in self.experiment_type else False
        h5_ = h5py.File("{}".format(self.data_root))

        get_data = (h5_.get("test") if self.set_type == "test" else h5_.get("%s_%s" % (name, self.seed)))
        assert get_data is not None, '{}_{} not a key of the dataset'.format(name,self.seed)
        self.these_idx = get_data[:]
        import numpy as np

        #np.random.seed(0)
        #if set_type != 'test':
        #    self.these_idx = np.random.choice(self.these_idx,int(5e4))
        log_message = (
        f"Dataset Configuration:\n"
        f"{'='*30}\n"
        f"• Set Type         : {self.set_type}\n"
        f"• Total Indices    : {len(self.these_idx)}\n"
        f"• Light Curves     : {'✓' if self.use_lightcurves else '✗'}\n"
        f"• Metadata         : {'✓' if self.use_metadata else '✗'}\n"
        f"• Features         : {'✓' if self.use_features else '✗'}\n"
        f"{'='*30}"
        )
        assert self.use_lightcurves == True
        logging.info(log_message)
        self.data = h5_.get(self.observation_key)
        self.data_err = h5_.get(self.observation_err_key)
        self.mask = h5_.get(self.mask_key)
        self.time = h5_.get(self.time_key)
        self.time_alert = h5_.get(self.time_alert_key)
        if 'labels' in h5_.keys():
            self.target = h5_.get(self.label_key)
            self.labels =  torch.from_numpy(self.target[:][self.these_idx]).long()
        logging.info(f"Partition : {self.seed} Set Type : {self.set_type}")
        
        if self.use_metadata:
            metadata_feat = h5_.get(self.metadata_key)[:]
            path_QT = f"{self.data_root}/quantiles/metadata/fold_{self.seed}.joblib".format(
                self.data_root, self.seed
            )
            self.metadata_feat = self.get_tabular_data(
                metadata_feat, path_QT, "metadata"
            )
        if self.use_features:
            self.extracted_feat = dict()
            for time_eval in self.list_time_to_eval:
                path_QT = f"{self.data_root}/quantiles/features/fold_{self.seed}.joblib"
                extracted_feat = h5_.get("{}_{}".format(self.feature_key,time_eval))[:]
                self.extracted_feat.update(
                    {
                        time_eval: self.get_tabular_data(
                            extracted_feat, path_QT, f"features_{time_eval}"    
                        )
                    }
                )