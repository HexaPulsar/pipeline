import logging
import h5py
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from joblib import load
import numpy as np
import pandas as pd
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
        mask_photometry_key = '',
        mask_detection_key = '',
        time_key = 'time',
        time_alert_key = 'time_alert',
        label_key = 'labels',
        feature_key = 'ft_cols',
        metadata_key = 'metadata_feat',
        metadata_qt_dir = '',
        feature_qt_dir = '',
        list_time_to_eval = ['']):

        """loading dataset from H5 file"""
        """ dataset is composed for all samples, where self.these__idx dart to samples for each partition"""
        
        if self.set_type in ['train','train_step']:
            name = self.train_key
        elif self.set_type in ['validation']:
            name =  self.validation_key
        else:
            name = self.test_key
        self.use_lightcurves  = True if 'LC' in self.experiment_type else False
        self.use_metadata  = True if 'MD' in self.experiment_type else False
        self.use_features  = True if 'FEAT' in self.experiment_type else False
        self.use_lightcurves_err  = True if 'ERR' in self.experiment_type else False
        self.use_QT = True
        self.metadata_key = metadata_key
        self.feature_key = feature_key

        self.list_time_to_eval = list_time_to_eval

        h5_ = h5py.File("{}".format(self.data_root))
        assert all([metadata_key in h5_.keys()]), 'metadata_key {} not in dataset keys. dataset keys are {}'.format(self.metadata_key, h5_.keys())
        get_data = (h5_.get("test") if self.set_type == "test" else h5_.get("%s_%s" % (name, self.seed)))
        assert get_data is not None, '{}_{} not a key of the dataset'.format(name,self.seed)
        self.these_idx = get_data[:]
        
        """
        new_idx = []
        for i in self.these_idx:
            lc = h5_.get(self.observation_key)[i]
            if np.count_nonzero(lc, axis = (0,1)) < 6:
                continue
            else:
                new_idx.append(i)
        """
        import numpy as np

        #np.random.seed(0)
        #if set_type != 'test':
        #    self.these_idx = np.random.choice(self.these_idx,int(5e4))
        
        #assert self.use_lightcurves == True
        #logging.info(log_message)
        self.data = h5_.get(self.observation_key)
        self.data_err = h5_.get(self.observation_err_key)
        self.mask = h5_.get(self.mask_key)
        if self.mask_photometry_key !='':
            self.mask_photometry = h5_.get(self.mask_photometry_key) if self.mask_photometry_key in h5_.keys() else None
        if self.mask_detection_key !='':
            self.mask_detection = h5_.get(self.mask_photometry_key) if self.mask_detection_key in h5_.keys() else None
        self.time = h5_.get(self.time_key)
        self.time_alert = h5_.get(self.time_alert_key)
        if 'labels' in h5_.keys():
            self.target = h5_.get(self.label_key)
            self.labels =  torch.from_numpy(self.target[:][self.these_idx].astype(int))
        #if 'nonzero_count' in h5_.keys():
        #    self.target = h5_.get('nonzero_count')
        #    self.nz_count =  torch.from_numpy(self.target[:][self.these_idx])
        #logging.info(f"Partition : {self.seed} Set Type : {self.set_type}")
        use_metadata = True
        if use_metadata:
            metadata_feat = h5_.get(self.metadata_key)[:]
            path = '/'.join(self.data_root.split('/')[:-1])
            add = 'metadata_qt'
            add = 'fold'
            path_QT = f"{path}/metadata/{add}_{self.seed}.joblib".format(
                self.data_root, self.seed
            )
            #path_QT = '/home/mdelafuente/ORIGINAL/QT-New/finetune/finetune_md_fold_{}.joblib'.format(self.seed)
            self.metadata_feat = self.get_tabular_data(
                metadata_feat, path_QT, "metadata"
            )
        use_features = True 
        if use_features:
            extracted_feat = h5_.get("{}".format(self.feature_key))[:]
             
            path = '/'.join(self.data_root.split('/')[:-1])
            add = 'features_qt'
            add = 'fold'
            path_QT = f"{path}/features/{add}_{self.seed}.joblib".format(
                self.data_root, self.seed
            )   
            data = self.get_tabular_data(
                        extracted_feat, path_QT, self.feature_key    
                    )
            
            self.extracted_feat = data
    def get_tabular_data(self, tabular_data, path_QT, type_data):
        logging.info(f"Loading and procesing {type_data}. Using QT: {self.use_QT}")
        if self.use_QT:
            QT = load(path_QT)
          
             
            if tabular_data.shape[-1]  ==  1:
                tabular_data = tabular_data.squeeze(-1)
            else:
                tabular_data = tabular_data
            
            tabular_data = QT.transform(tabular_data)
            tabular_data = np.nan_to_num(tabular_data,-0.1)

            assert np.isnan(tabular_data).sum() == 0
           # print(tabular_data.isnan())
        return torch.from_numpy(tabular_data).float()
    