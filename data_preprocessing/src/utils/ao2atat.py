
import os
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import glob
import numpy as np
import yaml
from typing import Dict, List 
from pipeline.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from pipeline.lc_classifier.lc_classifier.features.core.base import AstroObject
from pipeline.lc_classifier.lc_classifier.features.preprocess.ztf import (
    ZTFLightcurvePreprocessor,
    ShortenPreprocessor,
)
import warnings

from scipy.optimize import OptimizeWarning

from pipeline.training.lc_classifier_ztf.ATAT_ALeRCE.data.src.processing import normalizing_time
print('SUPRESSING WARNINGS') 

from data_preprocessing.ztf_prod_keys import ZTF_ff_columns_to_PROD


class AO2ATAT():
    def __init__(self, output_dir: str, config_dict: dict):
        """Creates data arrays in ATAT format from an AstroObject.

        Args:
            output_dir (str): _description_
            config_dict (dict): config_dict for ZTF available in alerce pipeline repo.
        """

        self.config_dict = config_dict
        self.ft_ex = ZTFFeatureExtractor()
        self.lc_ex = ZTFLightcurvePreprocessor()

    @staticmethod
    def extract_diff_flux_per_band(detections)-> Dict:
        detections_per_band = {}
        for band in detections['fid'].unique():
            ith_band = detections[detections['fid'] == f'{band}'  ]
            ith_band = ith_band[ith_band['unit'] == 'diff_flux']
            detections_per_band.update({f'{band}': ith_band})
        return detections_per_band

    @staticmethod
    def band_list_to_time_flux_mask_arrays(band_detection_dict: Dict, seq_len: int = 200):
            """
            Convert band detection dictionary to flux, time, and mask arrays with improved efficiency.
            
            Args:
                band_detection_dict: Dictionary containing band detection data
                seq_len: Sequence length for output arrays (default: 200)
                
            Returns:
                tuple: (flux, time, mask) arrays of shape (seq_len, 2)
            """
            # Initialize arrays for both bands
            flux_arr = np.zeros((seq_len, 2))
            time_arr = np.zeros((seq_len, 2))
            
            # Define band mapping for position indexing
            band_to_idx = {'g': 0, 'r': 1}
            
            # Process available bands
            for band_id, band_data in band_detection_dict.items():
                # Verify unit
                assert band_data['unit'].unique() == 'diff_flux'
                
                # Sort and extract data
                band_data = band_data.sort_values(by='mjd')
                time = band_data['mjd'].values
                flux = band_data['brightness'].values
                
                # Get correct array length
                data_len = min(time.shape[0], seq_len)
                
                # Fill arrays at correct band index
                band_idx = band_to_idx[band_id]
                flux_arr[:data_len, band_idx] = flux[:data_len]
                time_arr[:data_len, band_idx] = time[:data_len]
            
            # Create mask based on time values
            mask_arr = (time_arr > 1).astype(float)
            
            # Verify output shapes
            assert flux_arr.shape == (seq_len, 2)
            assert time_arr.shape == (seq_len, 2)
            assert mask_arr.shape == (seq_len, 2)
            
            return flux_arr, time_arr, mask_arr

    def transform(self, ao:AstroObject, windows:list = []) -> dict:
        """ Pipeline to construct minimum data to train ZTF-ATAT
            This pipeline goes through the following steps:
            - LightCurve processor
            - FeatureExtractor for entire curve
            - FeatureExtractor for curve windows (optional, given a 'windows' list)
            - feature/metadata df split (one for ft, one for md)
            - Data array construction: {oid, flux,time,mask, ft_cols, md_cols}
        Args:
            ao (AstroObject): AstroObject
            windows (list, optional): _description_. Defaults to [].

        Returns:
            dict: Dictionary containing the data arrays {oid, flux,time,mask, ft_cols, md_cols}
        """
        #TODO: add error flux processing
        ao.features = ao.features.iloc[0:0]

        self.lc_ex.preprocess_batch([ao])
        ao.detections.sort_values(by = 'fid',inplace = True)
        
        det_per_band = self.extract_diff_flux_per_band(ao.detections)
        flux, time, mask = self.band_list_to_time_flux_mask_arrays(
            det_per_band)
        time = normalizing_time(time)
        self.ft_ex.compute_features_batch([ao])

        ao.features['fid'] = ao.features['fid'].fillna('')
        ao.features["name_fid"] = ao.features.apply(
            lambda row: f"{row['name']}_{row['fid']}" if row['fid'] != '' else row['name'], axis=1
        )
        ao.features['prod'] = ao.features['name_fid'].map(
            ZTF_ff_columns_to_PROD)
        ao.features = ao.features.dropna(subset='prod')
        features = ao.features[~ao.features['prod'].isin(
            self.config_dict["md_cols"])]
        metadata = ao.features[ao.features['prod'].isin(
            self.config_dict["md_cols"])]
        return {'oid': ao.detections['oid'].unique()[0], 'flux': flux, 'time': time, 'mask': mask, 'ft_cols': features, 'md_cols': metadata}
    
def process_chunk(chunk, output_dir,pipe):
    ###WARNING SUPRESSION
    warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")
    warnings.filterwarnings("ignore", category=np.RankWarning)
    ####

    with open(chunk[0], 'rb') as f:
        ao = pickle.load(f)
    ao_dict = pipe.transform(ao)
    out_dir = os.path.join(
        output_dir, f"ao_array_{ao_dict['oid']}.pkl")
    #print(out_dir)
    with open(out_dir, "wb") as f:
        pickle.dump(ao_dict, f)

def chunk_list(lst, chunk_size):
    """Yield successive chunks of size chunk_size from list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

### RUNTIME
from data_preprocessing.utils import clear_export_directory
with open("/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/dict_info.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
aos_paths = glob.glob('/home/mdelafuente/SSL/aos/*/*',recursive=True)  # [3:]
print(len(aos_paths))
out_directory = '/home/mdelafuente/SSL/out_directory/'
n_jobs = 10

# Pre-instantiate a pipe for each process
pipes = [AO2ATAT(out_directory, config) for _ in range(n_jobs)]

clear_export_directory(out_directory)   

chunk_size = 1
chunks = list(chunk_list(aos_paths, chunk_size))

Parallel(n_jobs=n_jobs)(
    delayed(process_chunk)(chunk, out_directory, pipes[i % n_jobs])
    for i, chunk in enumerate(tqdm(chunks, desc='Processing chunks', total=len(chunks)))
)