
import os
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import glob
import numpy as np
import yaml
from typing import Dict, List
from pipeline.lc_classifier.lc_classifier.features.composites.core.base import AstroObject
from pipeline.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from pipeline.lc_classifier.lc_classifier.features.preprocess.ztf import (
    ZTFLightcurvePreprocessor,
    ShortenPreprocessor,
)
import warnings

from scipy.optimize import OptimizeWarning

from pipeline.training.lc_classifier_ztf.ATAT_ALeRCE.data.src.processing import normalizing_time
print('SUPRESSING WARNINGS') 


ZTF_ff_columns_to_PROD = {
    "g-r_mean_g,r": "g_r_mean_12",
    "g-r_max_g,r": "g_r_max_12",
    "g-r_mean_corr_g,r": "g_r_mean_corr_12",
    "g-r_max_corr_g,r": "g_r_max_corr_12",
    "W1-W2": "W1_W2",
    "W2-W3": "W2_W3",
    "W3-W4": "W3_W4",
    "g-W1": "g_W1",
    "r-W1": "r_W1",
    "g-W2": "g_W2",
    "r-W2": "r_W2",
    "g-W3": "g_W3",
    "r-W3": "r_W3",
    "g-W4": "g_W4",
    "r-W4": "r_W4",
    "MHPS_ratio_g": "MHPS_ratio_1",
    "MHPS_low_g": "MHPS_low_1",
    "MHPS_high_g": "MHPS_high_1",
    "MHPS_non_zero_g": "MHPS_non_zero_1",
    "MHPS_PN_flag_g": "MHPS_PN_flag_1",
    "MHPS_ratio_r": "MHPS_ratio_2",
    "MHPS_low_r": "MHPS_low_2",
    "MHPS_high_r": "MHPS_high_2",
    "MHPS_non_zero_r": "MHPS_non_zero_2",
    "MHPS_PN_flag_r": "MHPS_PN_flag_2",
    "GP_DRW_sigma_g": "GP_DRW_sigma_1",
    "GP_DRW_tau_g": "GP_DRW_tau_1",
    "GP_DRW_sigma_r": "GP_DRW_sigma_2",
    "GP_DRW_tau_r": "GP_DRW_tau_2",
    "Multiband_period_g,r": "Multiband_period_12",
    "PPE_g,r": "PPE_12",
    "Period_band_g": "Period_band_1",
    "delta_period_g": "delta_period_1",
    "Period_band_r": "Period_band_2",
    "delta_period_r": "delta_period_2",
    "Power_rate_1_4_g,r": "Power_rate_1_4_12",
    "Power_rate_1_3_g,r": "Power_rate_1_3_12",
    "Power_rate_1_2_g,r": "Power_rate_1_2_12",
    "Power_rate_2_g,r": "Power_rate_2_12",
    "Power_rate_3_g,r": "Power_rate_3_12",
    "Power_rate_4_g,r": "Power_rate_4_12",
    "Psi_CS_g": "Psi_CS_1",
    "Psi_eta_g": "Psi_eta_1",
    "Psi_CS_r": "Psi_CS_2",
    "Psi_eta_r": "Psi_eta_2",
    "Amplitude_g": "Amplitude_1",
    "AndersonDarling_g": "AndersonDarling_1",
    "Autocor_length_g": "Autocor_length_1",
    "Beyond1Std_g": "Beyond1Std_1",
    "Con_g": "Con_1",
    "Eta_e_g": "Eta_e_1",
    "Gskew_g": "Gskew_1",
    "MaxSlope_g": "MaxSlope_1",
    "Mean_g": "Mean_1",
    "Meanvariance_g": "Meanvariance_1",
    "MedianAbsDev_g": "MedianAbsDev_1",
    "MedianBRP_g": "MedianBRP_1",
    "PairSlopeTrend_g": "PairSlopeTrend_1",
    "PercentAmplitude_g": "PercentAmplitude_1",
    "Q31_g": "Q31_1",
    "Rcs_g": "Rcs_1",
    "Skew_g": "Skew_1",
    "SmallKurtosis_g": "SmallKurtosis_1",
    "Std_g": "Std_1",
    "StetsonK_g": "StetsonK_1",
    "Pvar_g": "Pvar_1",
    "ExcessVar_g": "ExcessVar_1",
    "SF_ML_amplitude_g": "SF_ML_amplitude_1",
    "SF_ML_gamma_g": "SF_ML_gamma_1",
    "IAR_phi_g": "IAR_phi_1",
    "LinearTrend_g": "LinearTrend_1",
    "Amplitude_r": "Amplitude_2",
    "AndersonDarling_r": "AndersonDarling_2",
    "Autocor_length_r": "Autocor_length_2",
    "Beyond1Std_r": "Beyond1Std_2",
    "Con_r": "Con_2",
    "Eta_e_r": "Eta_e_2",
    "Gskew_r": "Gskew_2",
    "MaxSlope_r": "MaxSlope_2",
    "Mean_r": "Mean_2",
    "Meanvariance_r": "Meanvariance_2",
    "MedianAbsDev_r": "MedianAbsDev_2",
    "MedianBRP_r": "MedianBRP_2",
    "PairSlopeTrend_r": "PairSlopeTrend_2",
    "PercentAmplitude_r": "PercentAmplitude_2",
    "Q31_r": "Q31_2",
    "Rcs_r": "Rcs_2",
    "Skew_r": "Skew_2",
    "SmallKurtosis_r": "SmallKurtosis_2",
    "Std_r": "Std_2",
    "StetsonK_r": "StetsonK_2",
    "Pvar_r": "Pvar_2",
    "ExcessVar_r": "ExcessVar_2",
    "SF_ML_amplitude_r": "SF_ML_amplitude_2",
    "SF_ML_gamma_r": "SF_ML_gamma_2",
    "IAR_phi_r": "IAR_phi_2",
    "LinearTrend_r": "LinearTrend_2",
    "sgscore1": "sgscore1",
    "dist_nr": "dist_nr",
    "ps_g-r": "ps_g_r",
    "SPM_A_g": "SPM_A_1",
    "SPM_t0_g": "SPM_t0_1",
    "SPM_gamma_g": "SPM_gamma_1",
    "SPM_beta_g": "SPM_beta_1",
    "SPM_tau_rise_g": "SPM_tau_rise_1",
    "SPM_tau_fall_g": "SPM_tau_fall_1",
    "SPM_A_r": "SPM_A_2",
    "SPM_t0_r": "SPM_t0_2",
    "SPM_gamma_r": "SPM_gamma_2",
    "SPM_beta_r": "SPM_beta_2",
    "SPM_tau_rise_r": "SPM_tau_rise_2",
    "SPM_tau_fall_r": "SPM_tau_fall_2",
    "SPM_chi_g": "SPM_chi_1",
    "SPM_chi_r": "SPM_chi_2",
    "TDE_decay_g": "TDE_decay_1",
    "TDE_decay_chi_g": "TDE_decay_chi_1",
    "TDE_decay_r": "TDE_decay_2",
    "TDE_decay_chi_r": "TDE_decay_chi_2",
    "fleet_a_g": "fleet_a_1",
    "fleet_w_g": "fleet_w_1",
    "fleet_chi_g": "fleet_chi_1",
    "fleet_a_r": "fleet_a_2",
    "fleet_w_r": "fleet_w_2",
    "fleet_chi_r": "fleet_chi_2",
    "color_variation_g,r": "color_variation_12",
    "positive_fraction_g": "positive_fraction_1",
    "n_forced_phot_band_before_g": "n_forced_phot_band_before_1",
    "dbrightness_first_det_band_g": "dbrightness_first_det_band_1",
    "dbrightness_forced_phot_band_g": "dbrightness_forced_phot_band_1",
    "last_brightness_before_band_g": "last_brightness_before_band_1",
    "max_brightness_before_band_g": "max_brightness_before_band_1",
    "median_brightness_before_band_g": "median_brightness_before_band_1",
    "n_forced_phot_band_after_g": "n_forced_phot_band_after_1",
    "max_brightness_after_band_g": "max_brightness_after_band_1",
    "median_brightness_after_band_g": "median_brightness_after_band_1",
    "positive_fraction_r": "positive_fraction_2",
    "n_forced_phot_band_before_r": "n_forced_phot_band_before_2",
    "dbrightness_first_det_band_r": "dbrightness_first_det_band_2",
    "dbrightness_forced_phot_band_r": "dbrightness_forced_phot_band_2",
    "last_brightness_before_band_r": "last_brightness_before_band_2",
    "max_brightness_before_band_r": "max_brightness_before_band_2",
    "median_brightness_before_band_r": "median_brightness_before_band_2",
    "n_forced_phot_band_after_r": "n_forced_phot_band_after_2",
    "max_brightness_after_band_r": "max_brightness_after_band_2",
    "median_brightness_after_band_r": "median_brightness_after_band_2",
    "ulens_u0_g": "ulens_u0_1",
    "ulens_tE_g": "ulens_tE_1",
    "ulens_fs_g": "ulens_fs_1",
    "ulens_chi_g": "ulens_chi_1",
    "ulens_u0_r": "ulens_u0_2",
    "ulens_tE_r": "ulens_tE_2",
    "ulens_fs_r": "ulens_fs_2",
    "ulens_chi_r": "ulens_chi_2",
    "Harmonics_mse_g": "Harmonics_mse_1",
    "Harmonics_chi_g": "Harmonics_chi_1",
    "Harmonics_mag_1_g": "Harmonics_mag_1_1",
    "Harmonics_mag_2_g": "Harmonics_mag_2_1",
    "Harmonics_phase_2_g": "Harmonics_phase_2_1",
    "Harmonics_mag_3_g": "Harmonics_mag_3_1",
    "Harmonics_phase_3_g": "Harmonics_phase_3_1",
    "Harmonics_mag_4_g": "Harmonics_mag_4_1",
    "Harmonics_phase_4_g": "Harmonics_phase_4_1",
    "Harmonics_mag_5_g": "Harmonics_mag_5_1",
    "Harmonics_phase_5_g": "Harmonics_phase_5_1",
    "Harmonics_mag_6_g": "Harmonics_mag_6_1",
    "Harmonics_phase_6_g": "Harmonics_phase_6_1",
    "Harmonics_mag_7_g": "Harmonics_mag_7_1",
    "Harmonics_phase_7_g": "Harmonics_phase_7_1",
    "Harmonics_mse_r": "Harmonics_mse_2",
    "Harmonics_chi_r": "Harmonics_chi_2",
    "Harmonics_mag_1_r": "Harmonics_mag_1_2",
    "Harmonics_mag_2_r": "Harmonics_mag_2_2",
    "Harmonics_phase_2_r": "Harmonics_phase_2_2",
    "Harmonics_mag_3_r": "Harmonics_mag_3_2",
    "Harmonics_phase_3_r": "Harmonics_phase_3_2",
    "Harmonics_mag_4_r": "Harmonics_mag_4_2",
    "Harmonics_phase_4_r": "Harmonics_phase_4_2",
    "Harmonics_mag_5_r": "Harmonics_mag_5_2",
    "Harmonics_phase_5_r": "Harmonics_phase_5_2",
    "Harmonics_mag_6_r": "Harmonics_mag_6_2",
    "Harmonics_phase_6_r": "Harmonics_phase_6_2",
    "Harmonics_mag_7_r": "Harmonics_mag_7_2",
    "Harmonics_phase_7_r": "Harmonics_phase_7_2",
    "Timespan": "Timespan",
    "Coordinate_x": "Coordinate_x",
    "Coordinate_y": "Coordinate_y",
    "Coordinate_z": "Coordinate_z",
}


with open("/home/mdelafuente/batch_processing/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/dict_info.yaml", 'r') as stream:
    config = yaml.safe_load(stream)


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

    def extract_diff_flux_per_band(self,detections)-> Dict:
        detections_per_band = {}
        for band in detections['fid'].unique():
            ith_band = detections[detections['fid'] == f'{band}'  ]
            ith_band = ith_band[ith_band['unit'] == 'diff_flux']
            detections_per_band.update({f'{band}': ith_band})
        return detections_per_band


  
    def band_list_to_time_flux_mask_arrays(self, band_detection_dict: Dict, seq_len: int = 200) -> tuple:
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

        def window_feature_cacl(window_in_mjd):
            
            pass
        
        
def clear_export_directory(directory):
        """Clear the export directory after user confirmation."""
        if os.path.exists(directory):
            confirm = input(f"The directory '{directory}' already exists. Do you want to delete it? (y/n): ")
            if confirm.lower() == 'y':
                shutil.rmtree(directory)
                print(f"Directory '{directory}' deleted.")
                os.makedirs(directory)
                print(f"New directory '{directory}' created.")
            else:
                print("Directory deletion canceled. Continuing without deletion.")
        else:
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")


def process_chunk(chunk, output_dir,pipe):
     
    #print('initialized pipe')

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

aos_paths = glob.glob('/home/mdelafuente/SSL/ao_2020/*')  # [3:]
out_directory = '/home/mdelafuente/SSL/out_directory/'
n_jobs = 8

# Pre-instantiate a pipe for each process
pipes = [AO2ATAT(out_directory, config) for _ in range(n_jobs)]

clear_export_directory(out_directory)

chunk_size = 1
chunks = list(chunk_list(aos_paths, chunk_size))

Parallel(n_jobs=n_jobs)(
    delayed(process_chunk)(chunk, out_directory, pipes[i % n_jobs])
    for i, chunk in enumerate(tqdm(chunks, desc='Processing chunks', total=len(chunks)))
)