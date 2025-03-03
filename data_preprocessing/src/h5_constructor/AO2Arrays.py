from .utils import extract_diff_flux_per_band,band_list_to_time_flux_mask_arrays
from ..utils.ztf_prod_keys import ZTF_ff_columns_to_PROD
from pipeline.training.lc_classifier_ztf.ATAT_ALeRCE.data.src.processing import normalizing_time
from pipeline.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from pipeline.lc_classifier.lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
import yaml
import pickle


class AO2Arrays:
    def __init__(self, config_yaml,db_params):
        self.ft_extractor = ZTFFeatureExtractor()
        self.lc_extractor = ZTFLightcurvePreprocessor()
        with open(config_yaml, 'r') as stream:
            self.config = yaml.safe_load(stream)
        pass
        
    def __call__(self, *args, **kwds):
        self.lc_ex.preprocess_batch(aos_list)
        self.ft_ex.compute_features_batch(aos_list)
        aos_dict_arrays = []
        for ao in aos_list:
            
            ao.detections.sort_values(by = 'fid',inplace = True)
            
            det_per_band = extract_diff_flux_per_band(ao.detections)
            flux, time, mask = band_list_to_time_flux_mask_arrays(
                det_per_band)
            time = normalizing_time(time)
            ao.features.fillna({'fid':''}, inplace=True)
            ao.features["name_fid"] = ao.features['name'] + ao.features['fid'].where(ao.features['fid'] != '', '')
            ao.features['prod'] = ao.features['name_fid'].map(
                ZTF_ff_columns_to_PROD)
            ao.features.dropna(subset=['prod'], inplace=True)
            features = ao.features[~ao.features['prod'].isin(
                self.config_dict["md_cols"])]
            metadata = ao.features[ao.features['prod'].isin(
                self.config_dict["md_cols"])]
            aos_dict_arrays.append({'oid': ao.detections['oid'].unique()[0], 'flux': flux, 'time': time, 'mask': mask, 'ft_cols': features, 'md_cols': metadata})
        return aos_dict_arrays