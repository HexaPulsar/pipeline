import os
from typing import Dict

import warnings
from scipy.optimize import OptimizeWarning
import numpy as np

###WARNING SUPRESSION
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")
warnings.filterwarnings("ignore", category=np.RankWarning)
####
import pandas as pd
import pickle
from numba import jit
from data_preprocessing.ztf_prod_keys import ZTF_ff_columns_to_PROD 

from pipeline.lc_classifier.lc_classifier.utils import create_astro_object
from pipeline.training.lc_classifier_ztf.ATAT_ALeRCE.data.src.processing import normalizing_time
 
def extract_from_db(oids_list,engine):
    oids_chunk = [f"'{oid}'" for oid in oids_list]

    # Query for detections
    query_detections = f"""
    SELECT * FROM detection
    WHERE oid in ({','.join(oids_chunk)});
    """
    detections = pd.read_sql_query(query_detections, con=engine)
    #detections_path = os.path.join(chunk_dir, "detections.parquet")
    #detections.to_parquet(detections_path)

    # Query for forced photometry
    query_forced_photometry = f"""
    SELECT * FROM forced_photometry
    WHERE oid in ({','.join(oids_chunk)});
    """
    forced_photometry = pd.read_sql_query(query_forced_photometry, con=engine)
    #forced_photometry_path = os.path.join(chunk_dir, "forced_photometry.parquet")
    #forced_photometry.to_parquet(forced_photometry_path)

    # Query for xmatch
    query_xmatch = f"""
    SELECT oid, oid_catalog, dist FROM xmatch
    WHERE oid in ({','.join(oids_chunk)}) and catid='allwise';
    """
    xmatch = pd.read_sql_query(query_xmatch, con=engine)
    xmatch = xmatch.sort_values("dist").drop_duplicates("oid")
    oid_catalog = [f"'{oid}'" for oid in xmatch["oid_catalog"].values]

    # Query for WISE data
    query_wise = f"""
    SELECT oid_catalog, w1mpro, w2mpro, w3mpro, w4mpro FROM allwise
    WHERE oid_catalog in ({','.join(oid_catalog)});
    """
    wise = pd.read_sql_query(query_wise, con=engine).set_index("oid_catalog")
    wise = pd.merge(xmatch, wise, on="oid_catalog", how="outer")
    wise = wise[["oid", "w1mpro", "w2mpro", "w3mpro", "w4mpro"]].set_index("oid")

    # Query for PS1 data
    query_ps = f"""
    SELECT oid, sgscore1, sgmag1, srmag1, distpsnr1 FROM ps1_ztf
    WHERE oid in ({','.join(oids_chunk)});
    """
    ps = pd.read_sql_query(query_ps, con=engine)
    ps = ps.drop_duplicates("oid").set_index("oid")

    # Merge xmatch and PS1 data
    xmatch = pd.concat([wise, ps], axis=1).reset_index()
    #xmatch_path = os.path.join(chunk_dir, "xmatch.parquet")
    #xmatch.to_parquet(xmatch_path)
    return detections, forced_photometry,xmatch
 
def create_astro_objects(detections,forced_photometry,xmatch):
    oids = detections["oid"].unique()
    aos_list = []
    for oid in oids:
        try:
            xmatch_oid = xmatch[xmatch["oid"] == oid]
            assert len(xmatch_oid) == 1
            xmatch_oid = xmatch_oid.iloc[0]
            
            ao = create_astro_object(
                data_origin="database",
                detections=detections[detections["oid"] == oid],
                forced_photometry=forced_photometry[forced_photometry["oid"] == oid],
                xmatch=xmatch_oid,
                non_detections=None,
            )
            aos_list.append(ao)
        except Exception as e:
            #print(f'Skipped {oid}: assertion error no xmatch for {oid}')
            
            continue
    return aos_list
 
def extract_diff_flux_per_band(detections)-> Dict:
        detections_per_band = {}
        for band in detections['fid'].unique():
            ith_band = detections[detections['fid'] == f'{band}'  ]
            ith_band = ith_band[ith_band['unit'] == 'diff_flux']
            detections_per_band.update({f'{band}': ith_band})
        return detections_per_band
 
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
 
def transform2arrays(aos_list,config_dict,ft_ex, lc_ex):
    lc_ex.preprocess_batch(aos_list)
    ft_ex.compute_features_batch(aos_list)
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
            config_dict["md_cols"])]
        metadata = ao.features[ao.features['prod'].isin(
            config_dict["md_cols"])]
        aos_dict_arrays.append({'oid': ao.detections['oid'].unique()[0], 'flux': flux, 'time': time, 'mask': mask, 'ft_cols': features, 'md_cols': metadata})
    return aos_dict_arrays


def save_arrays(array_dict, out_dir_path):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(array_dict)
    df.set_index('oid', inplace=True)  # Ensure the 'oid' column is set as the index
    
    # Check if the file already exists
    if os.path.exists(out_dir_path):
        # If the file exists, load the existing DataFrame
        existing_df = pd.read_pickle(out_dir_path)
        # Concatenate the new DataFrame with the existing one
        df = pd.concat([existing_df, df]).drop_duplicates(keep='last')
    
    # Save the DataFrame (either new or concatenated) to the file
    df.to_pickle(out_dir_path,protocol=pickle.HIGHEST_PROTOCOL)
