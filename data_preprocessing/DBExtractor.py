import glob
import pickle
from typing import Dict, List
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sqlalchemy as sa
import requests
import pandas as pd
import os
import sys
from pipeline.lc_classifier.lc_classifier.features.core.base import AstroObject
from data_preprocessing.utils import clear_export_directory
import shutil
from tqdm import tqdm

from pipeline.lc_classifier.lc_classifier.utils import create_astro_object 

class DBExtractor():

    def __init__(self, path_oids_to_pull:str,path_to_save_dir:str,chunk_size:int = 100) -> None:
    
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_folder_full_path = os.path.join(script_path, path_to_save_dir)

        if not os.path.exists(data_folder_full_path):
            os.makedirs(data_folder_full_path)

        url = "https://raw.githubusercontent.com/alercebroker/usecases/master/alercereaduser_v4.json"
        params = requests.get(url).json()["params"]

        engine = sa.create_engine(
            f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}"
        )
        engine.begin()
        
        oids = pd.read_parquet(f"{path_oids_to_pull}")
        oids = oids.index.tolist()

        self.clear_export_directory(data_folder_full_path)
        for i, chunk in tqdm(enumerate(self.chunk_list(oids, chunk_size)), 
                        desc=f'Pulling data by oid chunks of size {chunk_size}',
                        total = len(oids)//chunk_size):

            # Create a subdirectory for each chunk
            chunk_dir = os.path.join(data_folder_full_path, f"chunk_{i}")
            os.makedirs(chunk_dir)

            self.process_chunk(chunk, chunk_dir, engine)


    def process_chunk(self,chunk, chunk_dir, engine):
        # Format oids for SQL query
        oids_chunk = [f"'{oid}'" for oid in chunk]

        # Query for detections
        query_detections = f"""
        SELECT * FROM detection
        WHERE oid in ({','.join(oids_chunk)});
        """
        detections = pd.read_sql_query(query_detections, con=engine)
        detections_path = os.path.join(chunk_dir, "detections.parquet")
        detections.to_parquet(detections_path)

        # Query for forced photometry
        query_forced_photometry = f"""
        SELECT * FROM forced_photometry
        WHERE oid in ({','.join(oids_chunk)});
        """
        forced_photometry = pd.read_sql_query(query_forced_photometry, con=engine)
        forced_photometry_path = os.path.join(chunk_dir, "forced_photometry.parquet")
        forced_photometry.to_parquet(forced_photometry_path)

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
        xmatch_path = os.path.join(chunk_dir, "xmatch.parquet")
        xmatch.to_parquet(xmatch_path)
    
    def chunk_list(self,lst, chunk_size):
        """Yield successive chunks of size chunk_size from list."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    
class PreprocessByPKL():
    def __init__(self)-> None:
        from pipeline.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor  
        from pipeline.lc_classifier.lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor  
        
        aos_paths = glob.glob('/home/magdalena/Desktop/magister/SSL_DATASET/ZTF_SSL_Dataset/aos/*')
        print(len(aos_paths))
        self.ft_ex = ZTFFeatureExtractor()
        self.lc_ex = ZTFLightcurvePreprocessor()

        
        chunk_size = 1
        chunks = list(self.chunk_list(aos_paths, chunk_size))
         
        # Use joblib to parallelize the processing
        n_jobs = 8  # Use all available cores
        Parallel(n_jobs=n_jobs)(
            delayed(self.process_chunk)(chunk, self.ft_ex, self.lc_ex)
            for chunk in tqdm(chunks, desc='PreProcessing AstroObject chunks', total=len(chunks))
        )
        
    def process_chunk(self,chunk, ex, lc_ex):
        aos = []
        for path in chunk:
            with open(path, 'rb') as f:
                ao = pickle.load(f)
            #ao.features = ao.features = pd.DataFrame()
            aos.append(ao)
        
        ex.compute_features_batch(aos)
        lc_ex.preprocess_batch(aos)
        
        for ao, path in zip(aos, chunk):
            with open(path, "wb") as f:
                pickle.dump(ao, f)

    def chunk_list(self,lst, chunk_size):
        """Yield successive chunks of size chunk_size from list."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]
     