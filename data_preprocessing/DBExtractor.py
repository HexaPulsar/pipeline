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

import shutil
from tqdm import tqdm

#from pipeline.lc_classifier.lc_classifier.features.composites.core.base import AstroObject
from lc_classifier.lc_classifier.features.core.base import AstroObject
from pipeline.lc_classifier.lc_classifier.utils import create_astro_object 

class DBExtractor():

    def __init__(self) -> None:
    
        data_folder = "/home/mdelafuente/SSL/pulled_data/"
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_folder_full_path = os.path.join(script_path, data_folder)

        if not os.path.exists(data_folder_full_path):
            os.makedirs(data_folder_full_path)

        url = "https://raw.githubusercontent.com/alercebroker/usecases/master/alercereaduser_v4.json"
        params = requests.get(url).json()["params"]

        engine = sa.create_engine(
            f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}"
        )
        engine.begin()
        
        oids = pd.read_parquet("/home/mdelafuente/SSL/2020_oids.parquet")
        oids = oids.index.tolist()

        self.clear_export_directory(data_folder_full_path)
        chunk_size = 1000  
        #count = 0
        for chunk in tqdm(self.chunk_list(oids, chunk_size), 
                        desc=f'Pulling data by oid chunks',
                        total = len(oids)//chunk_size):

            self.process_chunk(chunk, data_folder_full_path, engine)
            pass
    
    def clear_export_directory(self,directory):
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


    def process_chunk(self,chunk, data_folder_full_path, engine):
        # Format oids for SQL query
        oids_chunk = [f"'{oid}'" for oid in chunk]

        # Query for detections
        query_detections = f"""
        SELECT * FROM detection
        WHERE oid in ({','.join(oids_chunk)});
        """
        detections = pd.read_sql_query(query_detections, con=engine)

        # Handle Parquet writing (check for existing file)
        detections_path = os.path.join(data_folder_full_path, "detections.parquet")
        if os.path.exists(detections_path):
            existing_detections = pd.read_parquet(detections_path)
            detections = pd.concat([existing_detections, detections], ignore_index=True)
        detections.to_parquet(detections_path)

        # Query for forced photometry
        query_forced_photometry = f"""
        SELECT * FROM forced_photometry
        WHERE oid in ({','.join(oids_chunk)});
        """
        forced_photometry = pd.read_sql_query(query_forced_photometry, con=engine)

        # Handle Parquet writing for forced photometry
        forced_photometry_path = os.path.join(data_folder_full_path, "forced_photometry.parquet")
        if os.path.exists(forced_photometry_path):
            existing_forced_photometry = pd.read_parquet(forced_photometry_path)
            forced_photometry = pd.concat([existing_forced_photometry, forced_photometry], ignore_index=True)
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

        # Handle Parquet writing for xmatch
        xmatch_path = os.path.join(data_folder_full_path, "xmatch.parquet")
        if os.path.exists(xmatch_path):
            existing_xmatch = pd.read_parquet(xmatch_path)
            xmatch = pd.concat([existing_xmatch, xmatch], ignore_index=True)
        xmatch.to_parquet(xmatch_path)
    
    def chunk_list(self,lst, chunk_size):
        """Yield successive chunks of size chunk_size from list."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]
#----###



class CreateAstroObjectPKL():
    def __init__(self)-> None:
        self.script_path = '/home/magdalena/Desktop/magister/SSL_DATASET/ZTF_SSL_Dataset/batch_processing/features/tests/data/'
        detections = pd.read_parquet(os.path.join(self.script_path, "detections.parquet"))
        forced_photometry = pd.read_parquet(
            os.path.join(self.script_path, "forced_photometry.parquet")
        )
        xmatch = pd.read_parquet(os.path.join(self.script_path, "xmatch.parquet"))
        self.dataframes_to_astro_object_pkl(detections, forced_photometry, xmatch)
        


    def dataframes_to_astro_object_pkl(self, detections, forced_photometry, xmatch):
        oids = detections["oid"].unique()
        print(f"Processing {len(oids)} unique objects")
        
        # Create output directory if it doesn't exist
        output_dir = '/home/magdalena/Desktop/magister/SSL_DATASET/ZTF_SSL_Dataset/aos'
        os.makedirs(output_dir, exist_ok=True)
        
        def save_single(astro_objects: AstroObject, filename: str):
            with open(filename, "wb") as f:
                pickle.dump(astro_objects, f)
                
        def process_oid(oid, detections, forced_photometry, xmatch):
            output_path = os.path.join(output_dir, f'ao_{oid}')
            
            # Skip if file already exists
            if os.path.exists(output_path):
                return
            
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
                save_single(ao, output_path)
            except Exception as e:
                print(f'skipped {oid}: assertion error no xmatch for {oid}')
                print(e)
                return
        
        # Get list of already processed oids
        existing_files = set(f.replace('ao_', '') for f in os.listdir(output_dir) if f.startswith('ao_'))
        
        # Filter out already processed oids
        remaining_oids = [oid for oid in oids if str(oid) not in existing_files]
        print(f"Found {len(existing_files)} existing objects")
        print(f"Remaining objects to process: {len(remaining_oids)}")
        
        ## Process remaining oids
        for oid in tqdm(remaining_oids, desc="Creating astro objects", unit="object"):
            process_oid(oid, detections, forced_photometry, xmatch)
            
            
            
         

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
     