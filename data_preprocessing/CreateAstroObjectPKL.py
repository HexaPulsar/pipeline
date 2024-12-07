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

import shutil
from tqdm import tqdm

from pipeline.lc_classifier.lc_classifier.utils import create_astro_object 

class CreateAstroObjectPKL():
    def __init__(self, chunks_dir: str,output_dir:str,n_jobs:int = 1):
        
        # List all chunk directories
        chunk_dirs = [d for d in os.listdir(chunks_dir) if d.startswith('chunk_')]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process chunks in parallel
        Parallel(n_jobs=n_jobs)(
            delayed(self.process_single_chunk)(chunk_dir, chunks_dir, output_dir)
            for chunk_dir in tqdm(chunk_dirs, desc = 'Creating AstroObjects', total = len(chunk_dirs),unit='chunks')
        )
    
    def process_single_chunk(self,chunk_dir, chunks_dir, output_dir):
        """
        Process a single chunk of data by reading parquet files and converting to astro objects
        
        Parameters:
        chunk_dir (str): Directory name for the specific chunk
        chunks_dir (str): Base directory containing all chunks
        output_dir (str): Directory to save processed results
        """
        chunk_path = os.path.join(chunks_dir, chunk_dir)
        
        # Read parquet files for this chunk
        detections = pd.read_parquet(os.path.join(chunk_path, "detections.parquet"))
        forced_photometry = pd.read_parquet(os.path.join(chunk_path, "forced_photometry.parquet"))
        xmatch = pd.read_parquet(os.path.join(chunk_path, "xmatch.parquet"))
        
        # Create chunk-specific output directory
        chunk_output_dir = os.path.join(output_dir, chunk_dir)
        os.makedirs(chunk_output_dir, exist_ok=True)
        
        # Process the dataframes
        self.dataframes_to_astro_object_pkl(detections, forced_photometry, xmatch, chunk_output_dir)
        
        return chunk_dir  # Return chunk_dir to track completion


    def dataframes_to_astro_object_pkl(self, detections, forced_photometry, xmatch,output_dir):
        oids = detections["oid"].unique()

        @staticmethod
        def save_single(astro_objects: AstroObject, filename: str):
            with open(filename, "wb") as f:
                pickle.dump(astro_objects, f,protocol=pickle.HIGHEST_PROTOCOL)
        @staticmethod
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
                #print(f'Skipped {oid}: assertion error no xmatch for {oid}')
                
                return

        # Process remaining oids
        for oid in oids:
            process_oid(oid, detections, forced_photometry, xmatch)
         