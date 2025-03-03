import pandas as pd
import requests
import sqlalchemy as sa
import os
import pickle
import yaml
from joblib import Parallel, delayed
from .end2end import *


import warnings
from scipy.optimize import OptimizeWarning
from tqdm import tqdm

class AstroObjectMiner:
    def __init__(self, db_params:dict):
        """Extracts detections, forced_photometry, xmatch from alerce DB and creates astroobjects for each OID.
        Args:
            oid_list (list): a list of OIDs in the alerce database
            db_params (dict): database access params as a dictionary: user,password,host,dbname
        """
        pass
    def initialize_directories(self):
        pass
    def initialize_database_conn(self,db_params):
        self.engine = sa.create_engine(
        f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['dbname']}"
    )
    def __call__(self, oids_chunk):
        detections, forced_photometry, xmatch = self.extract_from_db(oids_chunk, self.engine)
        aos_list = self.create_astro_objects(detections, forced_photometry, xmatch)
        #missing save functinoality
        return aos_list

    def extract_from_db(self,oids_list,engine):
        """function provided by alerce (batch_processing repo)"""
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

    def create_astro_objects(self,detections,forced_photometry,xmatch):
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