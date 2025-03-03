from tqdm import tqdm
import sqlalchemy as sa
import requests
import pandas as pd
import os

import shutil


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


def process_chunk(chunk, data_folder_full_path, engine):
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
 
def chunk_list(lst, chunk_size):
    """Yield successive chunks of size chunk_size from list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]




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

clear_export_directory(data_folder_full_path)
chunk_size = 1000  
#count = 0
for chunk in tqdm(chunk_list(oids, chunk_size), 
                  desc=f'Pulling data by oid chunks',
                  total = len(oids)//chunk_size):

    process_chunk(chunk, data_folder_full_path, engine)
    