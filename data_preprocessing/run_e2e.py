import pandas as pd
import requests
import sqlalchemy as sa
import os
import pickle
import yaml
from joblib import Parallel, delayed
import warnings
from scipy.optimize import OptimizeWarning
#from tqdm import tqdm
from tqdm.asyncio import trange, tqdm
import numpy as np   
from src.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from src.lc_classifier.lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
from src.utils.end2end import extract_from_db,create_astro_objects,transform2arrays 
import sys
import os 
###WARNING SUPRESSION
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")
warnings.filterwarnings("ignore", category=np.RankWarning)
####

 
# Helper function to extract, process, and save data for a chunk of OIDs
def process_chunk(oids_chunk, DB_PARAMs, config,ft_ex,lc_ex, out_dir):
    # Recreate the engine inside the worker process
    engine = sa.create_engine(
        f"postgresql+psycopg2://{DB_PARAMs['user']}:{DB_PARAMs['password']}@{DB_PARAMs['host']}/{DB_PARAMs['dbname']}"
    )
    
    # Extract data from the database for the chunk of OIDs
    detections, forced_photometry, xmatch = extract_from_db(oids_chunk, engine)
    # Create astro objects from the extracted data
    aos_list = create_astro_objects(detections, forced_photometry, xmatch)
    if aos_list is None:
        #print(f"Warning: create_astro_objects returned None for OID chunk: {oids_chunk}")
        return  # Skip processing this chunk
    # Transform to array format using the provided config
    dict_array_list = transform2arrays(aos_list=aos_list, config_dict=config,ft_ex = ft_ex,lc_ex = lc_ex)
    
    # Save the result to a pickle file for each OID in the chunk
    for dict_array in dict_array_list:
        out_dir_temp = f'{out_dir}/array_{dict_array["oid"]}.pkl'
        # Write the pickle file
        with open(out_dir_temp, "wb") as f:
            pickle.dump(dict_array, f,protocol=pickle.HIGHEST_PROTOCOL)

# Helper function to split the list into chunks
def chunkify(lst, chunk_size):
    """Split list into smaller chunks."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


######### RUNTIME##############################################

import jax
#CONFIGS
jax.config.update('jax_platform_name', 'cpu')
os.environ['JAX_PLATFORMS'] = 'cpu'


#GLOBAL VARS
CHUNK_SIZE = 100
N_JOBS =10 
OUT_PATH = '/home/magdalena/pipeline/data_preprocessing/data/AO_PKL/2023_out/'
YAML_PATH = '/home/magdalena/pipeline/data_preprocessing/src/utils/config_dict/dict_info.yaml'
NDET_DF = pd.read_parquet('/home/magdalena/pipeline/data_preprocessing/data/OID_PARQUETS/2023_plussixdet_oids_.parquet')
OIDS_LIST = NDET_DF.index.tolist()
 
URL = "https://raw.githubusercontent.com/alercebroker/usecases/master/alercereaduser_v4.json"
PARAMS = requests.get(URL).json()["params"]
DB_PARAMs = {
    'user': PARAMS['user'],
    'password': PARAMS['password'],
    'host': PARAMS['host'],
    'dbname': PARAMS['dbname']
}

# Load YAML config
with open(YAML_PATH, 'r') as stream:
    config = yaml.safe_load(stream)
 
# Define the chunk size (you can adjust this based on your system's capabilities)

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
# Split the OIDs list into chunks
chunks = list(chunkify(OIDS_LIST, CHUNK_SIZE))
ft_exs = [ZTFFeatureExtractor() for i in range(N_JOBS)]
lc_exs =[ZTFLightcurvePreprocessor() for i in range(N_JOBS)]

# Run the processing in parallel for each chunk of OIDs
Parallel(n_jobs=N_JOBS)(  # Use -1 to utilize all available CPU cores
    delayed(process_chunk)(chunk, DB_PARAMs, config,ft_exs[i%N_JOBS],lc_exs[i%N_JOBS], OUT_PATH ) 
    for i,chunk in tqdm(enumerate(chunks), desc = f'Processing chunks of size {CHUNK_SIZE}', total = len(chunks),miniters=1,dynamic_ncols=True)
)
