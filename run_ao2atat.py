import pandas as pd
import requests
import sqlalchemy as sa
import os
import pickle
import yaml
from joblib import Parallel, delayed
from end2end import *
import warnings
from scipy.optimize import OptimizeWarning
import glob
from tqdm import tqdm
###WARNING SUPRESSION
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")
warnings.filterwarnings("ignore", category=np.RankWarning)
####

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
jax.config.update('jax_platform_name', 'cpu')
os.environ['JAX_PLATFORMS'] = 'cpu'

# Load YAML config
with open("/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/dict_info.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Helper function to extract, process, and save data for a chunk of OIDs
def process_chunk(ao_path,  config,ft_ex,lc_ex):

    with open(f'{ao_path}', 'rb') as f:
        ao = pickle.load(f)

    # Transform to array format using the provided config
    dict_array = transform2arrays([ao], config,ft_ex, lc_ex)[0]

    out_dir = f'/home/mdelafuente/SSL/out_directory/array_{dict_array["oid"]}.pkl'
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    
    # Write the pickle file
    with open(out_dir, "wb") as f:
        pickle.dump(dict_array, f,protocol=pickle.HIGHEST_PROTOCOL)

# Helper function to split the list into chunks
def chunkify(lst, chunk_size):
    """Split list into smaller chunks."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

n_jobs =10
directories = glob.glob('/home/mdelafuente/SSL/aos/*/*',recursive=True)  

# Sort files by size
sorted_files = sorted(directories, key=os.path.getsize)

# Split the OIDs list into chunks
ft_exs = [ZTFFeatureExtractor() for i in range(n_jobs)]
lc_exs =[ ZTFLightcurvePreprocessor() for i in range(n_jobs)]
Parallel(n_jobs=n_jobs)(  # Use -1 to utilize all available CPU cores
    delayed(process_chunk)(directory, config,ft_exs[i % n_jobs], lc_exs[i%n_jobs]) 
    for i,directory in tqdm(enumerate(directories), desc = 'Processing aos', total =  len(directories), unit = 'aos2atat')
)