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
from tqdm import tqdm

from pipeline.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from pipeline.lc_classifier.lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
###WARNING SUPRESSION
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")
warnings.filterwarnings("ignore", category=np.RankWarning)
####



import jax
jax.config.update('jax_platform_name', 'cpu')
os.environ['JAX_PLATFORMS'] = 'cpu'
# Load OIDs and other configurations
oids = pd.read_parquet('/home/magdalena/pipeline/data_preprocessing/sixplusdets/2020_plussixdet_oids_.parquet')
oids_list = oids.index.tolist()

url = "https://raw.githubusercontent.com/alercebroker/usecases/master/alercereaduser_v4.json"
params = requests.get(url).json()["params"]

# Pass the database connection parameters
db_params = {
    'user': params['user'],
    'password': params['password'],
    'host': params['host'],
    'dbname': params['dbname']
}

# Load YAML config
with open("/home/magdalena/pipeline/h5file/dict_info.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Helper function to extract, process, and save data for a chunk of OIDs
def process_chunk(oids_chunk, db_params, config,ft_ex,lc_ex, out_dir):
    # Recreate the engine inside the worker process
    engine = sa.create_engine(
        f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['dbname']}"
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

# Define the chunk size (you can adjust this based on your system's capabilities)
chunk_size = 50
n_jobs =10
out_dir = '/home/magdalena/pipeline/data_preprocessing/sixplusdets/2020_out'

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
# Split the OIDs list into chunks
chunks = list(chunkify(oids_list, chunk_size))
ft_exs = [ZTFFeatureExtractor() for i in range(n_jobs)]
lc_exs =[ZTFLightcurvePreprocessor() for i in range(n_jobs)]

# Run the processing in parallel for each chunk of OIDs
Parallel(n_jobs=n_jobs)(  # Use -1 to utilize all available CPU cores
    delayed(process_chunk)(chunk, db_params, config,ft_exs[i%n_jobs],lc_exs[i%n_jobs], out_dir ) 
    for i,chunk in tqdm(enumerate(chunks), desc = f'Processing chunks of size {chunk_size}', total = len(chunks))
)
