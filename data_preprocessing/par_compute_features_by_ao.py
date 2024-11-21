import glob
import pickle
import jax
from joblib import Parallel, delayed
from pipeline.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from pipeline.lc_classifier.lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
from tqdm import tqdm
#print('SUPRESSING WARNINGS') 
#warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")
#warnings.filterwarnings("ignore", category=np.RankWarning)
def process_chunk(chunk, ex, lc_ex):
    aos = []
    for path in chunk:
        with open(path, 'rb') as f:
            ao = pickle.load(f)
        ao.features = ao.features.iloc[0:0]
        aos.append(ao)
    
    #ex.compute_features_batch(aos)
    lc_ex.preprocess_batch(aos)
    
    for ao, path in zip(aos, chunk):
        with open(path, "wb") as f:
            pickle.dump(ao, f)

def chunk_list(lst, chunk_size):
    """Yield successive chunks of size chunk_size from list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

if __name__ == "__main__":
    aos_paths = glob.glob('/home/mdelafuente/SSL/ao_2020/*')

    ex = ZTFFeatureExtractor()
    lc_ex = ZTFLightcurvePreprocessor()

     
    chunk_size = 1000
    chunks = list(chunk_list(aos_paths, chunk_size))

    # Use joblib to parallelize the processing
    n_jobs = 4  
    Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk, ex, lc_ex)
        for chunk in tqdm(chunks, desc='Processing chunks', total=len(chunks))
    )
    