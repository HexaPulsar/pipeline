import glob
import pickle
import jax
from pipeline.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from pipeline.lc_classifier.lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor  
from tqdm import tqdm
import warnings

from scipy.optimize import OptimizeWarning
print('SUPRESSING WARNINGS') 
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")
warnings.filterwarnings("ignore", category=np.RankWar1ning)

def chunk_list(lst, chunk_size):
    """Yield successive chunks of size chunk_size from list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


aos_paths = glob.glob('/home/mdelafuente/batch_processing/aos/*')#[3:]

ex = ZTFFeatureExtractor()
lc_ex = ZTFLightcurvePreprocessor()

print(len(aos_paths))
chunk_size = 1
chunk_list = chunk_list(aos_paths, chunk_size)


for chunk in tqdm(chunk_list, desc = 'Processing by chunks', total = len(aos_paths)//chunk_size):
    aos = []
    for path in chunk:
        with open(f'{path}', 'rb') as f:
            ao = pickle.load(f)
        #assert ao.features.empty
        ao.features = ao.features.iloc[0:0]
        aos.append(ao)
    print(aos[0].detections.shape[0])
    #ex.compute_features_batch(aos)
    lc_ex.preprocess_batch(aos) 
    #print(aos[0].features)
    for ao,path in zip(aos,chunk):
        with open(f'{path}', "wb") as f:
            pickle.dump(ao, f)
    
 
    