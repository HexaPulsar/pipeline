# %% [markdown]
# #P4J library must be compiled and installed
# #mexican hatt too
# 
# -march=x86-64
# python setup.py build_ext --inplace
# pip install .

# %%
import numbers
import sys
import os  
from pipeline.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor


print(sys.path)
from features.tests.test_create_astro_objects import test_check_input


# %%
from pipeline.lc_classifier.lc_classifier.features.preprocess.ztf  import (
    ZTFLightcurvePreprocessor,
    ShortenPreprocessor,
)
from pipeline.lc_classifier.lc_classifier.features.composites.ztf import ZTFFeatureExtractor  


from tqdm import tqdm
import pandas as pd

from pipeline.lc_classifier.lc_classifier.features.composites.core.base import astro_object_from_dict
from pipeline.lc_classifier.lc_classifier.utils import all_features_from_astro_objects

import pickle
from typing import List
from pipeline.lc_classifier.lc_classifier.features.composites.core.base import AstroObject
from pipeline.lc_classifier.lc_classifier.utils import create_astro_object
from tqdm.auto import tqdm
from joblib import Parallel, delayed

import sys
from collections import abc
import types

def get_size(obj, seen=None):
    """Recursively calculate size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, (str, bytes, numbers.Number, range, bytearray)):
        pass
    elif isinstance(obj, (tuple, list, set, frozenset)):
        size += sum(get_size(item, seen) for item in obj)
    elif isinstance(obj, abc.Mapping):
        size += sum(get_size(key, seen) + get_size(value, seen) for key, value in obj.items())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__slots__'):
        size += sum(get_size(getattr(obj, slot), seen) for slot in obj.__slots__ if hasattr(obj, slot))
    return size

def print_size(obj):
    """Print the size of an object in bytes and KB"""
    size_bytes = get_size(obj)
    size_kb = size_bytes / 1024
    print(f"Size: {size_bytes} bytes ({size_kb:.2f} KB)")


def save_single(astro_objects:AstroObject, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(astro_objects, f)
            
def process_oid(oid, detections, forced_photometry, xmatch):
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
        

        save_single(ao,f'/home/mdelafuente/batch_processing/aos/ao_{oid}')
    except:
        print(f'skipped {oid}: assertion error')
        return
    

def dataframes_to_astro_object_list(detections, forced_photometry, xmatch, n_jobs=1):
    oids = detections["oid"].unique()
    print(f"Processing {len(oids)} unique objects")
    
    astro_objects_list = Parallel(n_jobs=n_jobs)(
        delayed(process_oid)(oid, detections, forced_photometry, xmatch) 
        for oid in tqdm(oids, desc="Processing objects", unit="object")
    )
    
    return astro_objects_list


def dataframes_to_astro_object_pkl(detections, forced_photometry, xmatch, n_jobs=1):
    oids = detections["oid"].unique()
    print(f"Processing {len(oids)} unique objects")
    
    astro_objects_list = Parallel(n_jobs=n_jobs)(
        delayed(process_oid)(oid, detections, forced_photometry, xmatch) 
        for oid in tqdm(oids, desc="Creating astro objects", unit="object")
    )

def get_shorten(filename: str):
    possible_n_days = filename.split("_")[-2]
    return possible_n_days

def extract_features_from_astro_objects(aos_filename: str, features_filename: str):
    shorten = get_shorten(aos_filename)
    astro_objects_batch = pd.read_pickle(aos_filename)
    print(astro_objects_batch)
    #astro_objects_batch = [astro_object_from_dict(ao) for ao in astro_objects_batch]
    features_batch = all_features_from_astro_objects(astro_objects_batch)
    features_batch["shorten"] = shorten
    features_batch.to_parquet(features_filename)



script_path = os.path.dirname(os.path.abspath(__file__))

script_path = '/home/mdelafuente/batch_processing/features/tests'


def test_check_input():
    detections = pd.read_parquet(os.path.join(script_path, "data/detections.parquet"))
    forced_photometry = pd.read_parquet(
        os.path.join(script_path, "data/forced_photometry.parquet")
    )
    xmatch = pd.read_parquet(os.path.join(script_path, "data/xmatch.parquet"))
    aos = dataframes_to_astro_object_pkl(detections, forced_photometry, xmatch)
    #print(len(aos), "astro objects")
    return aos


aos = test_check_input()

exit()
with open('/home/mdelafuente/batch_processing/test.pkl', 'rb') as f:
    aos = pickle.load(f)

# %%


ex = ZTFFeatureExtractor()

# %%
import jax
jax.devices()


# %%
ex.compute_features_batch(aos)
#re-execute when no background is running 


# %%
for ao in aos:
    print(ao.features)

# %%
lc_ex = ZTFLightcurvePreprocessor()

# %%
lc_ex.preprocess_batch(aos)

# %%
save_batch(aos,'./test.pkl')

# %%

extract_features_from_astro_objects('/home/mdelafuente/batch_processing/test.pkl','./test_features.parquet')

# %%
# Open the file in binary read mode ('rb') and load the content
with open('/home/mdelafuente/batch_processing/test.pkl', 'rb') as f:
    data = pickle.load(f)

# Now, 'data' contains the loaded object
print(data[0].features)


