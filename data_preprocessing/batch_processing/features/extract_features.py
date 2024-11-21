import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed
import pandas as pd
import sys
from tqdm import tqdm
from lc_classifier.utils import all_features_from_astro_objects
from lc_classifier.features.core.base import astro_object_from_dict


def get_shorten(filename: str):
    possible_n_days = filename.split("_")[-2]
    return possible_n_days


def extract_features_from_astro_objects(aos_filename: str, features_filename: str):
    shorten = get_shorten(aos_filename)
    astro_objects_batch = pd.read_pickle(aos_filename)
    astro_objects_batch = [astro_object_from_dict(ao) for ao in astro_objects_batch]
    features_batch = all_features_from_astro_objects(astro_objects_batch)
    features_batch["shorten"] = shorten
    features_batch.to_parquet(features_filename)


if __name__ == "__main__":
    dir_name = sys.argv[1]
    file_list = os.listdir(dir_name)
    output_dir = os.path.join(dir_name, 'features')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = [filename for filename in file_list if "astro_objects_batch" in filename]
    file_list = sorted(file_list)

    tasks = []
    for batch_filename in tqdm(file_list):
        full_filename = os.path.join(dir_name, batch_filename)
        output_filename = os.path.join(output_dir, 'features_' + batch_filename.replace('.pkl', '.parquet'))
        tasks.append(
            delayed(extract_features_from_astro_objects)(full_filename, output_filename)
        )

    Parallel(n_jobs=230, verbose=11, backend="loky")(tasks)
