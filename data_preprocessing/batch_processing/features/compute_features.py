import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed
from typing import List
from lc_classifier.features.core.base import AstroObject
import pickle


def save_batch(astro_objects: List[AstroObject], filename: str):
    astro_objects_dicts = [ao.to_dict() for ao in astro_objects]
    with open(filename, "wb") as f:
        pickle.dump(astro_objects_dicts, f)


def extract_features_many_files(aos_input_dir: str, output_dir: str, n_jobs: int):
    assert os.path.exists(aos_input_dir) and os.path.exists(output_dir)
    assert n_jobs >= 1
    aos_filenames = os.listdir(aos_input_dir)

    tasks = []
    for ao_filename in aos_filenames:
        tasks.append(
            delayed(extract_features)(
                os.path.join(aos_input_dir, ao_filename),
                os.path.join(output_dir, ao_filename),
            )
        )

    Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(tasks)


def extract_features(input_filename, output_filename, skip_if_output_exists=False):
    import pandas as pd
    from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
    from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
    from lc_classifier.features.core.base import astro_object_from_dict

    if skip_if_output_exists and os.path.exists(output_filename):
        return

    batch_astro_object_pickles = pd.read_pickle(input_filename)
    batch_astro_objects = [
        astro_object_from_dict(d) for d in batch_astro_object_pickles
    ]

    lightcurve_preprocessor = ZTFLightcurvePreprocessor()
    lightcurve_preprocessor.preprocess_batch(batch_astro_objects)

    feature_extractor = ZTFFeatureExtractor()
    feature_extractor.compute_features_batch(batch_astro_objects, progress_bar=False)

    save_batch(batch_astro_objects, output_filename)
