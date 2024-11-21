import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from joblib import Parallel, delayed
import pandas as pd

from mbappe_mapper_batch import MbappeMapperBatchProcessing
from alerce_classifiers.mbappe.model import MbappeClassifier


def evaluate_mbappe(input_aos_filename, output_file):
    from lc_classifier.features.core.base import astro_object_from_dict

    model = MbappeClassifier(
        model_path=os.getenv("TEST_MBAPPE_MODEL_PATH"),
        metadata_quantiles_path=os.getenv("TEST_MBAPPE_METADATA_QUANTILES_PATH"),
        features_quantiles_path=os.getenv("TEST_MBAPPE_FEATURES_QUANTILES_PATH"),
        mapper=MbappeMapperBatchProcessing(),
    )
    batch_astro_object_pickles = pd.read_pickle(input_aos_filename)
    batch_astro_objects = [
        astro_object_from_dict(d) for d in batch_astro_object_pickles
    ]
    predicted_probs = model.predict(batch_astro_objects)
    predicted_probs.to_parquet(output_file)


def evaluate_mbappe_many_files(aos_input_dir: str, output_dir: str, n_jobs: int):
    assert os.path.exists(aos_input_dir) and os.path.exists(output_dir)
    assert n_jobs >= 1
    model = MbappeClassifier(
        model_path=os.getenv("TEST_MBAPPE_MODEL_PATH"),
        metadata_quantiles_path=os.getenv("TEST_MBAPPE_METADATA_QUANTILES_PATH"),
        features_quantiles_path=os.getenv("TEST_MBAPPE_FEATURES_QUANTILES_PATH"),
        mapper=MbappeMapperBatchProcessing(),
    )
    aos_filenames = os.listdir(aos_input_dir)

    tasks = []
    for aos_filename in aos_filenames:
        output_filename = ".".join(aos_filename.split(".")[:-1])
        output_filename += ".parquet"
        output_filename = os.path.join(output_dir, output_filename)

        tasks.append(
            delayed(evaluate_mbappe)(
                os.path.join(aos_input_dir, aos_filename),
                output_filename,
            )
        )

    Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(tasks)
