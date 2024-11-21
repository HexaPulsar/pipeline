import os

from alerce_classifiers.base.dto import OutputDTO

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from typing import List
from lc_classifier.features.core.base import AstroObject
from lc_classifier.utils import all_features_from_astro_objects

from joblib import Parallel, delayed
import pandas as pd

from alerce_classifiers.squidward.model import SquidwardFeaturesClassifier
from alerce_classifiers.base.mapper import Mapper


class SquidwardAstroObjectMapper(Mapper):
    def postprocess(self, model_output, **kwargs) -> OutputDTO:
        return model_output

    def preprocess(self, input_aos: List[AstroObject], **kwargs) -> pd.DataFrame:
        features = all_features_from_astro_objects(input_aos)
        return features


def evaluate_squidward(input_aos_filename, output_file):
    from lc_classifier.features.core.base import astro_object_from_dict

    model = SquidwardFeaturesClassifier(
        model_path=os.getenv("TEST_SQUIDWARD_MODEL_PATH"),
        mapper=SquidwardAstroObjectMapper(),
    )
    batch_astro_object_pickles = pd.read_pickle(input_aos_filename)
    batch_astro_objects = [
        astro_object_from_dict(d) for d in batch_astro_object_pickles
    ]
    predicted_probs = model.predict(input_dto=batch_astro_objects)
    predicted_probs.to_parquet(output_file)


def evaluate_squidward_many_files(aos_input_dir: str, output_dir: str, n_jobs: int):
    assert os.path.exists(aos_input_dir) and os.path.exists(output_dir)
    assert n_jobs >= 1
    model = SquidwardFeaturesClassifier(
        model_path=os.getenv("TEST_SQUIDWARD_MODEL_PATH"),
        mapper=SquidwardAstroObjectMapper(),
    )
    aos_filenames = os.listdir(aos_input_dir)

    tasks = []
    for aos_filename in aos_filenames:
        output_filename = ".".join(aos_filename.split(".")[:-1])
        output_filename += ".parquet"
        output_filename = os.path.join(output_dir, output_filename)

        tasks.append(
            delayed(evaluate_squidward)(
                os.path.join(aos_input_dir, aos_filename),
                output_filename,
            )
        )

    Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(tasks)
