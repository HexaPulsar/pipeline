import os
import unittest
from shutil import rmtree
from ..compute_features import extract_features_many_files


script_path = os.path.dirname(os.path.abspath(__file__))


class TestDistributedFeatureComputation(unittest.TestCase):
    def setUp(self):
        self.output_directory = "/tmp/test_feature_computation_many_files"
        if os.path.exists(self.output_directory):
            rmtree(self.output_directory)
        os.makedirs(self.output_directory)

        self.aos_directory = os.path.join(script_path, "data/aos")

    def tearDown(self):
        rmtree(self.output_directory)

    def test_feature_computation(self):
        # Compute features
        extract_features_many_files(self.aos_directory, self.output_directory, n_jobs=2)
        assert len(os.listdir(self.output_directory)) == len(
            os.listdir(self.aos_directory)
        )

        # TODO: save in a format that's compatible with the DB
        assert False
