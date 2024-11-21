import os
import unittest
from shutil import rmtree
from ..evaluate_mbappe_model import evaluate_mbappe_many_files


script_path = os.path.dirname(os.path.abspath(__file__))


class TestDistributedMbappe(unittest.TestCase):
    def setUp(self):
        self.output_directory = "/tmp/test_distributed_mbappe"
        if os.path.exists(self.output_directory):
            rmtree(self.output_directory)
        os.makedirs(self.output_directory)

        self.aos_w_features_directory = os.path.join(script_path, "data/aos_w_features")

    def tearDown(self):
        rmtree(self.output_directory)

    def test_feature_computation(self):
        # Compute probabilities
        evaluate_mbappe_many_files(
            self.aos_w_features_directory, self.output_directory, n_jobs=2
        )
        assert len(os.listdir(self.output_directory)) == len(
            os.listdir(self.aos_w_features_directory)
        )
