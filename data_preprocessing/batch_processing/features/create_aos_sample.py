import os
import pandas as pd
from create_astro_objects import dataframes_to_astro_object_list
from lc_classifier.features.core.base import save_astro_objects_batch
from compute_features import extract_features_many_files


aos_folder = "data/aos"
test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
aos_folder_full_path = os.path.join(test_path, aos_folder)
del aos_folder

if not os.path.exists(aos_folder_full_path):
    os.makedirs(aos_folder_full_path)

# Load data
detections = pd.read_parquet(os.path.join(test_path, "data/detections.parquet"))

forced_photometry = pd.read_parquet(
    os.path.join(test_path, "data/forced_photometry.parquet")
)

xmatch = pd.read_parquet(os.path.join(test_path, "data/xmatch.parquet"))

aos = dataframes_to_astro_object_list(detections, forced_photometry, xmatch)

assert len(aos) == 5

save_astro_objects_batch(aos[:2], os.path.join(aos_folder_full_path, "aos_0.pkl"))
save_astro_objects_batch(aos[2:], os.path.join(aos_folder_full_path, "aos_1.pkl"))

# Compute features and save aos_w_features
aos_w_features_folder = "data/aos_w_features"
aos_w_features_folder_full_path = os.path.join(test_path, aos_w_features_folder)
del aos_w_features_folder

if not os.path.exists(aos_w_features_folder_full_path):
    os.makedirs(aos_w_features_folder_full_path)

extract_features_many_files(
    aos_folder_full_path, aos_w_features_folder_full_path, n_jobs=2
)
