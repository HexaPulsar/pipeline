import pandas as pd
import os
from ..create_astro_objects import dataframes_to_astro_object_list


script_path = os.path.dirname(os.path.abspath(__file__))


def test_check_input():
    detections = pd.read_parquet(os.path.join(script_path, "data/detections.parquet"))
    forced_photometry = pd.read_parquet(
        os.path.join(script_path, "data/forced_photometry.parquet")
    )
    xmatch = pd.read_parquet(os.path.join(script_path, "data/xmatch.parquet"))
    aos = dataframes_to_astro_object_list(detections, forced_photometry, xmatch)
    print(len(aos), "astro objects")
