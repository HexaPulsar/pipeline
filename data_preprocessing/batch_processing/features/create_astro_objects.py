import pickle

import pandas as pd

from lc_classifier.features.core.base import AstroObject
from lc_classifier.utils import create_astro_object
from typing import List


class NoDetections(Exception):
    pass


def dataframes_to_astro_object_list(detections, forced_photometry, xmatch):
    oids = detections["oid"].unique()
    astro_objects_list = []
    for oid in oids:
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
        astro_objects_list.append(ao)
    return astro_objects_list


def xmatch_df_to_astro_object_list(xmatch_dataframe: pd.DataFrame) -> List[AstroObject]:
    return None
