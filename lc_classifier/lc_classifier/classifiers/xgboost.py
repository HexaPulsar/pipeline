from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import os
import pickle

import xgboost as xgb

from base import AstroObject
from .base import all_features_from_astro_objects
from .base import Classifier, NotTrainedException
from .preprocess import RandomForestPreprocessor


class XGBoostClassifier(Classifier):
    version = '1.0.0'

    def __init__(self, list_of_classes: List[str]):
        self.list_of_classes = list_of_classes
        self.model = None
        self.feature_list = None
        self.preprocessor = RandomForestPreprocessor()

    def classify_batch(
            self,
            astro_objects: List[AstroObject],
            return_dataframe: bool = False) -> Optional[pd.DataFrame]:

        if self.feature_list is None or self.model is None:
            raise NotTrainedException(
                'This classifier is not trained or has not been loaded')
        features_df = all_features_from_astro_objects(astro_objects)
        features_df = features_df[self.feature_list]
        features_np = self.preprocessor.preprocess_features(
            features_df).values
        probs_np = self.model.predict_proba(features_np)
        for object_probs, astro_object in zip(probs_np, astro_objects):
            data = np.stack(
                [
                    self.list_of_classes,
                    object_probs.flatten()
                ],
                axis=-1
            )
            object_probs_df = pd.DataFrame(
                data=data,
                columns=[['name', 'value']]
            )
            object_probs_df['fid'] = None
            object_probs_df['sid'] = 'ztf'
            object_probs_df['version'] = self.version
            astro_object.predictions = object_probs_df

        if return_dataframe:
            dataframe = pd.DataFrame(
                data=probs_np,
                columns=self.list_of_classes,
                index=features_df.index
            )
            return dataframe

    def classify_batch_from_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_list is None or self.model is None:
            raise NotTrainedException(
                'This classifier is not trained or has not been loaded')
        features_df = features_df[self.feature_list]
        features_np = self.preprocessor.preprocess_features(
            features_df).values
        probs_np = self.model.predict_proba(features_np)

        dataframe = pd.DataFrame(
            data=probs_np,
            columns=self.list_of_classes,
            index=features_df.index
        )
        return dataframe

    def classify_single_object(self, astro_object: AstroObject) -> None:
        self.classify_batch([astro_object])

    def fit(
            self,
            astro_objects: List[AstroObject],
            labels: pd.DataFrame,
            config: Dict):

        assert len(astro_objects) == len(labels)
        all_features_df = all_features_from_astro_objects(astro_objects)
        self.fit_from_features(all_features_df, labels, config)

    def fit_from_features(
            self,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            config: Dict):

        self.feature_list = features.columns.values
        labels = labels.copy()
        labels['astro_class_num'] = labels['astro_class'].map(
            dict(zip(self.list_of_classes, range(len(self.list_of_classes)))))
        training_labels = labels[labels['partition'] == 'training_0']
        training_features = features.loc[training_labels['aid'].values]
        training_features = self.preprocessor.preprocess_features(
            training_features)

        validation_labels = labels[labels['partition'] == 'validation_0']
        validation_features = features.loc[validation_labels['aid'].values]
        validation_features = self.preprocessor.preprocess_features(
            validation_features)

        self.model = xgb.XGBClassifier(
            tree_method='hist',
            early_stopping_rounds=3
        )

        self.model.fit(
            training_features.values,
            training_labels['astro_class_num'].values,
            eval_set=[(validation_features.values, validation_labels['astro_class_num'].values)]
        )

    def save_classifier(self, directory: str):
        if self.model is None:
            raise NotTrainedException(
                'Cannot save model that has not been trained')

        if not os.path.exists(directory):
            os.mkdir(directory)

        with open(
                os.path.join(
                    directory,
                    'xgboost.pkl'),
                'wb') as f:
            pickle.dump(
                {
                    'feature_list': self.feature_list,
                    'list_of_classes': self.list_of_classes,
                    'model': self.model
                },
                f
            )

    def load_classifier(self, directory: str):
        loaded_data = pd.read_pickle(os.path.join(directory, 'xgboost.pkl'))
        self.feature_list = loaded_data['feature_list']
        self.list_of_classes = loaded_data['list_of_classes']
        self.model = loaded_data['model']