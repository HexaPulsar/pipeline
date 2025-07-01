import numpy as np
from .InitBackbone import InitBackbone

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from .utils import get_confusion_matrix




ztf_order_classes = ['SNIa', # yes
                'SNIbc', # yes
                'SNIIb', # yes
                'SNII', # yes
                'SNIIn', # yes
                'SLSN', # yes
                'TDE', # yes
                'Microlensing', # yes
                'QSO', 
                'AGN', # yes
                'Blazar', 
                'YSO', 
                'CV/Nova', 
                'LPV', 
                'EA', 
                'EB/EW', # yes
                'Periodic-Other', 
                'RSCVn', 
                'CEP', 
                'RRLab', 
                'RRLc', 
                'DSCT']

class ExplorePretraining(InitBackbone):
    def __init__(self, path_to_config_yaml, model, arg_key, device='cpu'):
        super().__init__(path_to_config_yaml, model, arg_key, device)

    def map_to_hierarchy(self,preds,target):
        mapping_dict = {
            0: 1, 1: 1, 3: 1, 5: 1, 8: 1,
            2: 2, 6: 2, 7: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2,
            4: 0, 9: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0
        }
        map_preds = np.array([mapping_dict.get(int(label), -1) for label in preds], device = preds.device)
        map_target = np.array([mapping_dict.get(int(label), -1) for label in target], device = target.device)
        return map_preds, map_target
    
    def compute_logistic_regressor(self,X_train, y_train, X_val, y_val,taxonomy, model_args = {},weights= {} ):

        std_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # z = (x - mean) / std
            ('model', LogisticRegression(random_state=0, max_iter = 1000, multi_class = 'ovr', class_weight=weights))
        ])

        # You can change n_neighbors as needed
        std_pipeline.fit(X_train,y_train)
        train_y_pred = std_pipeline.predict(X_train)
        val_y_pred = std_pipeline.predict(X_val)
        return train_y_pred,val_y_pred
    
    def compute_knn(self,X_train, y_train, X_val, y_val,taxonomy, model_args = {}, ):
        knn_pipeline = Pipeline([
            
            ('scaler', StandardScaler()),  # z = (x - mean) / std
            #('pca', PCA()),
            ('model', KNeighborsClassifier(**model_args))
        ])
        knn_pipeline.fit(X_train, y_train)
        train_y_pred = knn_pipeline.predict(X_train)
        val_y_pred = knn_pipeline.predict(X_val)
        return train_y_pred,val_y_pred
    def get_confusion_matrix(self, preds,target, taxonomy, dataset_type:str, plot_title = 'logit reg'):
        return get_confusion_matrix(preds,target,taxonomy, dataset_type,plot_title, order_classes=ztf_order_classes)
   


