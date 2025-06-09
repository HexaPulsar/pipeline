import numpy as np
import pandas as pd
from tqdm import tqdm 
import glob
from copy import deepcopy
import torch
from sklearn.neighbors import KNeighborsClassifier

from src.layers.classifiers.MultimodalClassifier import MultimodalClassifier
from src.data.modules.LitData import LitData
from src.layers.ClassifierBaseModel import ClassifierBaseModel
from src.layers.classifiers import TokenClassifier
from torch import device, load
from collections import OrderedDict
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class QuickLoader: 
    def __init__(self,
                 batch_size = 256,
                 experiment_type =  "LC",
                data_root =  "/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/dataset.h5",
                seed =  0,
                train_apply_transform =  False,
                validation_apply_transform =  False,
                transforms =  [] , # List of transform modules
                train_key =  'training',
                validation_key =  'validation',
                test_key =  'test',
                observation_key =  'flux' ,
                observation_err_key =  'flux_err',
                time_key =  'time',
                time_alert_key =  'time_alert' ,
                mask_key =  'mask',
                feature_key =  'feat_cols',
                metadata_key =  'metadata_feat',
                label_key =  'labels',
                use_sampler = True):
        from src.data.modules.LitData import LitData
        datamodule = {'dataset':
                            {'experiment_type': experiment_type,
                            'data_root': data_root,
                            'seed': seed,
                            'train_apply_transform': train_apply_transform,
                            'validation_apply_transform': validation_apply_transform,
                            'transforms': transforms,
                            'train_key': train_key,
                            'validation_key': validation_key,
                            'test_key':  test_key,
                            'observation_key': observation_key ,
                            'observation_err_key': observation_err_key,
                            'time_key': time_key,
                            'time_alert_key': time_alert_key ,
                            'mask_key': mask_key,
                            'mask_photometry_key': mask_key,
                            'mask_detection_key': mask_key,
                            'feature_key': feature_key,
                            'metadata_key': metadata_key,
                            'label_key': label_key},
            'train_use_sampler': use_sampler,
            'train_shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            'batch_size': batch_size,
        }
        pl_datal = LitData(**datamodule)
         
        self.train = pl_datal.train_dataloader()
        self.validation = pl_datal.val_dataloader()
        self.test = pl_datal.test_dataloader()

class InitBackbone:
    def __init__(
        self,
        path_to_config_yaml, 
        model,
        arg_key,
        device = 'cpu', ):
        self.device = device
        self.path_to_config_yaml = path_to_config_yaml
        self.args = self._load_yaml_args(path_to_config_yaml).ATATConfig
        self.backbone = self.init_model(model, arg_key)

    @staticmethod
    def _load_yaml_args(path_to_config_yaml):
        import yaml
        from  hydra.utils import instantiate
        path_args = glob.glob(f"{path_to_config_yaml}/.hydra/*config*")[0]
        with open(path_args, "r") as file:
            args = yaml.safe_load(file)
        args =  instantiate(args)
        return args
    
    def init_model(self,model, arg_key:str):
        model = model(**self.args[arg_key]) 
        return model   
       
    def create_ordered_dict(self,
                            checkpoint_name: str = 'pretrain_ckpt',
                            remove_if_in_key_list:list = ['projection'], 
                            rename_keys: tuple = ('model.',''), print_keys = False):
            checkpoint_path_clip = glob.glob(f"{self.path_to_config_yaml}*{checkpoint_name}*")
            assert isinstance(rename_keys,tuple)
            print('Found checkpoint {}'.format(checkpoint_path_clip[-1].split('=')[-1]))
            checkpoint_clip = load(
                checkpoint_path_clip[-1], map_location=device(self.device)
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                if print_keys:
                    print('old key name:',key)
                if any([remove_if_in_key in key for remove_if_in_key in remove_if_in_key_list ]):
                    continue
                od_atat[key.replace(f"{rename_keys[0]}", f"{rename_keys[1]}")] = checkpoint_clip[
                    "state_dict"
                ][key]
                #if print_keys:
                 #   print('new key name:', od_atat[key])
            return od_atat
    
    def load_backbone_weights(self,weights: dict, strict = True):
        self.backbone.load_state_dict(weights, strict=strict)
        print("Loaded backbone weights")

    def predict(self,dataloader, device = None, return_count_len  = False):
        if device is not None:
            self.device = device
            print("device set to {}".format(device))
        target = None
        preds_out = None
        count_len = None
       
        self.backbone.eval().to(device=self.device)
        
        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            emb = self.backbone(**b1) 
            flux = np.count_nonzero(b1['data'].clone().detach().cpu().numpy(), axis =1)
            count_len = (
                np.concatenate([count_len, flux])
                if count_len is not None
                else flux
            )   
            preds_out = (
                np.concatenate([preds_out, emb.detach().cpu().numpy()])
                if preds_out is not None
                else emb.cpu().detach().numpy()
            )
            target = (
                np.concatenate([target, t.cpu().detach().numpy()])
                if target is not None
                else t.detach().cpu().numpy()
            )
        self.backbone.to(device="cpu")
        return (preds_out, target, count_len) if return_count_len else (preds_out, target)


class InitDataLoader:
    def __init__(self,update_dataset_path,datamodule_args,update_batch_size =128):
        dataset ={
            key: value 
            for key, value in datamodule_args.dataset.items() 
            if key not in ['transforms_1','transforms_2']
        }
        dataset['feature_key'] = 'extracted_feat_2048'
        datamodule_args_copy = deepcopy(datamodule_args)
        datamodule_args_copy['dataset'] = dataset
        datamodule_args_copy['dataset']['data_root'] = update_dataset_path
        datamodule_args_copy['batch_size'] = update_batch_size
        self.datamodule = datamodule_args_copy
        self.pl_datal = LitData(**self.datamodule)
         
        self.train_dataset = self.pl_datal.train_dataloader()
        self.validation_dataset = self.pl_datal.val_dataloader()
        self.test_dataset = self.pl_datal.test_dataloader()

    def set_sampler(self,use_sampler:bool = False):
        if use_sampler == True:
            self.datamodule['train_use_sampler']  = True
            self.pl_datal =  LitData(**self.datamodule)
            self.train = self.pl_datal.train_dataloader()
        else:
            self.datamodule['train_use_sampler']  = False
            self.pl_datal =  LitData(**self.datamodule)
            self.train = self.pl_datal.train_dataloader()



class InitClassifier(InitBackbone):
    def __init__(
        self,
        path_to_config_yaml, 
        model,
        classifier,
        arg_key,
        use_lc = False,
        use_tab = False,
        use_mix = False,
        device = 'cpu'):
        super().__init__(path_to_config_yaml, 
                        model,
                        arg_key,
                        device)
        self.classifier = classifier(lc_input_size = self.args.lc.embedding_size,
                 tab_input_size = self.args.tab.embedding_size,
                 use_lc = use_lc,
                 use_tab = use_tab,
                 use_mix = use_mix,
                 num_classes = self.args.num_classes,
                 dropout=0.0)
        print(self.backbone)
        print(self.classifier)

    def load_classifier_weights(self,weights: dict, strict = True):
        self.classifier.load_state_dict(weights, strict=strict)
        print("LOaded classifier weights")
    
    def predict(self,dataloader, device = None):
        if device is not None:
            self.device = device
            print("device set to {}".format(device))
        target = None
        preds_out = None
        print(self.backbone)
        self.backbone.eval().to(device=self.device)
        self.classifier.eval().to(device=self.device)
        
        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            emb = self.backbone(**b1) 
            emb = self.classifier(emb)
            if isinstance(emb, dict):
                if 'LC' in emb.keys():
                    output = emb["LC"]
                if 'TAB' in emb.keys():
                    output = emb["TAB"]
                if 'MIX' in emb.keys():
                    output = emb["MIX"]
            preds_out = (
                np.concatenate([preds_out, output.detach().cpu().numpy()])
                if preds_out is not None
                else output.cpu().detach().numpy()
            )
            target = (
                np.concatenate([target, t.cpu().detach().numpy()])
                if target is not None
                else t.detach().cpu().numpy()
            )
        self.backbone.to(device="cpu")
        self.classifier.to(device="cpu")
        return preds_out, target
    
    def logistic_regression(self, X_train, y_train,X_test, y_test,taxonomy,knn_args:dict):
        std_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        ('model', LogisticRegression(random_state=0, max_iter = 500, multi_class = 'ovr'))
        ])

        # You can change n_neighbors as needed
        std_pipeline.fit(X_train,y_train)
        train_y_pred = std_pipeline.predict(X_train)

        classification = classification_report(y_train,train_y_pred, target_names=list(taxonomy.keys()),digits = 4)
        #print(classification)
        test_y_pred = std_pipeline.predict(X_test)

        classification = classification_report(y_test,test_y_pred, target_names=list(taxonomy.keys()),digits = 4)
        print(classification)

    def knn_classifier(self, X_train, y_train,X_test, y_test,taxonomy,knn_args:dict):

        knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        ('model', KNeighborsClassifier(**knn_args))
        ])

        knn_pipeline.fit(X_train, y_train)
        knn_preds = knn_pipeline.predict(X_test)

        classification = classification_report(y_test,knn_preds, target_names=list(taxonomy.keys()),digits = 4)
        print(classification)
        return knn_preds
    
    def get_confusion_matrix( self,preds,target,  dataset_type:str, taxonomy):
        from sklearn.metrics import classification_report
        import matplotlib.pyplot as plt
        order_classes = ['SNIa', # yes
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
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        fs = 11
        y_true = [taxonomy.values_as_keys()[i] for i in np.array(target).astype(int)]
        y_pred = [taxonomy.values_as_keys()[i] for i in np.array(preds).astype(int)]

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=order_classes, normalize='true')
        np.set_printoptions(precision=4, suppress=True)
        cmap = plt.cm.Blues
        fig, ax = plt.subplots(figsize=(11, 11)) #, dpi=110)
        decimals = 2
        im = ax.imshow(np.around(cm, decimals=decimals), interpolation='nearest', cmap=cmap)
        # color map
        new_color = cmap(1.0) 

        # Añadiendo manualmente las anotaciones con la media y desviación estándar
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] >= 0.005:
                    #print(cm[i, j])
                    text = f'{np.around(cm[i, j], decimals=decimals)}'
                    color = "white" if cm[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fs)
                else:
                    text = f'{np.around(cm[i, j], decimals=decimals)}'
                    color = "white" if cm[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fs)

        # Ajustes finales y mostrar la gráfica
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xticks(np.arange(len(order_classes)))
        ax.set_yticks(np.arange(len(order_classes)))
        ax.set_xticklabels(order_classes)
        ax.set_yticklabels(order_classes)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        f1_ = classification_report(y_true,y_pred, target_names=list(taxonomy().keys()),digits = 4, output_dict=True)['macro avg']['f1-score']
        ax.set_title(f'ATAT dataset zero shot knn: {dataset_type} | macro f1: {np.round(f1_,4)}', fontsize=16, pad=13)
        ax.set_xlabel('Predicted label', fontsize=16, labelpad=13)  # Label del eje x
        ax.set_ylabel('True label', fontsize=16, labelpad=13)        # Label del eje y

        #ax.xaxis.label.set_size(16)
        #ax.yaxis.label.set_size(16)
        #ax.xaxis.labelpad = 13
        #ax.yaxis.labelpad = 13
        return ax
class ZeroShotRegressor:
    def __init__(self,model,
                       dl, 
                       taxonomy,
                       device = 'cpu', regressor_args =  {'random_state':0, 'max_iter' : 100}):
        self.taxonomy = taxonomy
        self.pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        ('model', LogisticRegression(**regressor_args))
        ])
    def fit_regressor(self, ):
       
        X_train, y_train =model.predict(dl.train_dataset, device)
        X_test, y_test =model.predict(dl.validation_dataset, device)
        
        
        # You can change n_neighbors as needed
        std_pipeline.fit(X_train,y_train)
        train_y_pred = std_pipeline.predict(X_train)

        classification = classification_report(y_train,train_y_pred, target_names=list(self.taxonomy.keys()),digits = 4)
        print('train dataset classification report:')
        print(classification)
        print('\n')
        test_y_pred = std_pipeline.predict(X_test)

        classification = classification_report(y_test,test_y_pred, target_names=list(self.taxonomy.keys()),digits = 4)
        print('validation dataset classification report:')
        print(classification)
        print('\n')
        pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        ('model', KNeighborsClassifier())
        ])

        pipeline.fit(X_train, y_train)
        knn_preds = pipeline.predict(X_test)

        classification = classification_report(y_test,knn_preds, target_names=list(self.taxonomy.keys()),digits = 4)
        print(classification)




class ZeroShotKNN:
    def __init__(self, taxonomy):
        self.taxonomy = taxonomy

    def fit_regressor(self, model,
                       dl, 
                       device = 'cpu', regressor_args =  {'random_state':0, 'max_iter' : 100}):
        from sklearn.metrics import classification_report
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        X_train, y_train =model.predict(dl.train_dataset, device)
        X_test, y_test =model.predict(dl.validation_dataset, device)
        
        std_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        ('model', LogisticRegression(**regressor_args))
        ])
        # You can change n_neighbors as needed
        std_pipeline.fit(X_train,y_train)
        train_y_pred = std_pipeline.predict(X_train)

        classification = classification_report(y_train,train_y_pred, target_names=list(self.taxonomy.keys()),digits = 4)
        print('train dataset classification report:')
        print(classification)
        print('\n')
        test_y_pred = std_pipeline.predict(X_test)

        classification = classification_report(y_test,test_y_pred, target_names=list(self.taxonomy.keys()),digits = 4)
        print('validation dataset classification report:')
        print(classification)
        print('\n')
        pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        ('model', KNeighborsClassifier())
        ])

        pipeline.fit(X_train, y_train)
        knn_preds = pipeline.predict(X_test)

        classification = classification_report(y_test,knn_preds, target_names=list(self.taxonomy.keys()),digits = 4)
        print(classification)

class UMAPExplorer:
    def __init__(self, model, datamodule, taxonomy):
        self.model = model
        self.datamodule = datamodule
        self.taxonomy = taxonomy

        
    def umap_mapper(self, set_type = 'validation', umap_args: dict = {}, device = None):
        import pandas as pd
        import umap
        from bokeh.resources import INLINE
        import bokeh.io
        
        bokeh.io.output_notebook(INLINE)
        if set_type in ['training','train']:
            preds_out, target, count_len = self.model.predict(self.datamodule.train_dataset, device = device, return_count_len = True)
        elif set_type in ['validation']:
            preds_out, target, count_len = self.model.predict(self.datamodule.validation_dataset, device = device, return_count_len = True)
        elif set_type in ['test']:
            preds_out, target, count_len = self.model.predict(self.datamodule.test_dataset, device = device, return_count_len = True)
        else:
            return
        hover_data = pd.DataFrame({'index':range(len(preds_out)),
                                'label':target, 
                                'band_0_count':count_len[:,0],
                                'band_1_count':count_len[:,1] 
                                })
        hover_data['item'] = hover_data.label.map({value:key for key,value in self.taxonomy().items()})
        #preds_out = (preds_out/preds_out.max())*1000
        umap_obj = umap.UMAP(**umap_args)
        mapper = umap_obj.fit(preds_out)
        return hover_data, mapper, preds_out,target
    
    def find_DBSCAN_clusters(self,preds, target,eps=0.3, min_samples=5):
        dbscan = DBSCAN(eps = eps, min_samples =  min_samples)
        labels = dbscan.fit_predict(preds)
        plt.figure(figsize=(6, 4))
        plt.scatter(preds[:, 0], preds[:, 1], c=labels, cmap='viridis', s=30)
        plt.title("Clustering")  # Update title based on method
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return preds, labels, target


class AutoExperimentExplorer(InitDataLoader,InitBackbone):
    def __init__(self, path_to_config_yaml, model, arg_key,device = 'cpu'):
        self.args = self._load_yaml_args(path_to_config_yaml).ATATConfig
        InitDataLoader.__init__(device = device, **self.args['datamodule'])
        InitBackbone.__init__(**self.args[arg_key])
    
    def umap_mapper(self, set_type = 'validation'):
        import pandas as pd
        import umap.plot
        from bokeh.resources import INLINE
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=set_type, seed=self.seed
        )
        umap.plot.output_notebook(resources=INLINE)
        preds_out, target = self._predict(dataloader, modality = modality)
        hover_data = pd.DataFrame({'index':range(len(preds_out)),
                                'label':target})
        hover_data['item'] = hover_data.label.map({value:key for key,value in self.taxonomy().items()})
        preds_out = (preds_out/preds_out.max())*1000
        mapper = umap.UMAP(**self.umap_args).fit(preds_out)
        return hover_data,target, mapper
    
class Report:
    def __init__(
        self,
        path_to_training_dir, 
        path_to_dataset, 
        model_class, model_type, 
        custom_parse_key_str,
        taxonomy, 
        seed, 
        device, 
        batch_size, 
        load_checkpoint, 
        **kwargs,
    ):
        
        self.cfg = self._load_yaml_args(path_to_training_dir).ATATConfig
        self.taxonomy = taxonomy
        self.path_to_dataset = path_to_dataset 
        self.device = device
        self.load_checkpoint = load_checkpoint 
        self.seed = seed
        self.custom_parse_key_str = custom_parse_key_str
        self.model_type = model_type
        self.checkpoint_src = path_to_training_dir
        self.batch_size = batch_size
        
    @staticmethod
    def _load_yaml_args(path_args):
        import yaml
        from  hydra.utils import instantiate
        path_args = glob.glob(f"{path_args}/.hydra/*config*")[0]
        with open(path_args, "r") as file:
            args = yaml.safe_load(file)
        args =  instantiate(args)
        return args

    def _init_dataloader(
        self, path_to_dataset, 
        set_type="test", 
        seed=0,  
        sampler = None,
    ):
        from src.data.modules.LitData import LitData
        from collections import OrderedDict
        dataset_exclude_transform_keys ={
            key: value 
            for key, value in self.cfg.datamodule.dataset.items() 
            if key not in ['transforms_1','transforms_2', 'data_root', 'batch_size']
        }
        modded_config = deepcopy(self.cfg)
        modded_config.datamodule.dataset = dataset_exclude_transform_keys
        modded_config.datamodule.dataset.data_root = self.path_to_dataset
        modded_config.datamodule.batch_size = self.batch_size

       # modded_config.datamodule.dataset.train_key = 'modded_training'
       # modded_config.datamodule.dataset.validation_key = 'modded_validation'
       # modded_config.datamodule.dataset.test_key = 'modded_test'

        if sampler is not None:
            modded_config.datamodule.train_use_sampler = sampler
            print('sampler set to {}'.format(sampler))
        pl_datal = LitData(**modded_config.datamodule)
        if set_type in ["train","training"]:
            dataloader = pl_datal.train_dataloader()
        elif set_type == "validation":
            dataloader = pl_datal.val_dataloader()
        elif set_type == "test":
            dataloader = pl_datal.test_dataloader()
        return dataloader
    
    def _predict(self,dataloader, modality):
        target = None
        preds_out = None
        self.model.eval().to(device=self.device)
        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            lc_emb = self.model(**b1)  # [:, 0, :]
            if modality =='LC':
                lc_emb = lc_emb["LC"]
            else:
                lc_emb = lc_emb
            preds_out = (
                np.concatenate([preds_out, lc_emb.detach().cpu().numpy()])
                if preds_out is not None
                else lc_emb.cpu().detach().numpy()
            )
            target = (
                np.concatenate([target, t.cpu().detach().numpy()])
                if target is not None
                else t.detach().cpu().numpy()
            )
        self.model.to(device="cpu")
        return preds_out, target
    def _embeddings(self,dataloader, modality):
        target = None
        preds_out = None
        self.model.eval().to(device=self.device)
        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            lc_emb = self.model.get_embeddings(**b1)  # [:, 0, :]
            if modality =='LC':
                lc_emb = lc_emb["LC"]
            else:
                lc_emb = lc_emb
            preds_out = (
                np.concatenate([preds_out, lc_emb.detach().cpu().numpy()])
                if preds_out is not None
                else lc_emb.cpu().detach().numpy()
            )
            target = (
                np.concatenate([target, t.cpu().detach().numpy()])
                if target is not None
                else t.detach().cpu().numpy()
            )
        self.model.to(device="cpu")
        return preds_out, target

class ReportClassification(Report):
    def __init__(self, 
                 path_to_training_dir, 
                 path_to_dataset, 
                 taxonomy,
                 model_class, 
                 model_type, 
                 custom_parse_key_str,
                 seed = 0, 
                 device="cpu", 
                 batch_size=128, 
                 load_checkpoint=True, 
                 **kwargs):
        
        super().__init__(path_to_training_dir, 
                         path_to_dataset, 
                         model_class, model_type, 
                         custom_parse_key_str,
                         taxonomy, 
                         seed, 
                         device, 
                         batch_size, 
                         load_checkpoint, 
                         **kwargs)
        self.model = self._init_model(model_class)
        self.taxonomy = taxonomy

    def _init_model(self,model ):

        model = model(**self.cfg.lc) if self.model_type =='lc' else model(**self.cfg.tab)
        classifier =  MultimodalClassifier(lc_input_size=self.cfg.lc.embedding_size,
                                          tab_input_size=None, 
                                          use_lc= True,
                                          num_classes= self.cfg.num_classes)
        model = ClassifierBaseModel(model, classifier)
        if self.checkpoint_src is not None or self.load_checkpoint: 
            checkpoint_path_clip = glob.glob(f"{self.checkpoint_src}*classifier_ckpt*")
            print(checkpoint_path_clip)
            print('using checkpoint {}'.format(checkpoint_path_clip[-1].split('=')[-1]))
            checkpoint_clip = load(
                checkpoint_path_clip[-1], map_location=device(self.device)
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                print(key)
                if 'projection' in key:
                    continue
                od_atat[key.replace(f"{self.custom_parse_key_str}", "")] = checkpoint_clip[
                    "state_dict"
                ][key]
            model.load_state_dict(od_atat, strict=True)
        else:
            print('NO CKPT LOADED')
        return model
    
    def classification_report(self, dataset_type: str = 'validation', digits = 4, confusion_matrix = True, modality = 'LC'):
        from sklearn.metrics import classification_report
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=dataset_type, seed=self.seed
        )
        preds_out, target = self._predict(dataloader, modality)
        preds_out = np.argmax(preds_out, axis = -1)
        classification = classification_report(target,preds_out, target_names=list(self.taxonomy().keys()),digits = digits)
        print(classification)
        if confusion_matrix:
            self.get_confusion_matrix(preds_out,target, dataset_type=dataset_type)
        return classification_report(target,preds_out, target_names=list(self.taxonomy().keys()),digits = digits, output_dict=True)
    def predict_probs(self, dataset_type: str = 'validation', modality = 'LC'):
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=dataset_type, seed=self.seed
        )
        preds_out, target = self._predict(dataloader, modality)
        #preds_out = np.argmax(preds_out, axis = -1)
        return preds_out, target
    def get_embeddings(self, dataset_type: str = 'validation', modality = 'LC'):
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=dataset_type, seed=self.seed
        )
        preds_out, target = self._embeddings(dataloader, modality)
        #preds_out = np.argmax(preds_out, axis = -1)
        return preds_out, target

    def get_confusion_matrix(self, preds,target,  dataset_type:str):
        from src.utils.plots.ATATConfusionMatrix import elasticc_confusion_matrix
        from sklearn.metrics import classification_report
        import matplotlib.pyplot as plt
        order_classes = ['SNIa', # yes
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
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        fs = 11
        y_true = [self.taxonomy.values_as_keys()[i] for i in np.array(target).astype(int)]
        y_pred = [self.taxonomy.values_as_keys()[i] for i in np.array(preds).astype(int)]

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=order_classes, normalize='true')
        np.set_printoptions(precision=4, suppress=True)
        cmap = plt.cm.Blues
        fig, ax = plt.subplots(figsize=(11, 11)) #, dpi=110)
        decimals = 2
        im = ax.imshow(np.around(cm, decimals=decimals), interpolation='nearest', cmap=cmap)
        # color map
        new_color = cmap(1.0) 

        # Añadiendo manualmente las anotaciones con la media y desviación estándar
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] >= 0.005:
                    #print(cm[i, j])
                    text = f'{np.around(cm[i, j], decimals=decimals)}'
                    color = "white" if cm[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fs)
                else:
                    text = f'{np.around(cm[i, j], decimals=decimals)}'
                    color = "white" if cm[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fs)

        # Ajustes finales y mostrar la gráfica
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xticks(np.arange(len(order_classes)))
        ax.set_yticks(np.arange(len(order_classes)))
        ax.set_xticklabels(order_classes)
        ax.set_yticklabels(order_classes)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        f1_ = classification_report(y_true,y_pred, target_names=list(self.taxonomy().keys()),digits = 4, output_dict=True)['macro avg']['f1-score']
        ax.set_title(f'ATAT dataset: {dataset_type} | macro f1: {np.round(f1_,4)}', fontsize=16, pad=13)
        ax.set_xlabel('Predicted label', fontsize=16, labelpad=13)  # Label del eje x
        ax.set_ylabel('True label', fontsize=16, labelpad=13)        # Label del eje y

        #ax.xaxis.label.set_size(16)
        #ax.yaxis.label.set_size(16)
        #ax.xaxis.labelpad = 13
        #ax.yaxis.labelpad = 13
        return ax
    
    

class ReportZeroshot(Report):
    def __init__(
        self, 
        path_to_training_dir, 
        path_to_dataset, 
        taxonomy,
        model_class, 
        model_type, 
        
        custom_parse_key_str,
        seed = 0, 
        device="cpu", 
        batch_size=128, 
        load_checkpoint = True,
        figsize = (20,20),
        umap_args={"n_neighbors": 15, "min_dist": 0.25, "metric": "euclidean"},
        marker_size = 4
    ):
        super().__init__(path_to_training_dir=path_to_training_dir,
                        path_to_dataset=path_to_dataset,
                        model_type=model_type,
                        model_class=model_class,
                        taxonomy=taxonomy,
                        custom_parse_key_str=custom_parse_key_str,
                        seed=seed,
                        device= device,
                        batch_size=batch_size,
                        load_checkpoint=load_checkpoint)
        
        from torch import device, load
        from collections import OrderedDict
        self.model = model_class(**self.cfg.lc) if self.model_type =='lc' else model_class(**self.cfg.tab)
        if self.checkpoint_src is not None or self.load_checkpoint:
            print(glob.glob(self.checkpoint_src))
            checkpoint_path_clip = glob.glob(f"{self.checkpoint_src}*pretrain_ckpt*")
            print('using checkpoint {}'.format(checkpoint_path_clip[-1].split('=')[-1]))
            checkpoint_clip = load(
                checkpoint_path_clip[-1], map_location=device(self.device)
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                if 'model' not in key:
                    continue
                od_atat[key.replace(f"{self.custom_parse_key_str}", "")] = checkpoint_clip[
                    "state_dict"
                ][key]
            self.model.load_state_dict(od_atat, strict=True)
        else:
            print('NO CKPT LOADED')
        self.taxonomy = taxonomy
        self.figsize =figsize
        self.marker_size = marker_size
        self.umap_args = umap_args 
        
    
    
    def predict(self, dataset_type ='training', modality = 'LC', sampler = None):
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, 
            set_type=dataset_type, 
            seed=self.seed,
            sampler = sampler,
        )
        preds_out, target = self._predict(dataloader, modality)
        return preds_out, target

    def fit_knn(self, dataset_type ='training', modality = 'LC', sampler= False, knn_params:dict = { 'n_neighbors':3, 'weights':"distance"}):
        from sklearn.metrics import classification_report
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, 
            set_type=dataset_type, 
            seed=self.seed,
            sampler = sampler,
        )
        preds_out, target = self._predict(dataloader, modality)
        
        return preds_out, target

    
    def pred_knn(self, dataset_type ='validation', modality = 'LC', sampler= False):
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, 
            set_type=dataset_type, 
            seed=self.seed,
            sampler = sampler,
        )
        target = None
        preds_out = None
        self.model.eval().to(device=self.device)
        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            

            lc_emb = self.model(**b1)  # [:, 0, :]
            preds_out = (
                np.concatenate([preds_out, lc_emb.detach().cpu().numpy()])
                if preds_out is not None
                else lc_emb.cpu().detach().numpy()
            )
            target = (
                np.concatenate([target, t.cpu().detach().numpy()])
                if target is not None
                else t.detach().cpu().numpy()
            )
        self.model.to(device="cpu")
        preds_out = self.knn_model.predict(preds_out)
        return preds_out, target

class ReportPretraining(Report):
    
    def __init__(
        self, 
        path_to_training_dir, 
        path_to_dataset, 
        taxonomy,
        model_class, 
        model_type, 
        custom_parse_key_str,
        seed = 0, 
        device="cpu", 
        batch_size=128, 
        load_checkpoint = True,
        figsize = (20,20),
        umap_args={"n_neighbors": 15, "min_dist": 0.25, "metric": "euclidean"},
        marker_size = 4
    ):
        super().__init__(path_to_training_dir=path_to_training_dir,
                        path_to_dataset=path_to_dataset,
                        model_type=model_type,
                        model_class=model_class,
                        taxonomy=taxonomy,
                        custom_parse_key_str=custom_parse_key_str,
                        seed=seed,
                        device= device,
                        batch_size=batch_size,
                        load_checkpoint=load_checkpoint)
        self.model = self._init_model(model_class)
        self.taxonomy = taxonomy
        self.figsize =figsize
        self.marker_size = marker_size
        self.umap_args = umap_args 
    
    def mapper(self, set_type = 'validation', modality = "LC"):
        import pandas as pd
        import umap.plot
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=set_type, seed=self.seed
        )
        from bokeh.resources import INLINE

        umap.plot.output_notebook(resources=INLINE)
        preds_out, target = self._predict(dataloader, modality = modality)
        hover_data = pd.DataFrame({'index':range(len(preds_out)),
                                'label':target})
        hover_data['item'] = hover_data.label.map({value:key for key,value in self.taxonomy().items()})
        preds_out = (preds_out/preds_out.max())*1000
        mapper = umap.UMAP(**self.umap_args).fit(preds_out)
        return hover_data,target, mapper
    
    def generate_umap_plots(self, set_type = 'validation'):
        import matplotlib.pyplot as plt
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=set_type, seed=self.seed
        )
        self.preds_out, self.target = self._predict(dataloader)
        #umap_result = self._umap(n_components=2, **self.umap_args)
        #self._2d_umap_plot(umap_result, self.target, "by_class", type_="by_class")
        #plt.show() 
        #self._2d_umap_plot(umap_result, self.target, "by_hierarchy", type_="by_hierarchy")
        #plt.show()
        umap_result = self._umap(n_components=3, **self.umap_args)
        self._3d_umap_plot(umap_result, self.target, "3d", type_="by_class")
        plt.show()
        

    def _init_model(self,model ):
        from torch import device, load
        from collections import OrderedDict
        model = model(**self.cfg.lc) if self.model_type =='lc' else model(**self.cfg.tab)
        if self.checkpoint_src is not None or self.load_checkpoint:
            print(glob.glob(self.checkpoint_src))
            checkpoint_path_clip = glob.glob(f"{self.checkpoint_src}*pretrain_ckpt*")
            print('using checkpoint {}'.format(checkpoint_path_clip[-1].split('=')[-1]))
            checkpoint_clip = load(
                checkpoint_path_clip[-1], map_location=device(self.device)
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                if 'model' not in key:
                    continue
                od_atat[key.replace(f"{self.custom_parse_key_str}", "")] = checkpoint_clip[
                    "state_dict"
                ][key]
            model.load_state_dict(od_atat, strict=True)
        else:
            print('NO CKPT LOADED')
        return model
    
    def _umap(self, n_components, metric, min_dist, n_neighbors):
        import umap
        print(f"creating {n_components}D UMAP reduction for data...")
        umap_model = umap.UMAP(
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
        )
        umap_result = umap_model.fit_transform(self.preds_out)
        return umap_result

    def _2d_umap_plot(self, umap_result, target, title, type_="by_class"):
        import matplotlib.pyplot as plt
        from src.utils.plots.umap_plots import plot_umap,big_group_plot_umap
        fig, axs = plt.subplots(1, 1, figsize=self.figsize)
        plot_umap(
            axs, umap_result, target, len(self.taxonomy), title,marker_size=self.marker_size
        ) if type_ == "by_class" else big_group_plot_umap(
            axs, umap_result, target, len(self.taxonomy), title,marker_size =self.marker_size
        )
        plt.savefig('{}/2d_umap_{}.pdf'.format(self.checkpoint_src,type_))
        return fig, axs

    def _3d_umap_plot(self, umap_result, target, title, type_="by_class"):
        import matplotlib.pyplot as plt
        from src.utils.plots.umap_plots import plot_umap_3d
        fig = plt.figure(figsize=self.figsize, facecolor='black')
        ax = fig.add_subplot(111, projection="3d")
        plot_umap_3d(ax, umap_result, target, len(self.taxonomy), "UMAP 3D Visualization",marker_size=self.marker_size)
        plt.savefig('{}/3d_umap_{}.pdf'.format(self.checkpoint_src,type_))
        return fig, ax

    def _knn(self, plot_cm=True):
        from src.utils.KNNClassifier import KNNClassifier
        train_dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="train", seed=self.seed
        )
        val_dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="validation", seed=self.seed
        )
        test_dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="test", seed=self.seed
        )
        KNNClassifier(self._predict(train_dataloader),
                      self._predict(val_dataloader),
                      self._predict(test_dataloader), plot_cm=True)
        
class ComparePerformance:
    def __init__(self, 
                 path_1,
                 path_2,
                 path_to_dataset, 
                 model_class, 
                 model_type,  
                 custom_parse_key_str, 
                 seed = 0, 
                 device="cpu", 
                 batch_size=128, 
                 load_checkpoint=True ):
        self.taxonomy = None
        self.model_1 = ReportClassification(path_1, 
                                            path_to_dataset, 
                                            model_class, 
                                            model_type,  
                                            custom_parse_key_str, 
                                            seed = seed, 
                                            device=device, 
                                            batch_size=batch_size, 
                                            load_checkpoint=load_checkpoint )
        self.model_2 = ReportClassification(path_2, 
                                            path_to_dataset, 
                                            model_class, 
                                            model_type,  
                                            custom_parse_key_str, 
                                            seed = seed, 
                                            device=device, 
                                            batch_size=batch_size, 
                                            load_checkpoint=load_checkpoint )
    
    def compare(self, dataset_type:str= 'validation', eval_time = 2048):
        from sklearn.metrics import classification_report
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=dataset_type, seed=self.seed
        )
        preds_1, target = self.model_1._predict(dataloader)
        preds_2,_ = self.model_2._predict(dataloader)
        cr_1  = classification_report(np.argmax(preds_1, axis = -1), target, output_dict=True)
        cr_2  = classification_report(np.argmax(preds_2, axis = -1), target, output_dict=True)
        f1_1 = []
        f1_2 = []
        fig,ax = plt.subplots(1,1,figsize = (8,5))
        for key,value in cr_1.items():
            if key in self.taxonomy().keys():
                #print(key,value)
                f1_1.append(value['f1-score'])
        df_1 = pd.DataFrame({'group':self.taxonomy().keys(), 'values':f1_1 })
        ordered_df_1 = df_1#.sort_values(by='values')
        my_range_1=range(1,len(df_1.index)+1)
        for key,value in cr_2.items():
            if key in self.taxonomy().keys():
                #print(key,value)
                f1_2.append(value['f1-score'])
        df_2 = pd.DataFrame({'group':self.taxonomy().keys(), 'values':f1_2})
        ordered_df_2 = df_2#.sort_values(by='values')
        my_range_2=range(1,len(df_2.index)+1)
        import matplotlib.pyplot as plt
        plt.plot(ordered_df_1['values'].values, my_range_1, "o", alpha = 0.8,color='blue')
        plt.plot(ordered_df_2['values'].values, my_range_2, "o", alpha = 0.8,color='red')
        plt.legend(['1','2'])
        # The horizontal plot is made using the hline function
        plt.hlines(y=my_range_1, xmin=0, xmax=ordered_df_1['values'], color='blue', alpha = 0.5)
        # The horizontal plot is made using the hline function
        plt.hlines(y=my_range_2, xmin=0, xmax=ordered_df_2['values'], color='red', alpha = 0.5)
        # Add titles and axis names
        plt.yticks(my_range_1, ordered_df_1['group'])
        plt.title(f"F1-Score LC Classifier Comparison for Eval Time {eval_time} ", loc='center')
        plt.xlabel('F1-Score')
        plt.ylabel('Class')
        plt.xlim(0,1)
        plt.grid('on', )
        # Show the plot
        plt.show()



class ReportMultimodal(Report):
    
    def __init__(
        self, 
        path_to_training_dir, 
        path_to_dataset, 
        taxonomy,
        model_class, 
        model_type, 
        custom_parse_key_str,
        seed = 0, 
        device="cpu", 
        batch_size=128, 
        load_checkpoint = True,
        figsize = (20,20),
        umap_args={"n_neighbors": 15, "min_dist": 0.25, "metric": "euclidean"},
        marker_size = 4
    ):
        super().__init__(path_to_training_dir=path_to_training_dir,
                        path_to_dataset=path_to_dataset,
                        model_type=model_type,
                        model_class=model_class,
                        taxonomy=taxonomy,
                        custom_parse_key_str=custom_parse_key_str,
                        seed=seed,
                        device= device,
                        batch_size=batch_size,
                        load_checkpoint=load_checkpoint)
        self.model = self._init_model(model_class)
        self.taxonomy = taxonomy
        self.figsize =figsize
        self.marker_size = marker_size
        self.umap_args = umap_args 

    def predict(self, set_type = 'validation', modality = "LC"):
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=set_type, seed=self.seed
        )
        preds_out, target = self._predict(dataloader, modality = modality)
        return preds_out,target
    def mapper(self, set_type = 'validation', modality = "LC"):
        import pandas as pd
        import umap.plot
        from bokeh.resources import INLINE
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=set_type, seed=self.seed
        )
        umap.plot.output_notebook(resources=INLINE)
        preds_out, target = self._predict(dataloader, modality = modality)
        hover_data = pd.DataFrame({'index':range(len(preds_out)),
                                'label':target})
        hover_data['item'] = hover_data.label.map({value:key for key,value in self.taxonomy().items()})
        preds_out = (preds_out/preds_out.max())*1000
        mapper = umap.UMAP(**self.umap_args).fit(preds_out)
        return hover_data,target, mapper
    
    def generate_umap_plots(self, set_type = 'validation'):
        import matplotlib.pyplot as plt
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=set_type, seed=self.seed
        )
        self.preds_out, self.target = self._predict(dataloader)
        #umap_result = self._umap(n_components=2, **self.umap_args)
        #self._2d_umap_plot(umap_result, self.target, "by_class", type_="by_class")
        #plt.show() 
        #self._2d_umap_plot(umap_result, self.target, "by_hierarchy", type_="by_hierarchy")
        #plt.show()
        umap_result = self._umap(n_components=3, **self.umap_args)
        self._3d_umap_plot(umap_result, self.target, "3d", type_="by_class")
        plt.show()
        

    def _init_model(self,model ):
        from torch import device, load
        from collections import OrderedDict
        model = model(**self.cfg.lc) if self.model_type =='lc' else model(**self.cfg.tab)
        if self.checkpoint_src is not None or self.load_checkpoint:
            print(glob.glob(self.checkpoint_src))
            checkpoint_path_clip = glob.glob(f"{self.checkpoint_src}*pretrain_ckpt*")
            print('using checkpoint {}'.format(checkpoint_path_clip[-1].split('=')[-1]))
            checkpoint_clip = load(
                checkpoint_path_clip[-1], map_location=device(self.device)
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                if f"{self.custom_parse_key_str}" not in key:
                    continue
                od_atat[key.replace(f"{self.custom_parse_key_str}", "")] = checkpoint_clip[
                    "state_dict"
                ][key]
            model.load_state_dict(od_atat, strict=True)
        else:
            print('NO CKPT LOADED')
        return model
    
    def _umap(self, n_components, metric, min_dist, n_neighbors):
        import umap
        print(f"creating {n_components}D UMAP reduction for data...")
        umap_model = umap.UMAP(
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
        )
        umap_result = umap_model.fit_transform(self.preds_out)
        return umap_result

    def _2d_umap_plot(self, umap_result, target, title, type_="by_class"):
        import matplotlib.pyplot as plt
        from src.utils.plots.umap_plots import plot_umap,big_group_plot_umap
        fig, axs = plt.subplots(1, 1, figsize=self.figsize)
        plot_umap(
            axs, umap_result, target, len(self.taxonomy), title,marker_size=self.marker_size
        ) if type_ == "by_class" else big_group_plot_umap(
            axs, umap_result, target, len(self.taxonomy), title,marker_size =self.marker_size
        )
        plt.savefig('{}/2d_umap_{}.pdf'.format(self.checkpoint_src,type_))
        return fig, axs

    def _3d_umap_plot(self, umap_result, target, title, type_="by_class"):
        import matplotlib.pyplot as plt
        from src.utils.plots.umap_plots import plot_umap_3d
        fig = plt.figure(figsize=self.figsize, facecolor='black')
        ax = fig.add_subplot(111, projection="3d")
        plot_umap_3d(ax, umap_result, target, len(self.taxonomy), "UMAP 3D Visualization",marker_size=self.marker_size)
        plt.savefig('{}/3d_umap_{}.pdf'.format(self.checkpoint_src,type_))
        return fig, ax

    def _knn(self, plot_cm=True):
        from src.utils.KNNClassifier import KNNClassifier
        train_dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="train", seed=self.seed
        )
        
        val_dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="validation", seed=self.seed
        )
        test_dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="test", seed=self.seed
        )
        KNNClassifier(self._predict(train_dataloader),
                      self._predict(val_dataloader),
                      self._predict(test_dataloader), plot_cm=True)
        
class ComparePerformance:
    def __init__(self, 
                 path_1,
                 path_2,
                 path_to_dataset, 
                 model_class, 
                 model_type,  
                 custom_parse_key_str, 
                 seed = 0, 
                 device="cpu", 
                 batch_size=128, 
                 load_checkpoint=True ):
        self.taxonomy = None
        self.model_1 = ReportClassification(path_1, 
                                            path_to_dataset, 
                                            model_class, 
                                            model_type,  
                                            custom_parse_key_str, 
                                            seed = seed, 
                                            device=device, 
                                            batch_size=batch_size, 
                                            load_checkpoint=load_checkpoint )
        self.model_2 = ReportClassification(path_2, 
                                            path_to_dataset, 
                                            model_class, 
                                            model_type,  
                                            custom_parse_key_str, 
                                            seed = seed, 
                                            device=device, 
                                            batch_size=batch_size, 
                                            load_checkpoint=load_checkpoint )
    
    def compare(self, dataset_type:str= 'validation', eval_time = 2048):
        from sklearn.metrics import classification_report
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=dataset_type, seed=self.seed
        )
        preds_1, target = self.model_1._predict(dataloader)
        preds_2,_ = self.model_2._predict(dataloader)
        cr_1  = classification_report(np.argmax(preds_1, axis = -1), target, output_dict=True)
        cr_2  = classification_report(np.argmax(preds_2, axis = -1), target, output_dict=True)
        f1_1 = []
        f1_2 = []
        fig,ax = plt.subplots(1,1,figsize = (8,5))
        for key,value in cr_1.items():
            if key in self.taxonomy().keys():
                #print(key,value)
                f1_1.append(value['f1-score'])
        df_1 = pd.DataFrame({'group':self.taxonomy().keys(), 'values':f1_1 })
        ordered_df_1 = df_1#.sort_values(by='values')
        my_range_1=range(1,len(df_1.index)+1)
        for key,value in cr_2.items():
            if key in self.taxonomy().keys():
                #print(key,value)
                f1_2.append(value['f1-score'])
        df_2 = pd.DataFrame({'group':self.taxonomy().keys(), 'values':f1_2})
        ordered_df_2 = df_2#.sort_values(by='values')
        my_range_2=range(1,len(df_2.index)+1)
        import matplotlib.pyplot as plt
        plt.plot(ordered_df_1['values'].values, my_range_1, "o", alpha = 0.8,color='blue')
        plt.plot(ordered_df_2['values'].values, my_range_2, "o", alpha = 0.8,color='red')
        plt.legend(['1','2'])
        # The horizontal plot is made using the hline function
        plt.hlines(y=my_range_1, xmin=0, xmax=ordered_df_1['values'], color='blue', alpha = 0.5)
        # The horizontal plot is made using the hline function
        plt.hlines(y=my_range_2, xmin=0, xmax=ordered_df_2['values'], color='red', alpha = 0.5)
        # Add titles and axis names
        plt.yticks(my_range_1, ordered_df_1['group'])
        plt.title(f"F1-Score LC Classifier Comparison for Eval Time {eval_time} ", loc='center')
        plt.xlabel('F1-Score')
        plt.ylabel('Class')
        plt.xlim(0,1)
        plt.grid('on', )
        # Show the plot
        plt.show()







class InitDataLoaderOLD:

    def __init__(self,
                batch_size = 256,
                experiment_type =  "LC",
                data_root =  "/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/dataset.h5",
                seed =  0,
                train_apply_transform =  False,
                validation_apply_transform =  False,
                transforms =  [] , # List of transform modules
                train_key =  'training',
                validation_key =  'validation',
                test_key =  'test',
                observation_key =  'flux' ,
                observation_err_key =  'flux_err',
                time_key =  'time',
                time_alert_key =  'time_alert' ,
                mask_key = 'mask',
                mask_photometry_key = '',
                mask_detection_key = '',
                feature_key =  'feat_cols',
                metadata_key =  'metadata_feat',
                label_key =  'labels',
                train_use_sampler = False,
                train_shuffle = True,
                num_workers = 8,
                pin_memory = True,
                dataset = None):
        self.datamodule = {'dataset':
                            {'experiment_type': experiment_type,
                            'data_root': data_root,
                            'seed': seed,
                            'train_apply_transform': train_apply_transform,
                            'validation_apply_transform': validation_apply_transform,
                            'transforms': transforms,
                            'train_key': train_key,
                            'validation_key': validation_key,
                            'test_key':  test_key,
                            'observation_key': observation_key ,
                            'observation_err_key': observation_err_key,
                            'time_key': time_key,
                            'time_alert_key': time_alert_key ,
                            'mask_key': mask_key,
                            'mask_photometry_key': mask_photometry_key,
                            'mask_detection_key': mask_detection_key,
                            'feature_key': feature_key,
                            'metadata_key': metadata_key,
                            'label_key': label_key},
            'train_use_sampler': train_use_sampler,
            'train_shuffle': train_shuffle,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'batch_size': batch_size,
        }
        self.pl_datal = LitData(**self.datamodule)
         
        self.train = self.pl_datal.train_dataloader()
        self.validation = self.pl_datal.val_dataloader()
        self.test = self.pl_datal.test_dataloader()

    def set_sampler(self,use_sampler:bool = False):
        if use_sampler == True:
            self.datamodule['train_use_sampler']  = True
            self.pl_datal =  LitData(**self.datamodule)
            self.train = self.pl_datal.train_dataloader()
