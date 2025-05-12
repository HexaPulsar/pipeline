import numpy as np
import pandas as pd
from tqdm import tqdm 
import glob
from copy import deepcopy

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
                metadata_key =  'metadata_cols',
                label_key =  'labels'):
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
                            'feature_key': feature_key,
                            'metadata_key': metadata_key,
                            'label_key': label_key},
            'train_use_sampler': True,
            'train_shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            'batch_size': batch_size,
        }
        pl_datal = LitData(**datamodule)
         
        self.train = pl_datal.train_dataloader()
        self.validation = pl_datal.val_dataloader()
        self.test = pl_datal.test_dataloader()


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
        self, path_to_dataset, set_type="test", seed=0,  
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
        
        pl_datal = LitData(**modded_config.datamodule)
        if set_type == "train":
            dataloader = pl_datal.train_dataloader()
        elif set_type == "validation":
            dataloader = pl_datal.val_dataloader()
        elif set_type == "test":
            dataloader = pl_datal.test_dataloader()
        return dataloader
    
    def _predict(self,dataloader):
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
        from src.layers.ClassifierBaseModel import ClassifierBaseModel
        from src.layers.classifiers import TokenClassifier
        from torch import device, load
        from collections import OrderedDict
        model = model(**self.cfg.lc) if self.model_type =='lc' else model(**self.cfg.tab)
        classifier = TokenClassifier(num_classes=self.cfg.num_classes,**self.cfg.lc )
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
                if 'projection' in key:
                    continue
                od_atat[key.replace(f"{self.custom_parse_key_str}", "")] = checkpoint_clip[
                    "state_dict"
                ][key]
            model.load_state_dict(od_atat, strict=True)
        else:
            print('NO CKPT LOADED')
        return model
    
    def classification_report(self, dataset_type: str = 'validation', digits = 4, confusion_matrix = True):
        from sklearn.metrics import classification_report
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=dataset_type, seed=self.seed
        )
        preds_out, target = self._predict(dataloader)
        preds_out = np.argmax(preds_out, axis = -1)
        classification = classification_report(target,preds_out, target_names=list(self.taxonomy().keys()),digits = digits)
        print(classification)
        if confusion_matrix:
            self.get_confusion_matrix(preds_out,target, title = 'Classifier Results [{}]'.format(dataset_type.upper()))
        return classification_report(target,preds_out, target_names=list(self.taxonomy().keys()),digits = digits, output_dict=True)
    def get_confusion_matrix(self, preds,target, title = ''):
        from src.utils.plots.ATATConfusionMatrix import elasticc_confusion_matrix
        from sklearn.metrics import classification_report
        import matplotlib.pyplot as plt

        out_metrics_balto = classification_report(
            target, preds,target_names=list(self.taxonomy().keys()), output_dict=True
        )["macro avg"]
        template_balto = ""
        for key in out_metrics_balto.keys():
            template_balto += " {} : {:.3f} ".format(key.upper(), out_metrics_balto[key])
        fig, axes = plt.subplots(1,1,figsize = (15,15))
        return elasticc_confusion_matrix(
            y_true=np.array(target).astype(int),
            y_pred=np.array(preds).astype(int),
            classes= np.array(list(self.taxonomy().keys())),
            ax=axes,
            normalize=True,
            title=f"{template_balto}",
        )
