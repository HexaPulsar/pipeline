
import numpy as np
from src.layers.transformer.ATAT import TabularTransformer, Combinator
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
import glob

from torch import device, load
from collections import OrderedDict

import yaml
from  hydra.utils import instantiate
from .utils import get_confusion_matrix

class InitCombinator:
    def __init__(
        self,
        path_to_config_yaml,
        lc_model,
        tab_model,
        classifier,
        classifier_args,
        arg_key,
        device = 'cpu'):
        self.path_to_config_yaml = path_to_config_yaml
        self.device = device
        self.args = self._load_yaml_args(path_to_config_yaml).ATATConfig
        self.classifier = classifier(lc_input_size = self.args.lc.embedding_size,
                                     inner_size = self.args.lc.embedding_size,
                                     combine_logits = True,
                 tab_input_size = self.args.tab.embedding_size,
                 num_classes = self.args.num_classes
                 ,**classifier_args)

        lc_model = lc_model(**self.args["lc"])
        tab_model = tab_model(**self.args["tab"])
        lc_od = self.create_ordered_dict(remove_if_in_key_list=['projection', 'transformer_tab', 'classifier'],rename_keys = ('model.transformer_lc.',''), checkpoint_name='classifier_ckpt')
        tab_od = self.create_ordered_dict(remove_if_in_key_list=['projection', 'transformer_lc', 'classifier'],rename_keys = ('model.transformer_tab.',''), checkpoint_name='classifier_ckpt')
        self.load_weights(lc_model, lc_od)
        self.load_weights(tab_model, tab_od)
        classifier_od = self.create_ordered_dict(remove_if_in_key_list=['projection', 'model'],rename_keys = ('classifier.',''), checkpoint_name='classifier_ckpt')
        self.load_weights(self.classifier, classifier_od)
        self.atat = Combinator(lc_model, tab_model)


    @staticmethod
    def _load_yaml_args(path_to_config_yaml):
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
                checkpoint_path_clip[-1], map_location=device(self.device),
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                if print_keys:
                    print('old key name:',key)
                if any([remove_if_in_key in key for remove_if_in_key in remove_if_in_key_list ]):
                    continue
                od_atat[key.replace(f"{rename_keys[0]}", f"{rename_keys[1]}",1)] = checkpoint_clip[
                    "state_dict"
                ][key]
            return od_atat

    def load_weights(self,model,weights: dict, strict = True):
        model.load_state_dict(weights, strict=strict)
        print("Loaded backbone weights")

    def predict(self,dataloader, device = None, pred_type = 'class'):
        if device is not None:
            self.device = device
            print("device set to {}".format(device))
        target = None
        preds_out = None
       # print(self.atat)
        self.atat.eval().to(device=self.device)
        self.classifier.eval().to(device=self.device)

        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            emb = self.atat(**b1)
            if pred_type == 'class':

                emb = self.classifier(emb)
                if isinstance(emb, dict):
                    if 'LC' in emb.keys():
                        output = emb["LC"]
                    if 'TAB' in emb.keys():
                        output = emb["TAB"]
                    if 'MIX' in emb.keys():
                        output = emb["MIX"]

            elif pred_type == 'embeddings':
                output = emb

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
        self.atat.to(device="cpu")
        self.classifier.to(device="cpu")
        return preds_out, target
    def get_confusion_matrix(self, preds,target, taxonomy, dataset_type:str, plot_title,  order_classes  ):
        return get_confusion_matrix(preds,target,taxonomy, dataset_type,plot_title, order_classes)
