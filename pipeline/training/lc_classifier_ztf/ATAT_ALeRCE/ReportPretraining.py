
from src.data.modules.LitData import LitData
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm
import warnings
import time
from typing import Union, Optional

warnings.filterwarnings("ignore")
import glob
import os
from collections import OrderedDict
from src.layers.classifiers import TokenClassifier, MixedClassifier
import yaml
import matplotlib.pyplot as plt
from matplotlib import cm
from src.utils.plots.umap_plots import *
import umap
from src.utils.data.AlerceDictionaries import ALERCE_TAXONOMY
from src.utils.KNNClassifier import KNNClassifier
from src.utils.CustomParser import ATATConfig
from copy import deepcopy
from  hydra.utils import instantiate

class ReportPretraining:
    def __init__(
        self,
        path_to_training_dir: str,
        path_to_dataset: str,
        model_class,
        model_type: str,
        custom_parse_key_str:str,
        seed:int = 0,
        device="cpu",
        batch_size=128,
        load_checkpoint = True,
        figsize = (20,20),
        umap_args={"n_neighbors": 15, "min_dist": 0.25, "metric": "euclidean"},
        marker_size = 4
    ):
        self.cfg = self._load_yaml_args(path_to_training_dir).ATATConfig
        print(self.cfg)
        self.class_dict = ALERCE_TAXONOMY.all_classes
        self.path_to_dataset = path_to_dataset
        self.figsize =figsize
        self.marker_size = marker_size
        self.device = device
        self.load_checkpoint = load_checkpoint
        self.umap_args = umap_args
        self.seed = seed
        self.custom_parse_key_str = custom_parse_key_str
        self.model_type = model_type
        self.checkpoint_src = path_to_training_dir
        self.model = self._init_model(model_class)
        
    @staticmethod
    def _load_yaml_args(path_args):
        path_args = glob.glob(f"{path_args}/.hydra/*config*")[0]
        with open(path_args, "r") as file:
            args = yaml.safe_load(file)
        return instantiate(args)

    def generate_umap_plots(self):
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="validation", seed=self.seed
        )
        self.preds_out, self.target = self._predict(dataloader)
        umap_result = self._umap(n_components=2, **self.umap_args)
        self._2d_umap_plot(umap_result, self.target, "by_class", type="by_class")
        plt.show() 
        self._2d_umap_plot(umap_result, self.target, "main_type", type="main_type")
        plt.show()
        umap_result = self._umap(n_components=3, **self.umap_args)
        self._3d_umap_plot(umap_result, self.target, "3d", type="by_class")
        plt.show()

    def _init_dataloader(
        self, path_to_dataset, set_type="test", seed=0, batch_size=128
    ):
        dataset_exclude_transform_keys ={
            key: value 
            for key, value in self.cfg.datamodule.dataset.items() 
            if key not in ['transforms_1','transforms_2', 'data_root']
        }
         
        modded_config = deepcopy(self.cfg)
        modded_config.datamodule.dataset = dataset_exclude_transform_keys
        modded_config.datamodule.dataset.data_root = self.path_to_dataset
        
        pl_datal = LitData(**modded_config.datamodule)
        if set_type == "train":
            dataloader = pl_datal.train_dataloader()
        if set_type == "validation":
            dataloader = pl_datal.val_dataloader()
        if set_type == "test":
            dataloader = pl_datal.test_dataloader()
        return dataloader

    def _init_model(self,model ):
        model = model(**self.cfg.lc) if self.model_type =='lc' else model(**self.cfg.tab)
        if self.checkpoint_src is not None or self.load_checkpoint:
            print(glob.glob(self.checkpoint_src))
            checkpoint_path_clip = glob.glob(f"{self.checkpoint_src}*pretrain_ckpt*")
            print('using checkpoint {}'.format(checkpoint_path_clip[-1].split('=')[-1]))
            checkpoint_clip = torch.load(
                checkpoint_path_clip[-1], map_location=torch.device(self.device)
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

    def _umap(self, n_components, metric, min_dist, n_neighbors):
        print(f"creating {n_components}D UMAP reduction for data...")
        umap_model = umap.UMAP(
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
        )
        umap_result = umap_model.fit_transform(self.preds_out)
        return umap_result

    def _2d_umap_plot(self, umap_result, target, title, type="by_class"):
        fig, axs = plt.subplots(1, 1, figsize=self.figsize)
        plot_umap(
            axs, umap_result, target, len(ALERCE_TAXONOMY), title,marker_size=self.marker_size
        ) if type == "by_class" else big_group_plot_umap(
            axs, umap_result, target, len(ALERCE_TAXONOMY), title,marker_size =self.marker_size
        )
        return fig, axs

    def _3d_umap_plot(self, umap_result, target, title, type="by_class"):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")
        plot_umap_3d(ax, umap_result, target, len(ALERCE_TAXONOMY), "UMAP 3D Visualization",marker_size=self.marker_size)
        return fig, ax

    def _knn(self, plot_cm=True):
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
        