import os
from typing import Dict, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchmetrics
from collections import OrderedDict
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR
import torchmetrics.classification
from tqdm import tqdm  
from src.utils.data.AlerceDictionaries import ZTF_TAXONOMY

import matplotlib.pyplot as plt
import io
import seaborn as sns

from lion_pytorch import Lion

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import glob
from copy import deepcopy 
from io import BytesIO
from PIL import Image
import torchvision.transforms as T


        

class ClassifierModuleHier(pl.LightningModule):
    def __init__(self,model,classifier, loss,
                 experiment_type:str, 
                 lc_load_ckpt = None, 
                 tab_load_ckpt = None, 
                 freeze_lc = False,
                 freeze_tab = False,
                 report_lc = False,
                 report_tab = False,
                 report_mix = False,
                  weight_str_parse_lc= None,
                  weight_str_parse_tab=  None, **kwargs):
        super().__init__()
        self.gradients_ = None

        self.model = model
        self.classifier  = classifier
        self.init_model()
        self.warmup = 0
        self.loss = loss
        self.learning_rate = kwargs['learning_rate']
 
        parse_exp_type = experiment_type.split('_')
        self.modalities = []
        self.modalities+= ['LC'] if 'LC' in parse_exp_type else []
        self.modalities+= ['TAB'] if 'MD' in parse_exp_type or 'FEAT' in parse_exp_type else []
        self.modalities+= ['MIX'] if ('MD' in parse_exp_type or 'FEAT' in parse_exp_type) and ('LC' in parse_exp_type) else []
        
        self.init_metrics(report_lc, report_tab, report_mix) 

        if lc_load_ckpt is not None:
            _ckpt = glob.glob(lc_load_ckpt+ "*.ckpt")[-1]
            checkpoint_ = torch.load(_ckpt)
            weights = OrderedDict()
            for key in checkpoint_["state_dict"].keys():
                if 'loss' in key:
                    continue
                if 'tab' in key:
                    continue
                else:
                    #print(key)
                    weights[key.replace(f'{weight_str_parse_lc[0]}', f"{weight_str_parse_lc[1]}")] = checkpoint_["state_dict"][key]
            if 'LC' in self.modalities:
                self.model.load_state_dict(weights, strict=True)
                print(f"loaded LC checkpoint {_ckpt}".format(_ckpt))
            if "TAB" in self.modalities:
                self.model.transformer_tab.load_state_dict(weights, strict=True)
                print(f"loaded TAB checkpoint {_ckpt}".format(_ckpt))
            if freeze_lc:
                for name,param in self.model.transformer_lc.named_parameters():
                    if 'time_encoder' in key:
                        print('froze:',key)
                        param.requires_grad = False
            

        if tab_load_ckpt is not None:
            print("LOADING CKPT!!!")
            _ckpt = glob.glob(tab_load_ckpt+ "*.ckpt")
            checkpoint_ = torch.load(_ckpt[-1])
            weights = OrderedDict()
            for key in checkpoint_["state_dict"].keys():
                if 'loss' in key:
                    continue
                if 'lc' in key:
                    continue
                else:
                    #print(key)
                    weights[key.replace(f'{weight_str_parse_tab}', "")] = checkpoint_["state_dict"][key]
            self.model.transformer_tab.load_state_dict(weights, strict=True)
            if freeze_tab:
                for name,param in self.model.transformer_tab.named_parameters():
                    param.requires_grad = False
            print("loaded TAB checkpoint")
            
        
    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
       # input()
        

    def training_step(self, batch_data, batch_idx):
        labels = batch_data.pop('labels')
        embs = self.model(**batch_data) 
        preds = self.classifier(embs)
        loss = 0

        
        partial_loss = self.loss(preds,  labels.long())
        loss+=partial_loss
        class_preds = self.loss.modify_logits(preds[0],preds[1])
        #class_preds = preds[1]
        self.LC_train_metrics(class_preds, labels.long())
        self.log_dict(self.LC_train_metrics, on_step=True, on_epoch=True)
        
      
        self.log("loss_train/total", loss,on_step=True, on_epoch=True, sync_dist=True)
        return loss
        
    def on_validation_epoch_start(self):
        self.epoch_labels = None
        return super().on_validation_epoch_start()
     

    
    def validation_step(self, batch_data, batch_idx):
        labels = batch_data.pop('labels')
        embs = self.model(**batch_data) 
        preds = self.classifier(embs)
        loss = 0 
        partial_loss = self.loss(preds,  labels.long())
        loss+=partial_loss
        #coherence
        class_preds = self.loss.modify_logits(preds[0],preds[1])
        #class_preds = preds[1]

        self.LC_valid_metrics(class_preds, labels.long())
        self.log_dict(self.LC_valid_metrics, on_step=False, on_epoch=True)
        self.validation_cm(class_preds,labels.long())
         
        self.log(f"loss_validation/total",loss,on_step=False, on_epoch=True, sync_dist=True)
        return loss
        

    def on_validation_epoch_end(self):

        cm = self.validation_cm.compute().cpu().numpy().astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm, nan=0.0)
        fig = plt.figure(figsize=(12, 10))

        sns.heatmap(np.round(cm, decimals=2), annot=True, cmap=plt.cm.Blues, ax=fig.add_subplot(111))
        plt.xticks(ticks=range(0, 22), rotation=45, labels=ZTF_TAXONOMY().keys())
        plt.yticks(ticks=range(0, 22), rotation=45, labels=ZTF_TAXONOMY().keys())

        if 'LC' in self.modalities:
            plt.title(f"F1-Score: {self.LC_valid_metrics['f1_macro'].compute().item()}")
        elif 'MIX' in self.modalities:
            plt.title(f"F1-Score: {self.MIX_valid_metrics['f1_macro'].compute().item()}")

        plt.tight_layout()

        # Convert the Matplotlib figure to a tensor
        buf = BytesIO()
        fig.savefig(buf, format='png',dpi = 100,pad_inches = 0.05) #png
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = T.ToTensor()(image)  # Convert PIL image to torch tensor (C, H, W)
        self.logger.experiment.add_image('validation cm', image_tensor, self.global_step)
        plt.close(fig)  # Close the figure to free memory
        self.validation_cm.reset()
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        #optimizer = optim.AdamW([
           # {'params': self.model.transformer_tab.parameters(), 'lr': 1e-5},  # low learning rate
            #{'params': self.model.transformer_lc.parameters(), 'lr': 1e-3}       # higher learning rate
        #])c

       
        optimizer = Lion(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        constant = ConstantLR(optimizer,1)  
        cosine = CosineAnnealingWarmRestarts(optimizer,T_0=100,eta_min=1e-6)                                         

        #cosine = CosineAnnealingWarmRestarts(optimizer,T_0=9600,eta_min=1e-5)                                         

        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[constant,constant],
                    milestones=[self.warmup]
                )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]    

    def init_metrics(self,report_lc, report_tab, report_mix):
        metrics = torchmetrics.MetricCollection({
                'acc': torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.classifier.num_classes),
                

                'f1_macro': torchmetrics.classification.F1Score(task="multiclass", num_classes=self.classifier.num_classes, average="macro"),
                'f1_weighted': torchmetrics.classification.F1Score(task="multiclass", num_classes=self.classifier.num_classes, average="weighted"),
            'recall': torchmetrics.classification.Recall(task="multiclass", num_classes=self.classifier.num_classes, average="macro"),
            'precision': torchmetrics.classification.Precision(task="multiclass", num_classes=self.classifier.num_classes, average="macro"),
            })
        self.validation_cm = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=22, normalize=None)
        
        if 'LC' in self.modalities:
            self.LC_train_metrics = metrics.clone(prefix=f'{'training/LC/'}')
            self.LC_valid_metrics = metrics.clone(prefix=f'{'validation/LC/'}')
        if 'TAB' in self.modalities:
            self.TAB_train_metrics = metrics.clone(prefix=f'{'training/TAB/'}')
            self.TAB_valid_metrics = metrics.clone(prefix=f'{'validation/TAB/'}')
        if 'MIX' in self.modalities:
            self.MIX_train_metrics = metrics.clone(prefix=f'{'training/MIX/'}')
            self.MIX_valid_metrics = metrics.clone(prefix=f'{'validation/MIX/'}')


    def report(self, report_lc, report_tab, report_mix):
        pass