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
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR, LinearLR
import torchmetrics.classification
import torchmetrics.classification.precision_recall_curve
from tqdm import tqdm  
from src.utils.data.AlerceDictionaries import ELASTICC_TAXONOMY, ZTF_TAXONOMY

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
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
 
import pandas as pd

class ClassifierModule(pl.LightningModule):
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
                  weight_str_parse_tab=  None,
                   **kwargs):
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

            #self.model.transformer_lc.load_state_dict(weights, strict=True)
            self.model.load_state_dict(weights, strict=True)
            print(f"loaded LC checkpoint {_ckpt}".format(_ckpt))
        if freeze_lc:
            for name,param in self.model.transformer_lc.named_parameters():
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
                #if 'alpha_' in name:
                #        nn.init.uniform_(p,0,1)
     
    def training_step(self, batch_data, batch_idx):
        labels = batch_data.pop('labels')
        embs = self.model(**batch_data) 
        preds = self.classifier(embs)
        loss = 0

        if 'LC' in preds.keys():
            partial_loss = self.loss(preds['LC'],  labels.long())
            loss+=partial_loss
            self.LC_train_metrics(preds['LC'], labels.long())
           # hier_map_preds = self.map_label_tensor(torch.argmax(preds['LC'],dim = -1))
           # hier_map_labels = self.map_label_tensor(labels.long())
            #self.f1_hier_macro_val(hier_map_preds, hier_map_labels)
           # self.log('training/f1_score_harmonic',2 / (self.LC_train_metrics['f1_macro'].compute()**-1 + self.f1_hier_macro_train(hier_map_preds, hier_map_labels)**-1), on_step=True, sync_dist=True)
            self.log_dict(self.LC_train_metrics, on_step=False, on_epoch=True)
            self.log(f"loss_train/lc",partial_loss,on_step=False, on_epoch=True, sync_dist=True)
            
        if 'TAB' in preds.keys():
            
            self.TAB_train_metrics(preds['TAB'], labels.long())
            self.log_dict(self.TAB_train_metrics, on_step=False, on_epoch=True)
            partial_loss = self.loss(preds['TAB'],  labels.long())
            loss+=partial_loss
            self.log(f"loss_train/tab",partial_loss,on_step=False, on_epoch=True, sync_dist=True)

        if 'MIX' in preds.keys():
            self.MIX_train_metrics(preds['MIX'], labels.long())
            self.log_dict(self.MIX_train_metrics, on_step=False, on_epoch=True)
            partial_loss = self.loss(preds['MIX'],  labels.long())
            loss+=partial_loss
            self.log(f"loss_train/mix",partial_loss,on_step=False, on_epoch=True, sync_dist=True)

        #loss = loss / len(self.modalities)
        self.log("loss_train/total", loss,on_step=True, on_epoch=True, sync_dist=True)
       #self.log("logit_scale", self.classifier.temp.item(),on_step=True, on_epoch=True, sync_dist=True)
        return loss
        
    def on_validation_epoch_start(self):
        self.epoch_labels = None
        return super().on_validation_epoch_start()
    

    def infer_at_time(self,batch_data,time):
        labels = batch_data.pop('labels')
        batch_data['time'] = batch_data* (batch_data['time'].max()<time)
        batch_data['data'] = batch_data* (batch_data<time)
        embs = self.model(**batch_data) 
        preds = self.classifier(embs)


    def validation_step(self, batch_data, batch_idx):
        labels = batch_data.pop('labels')
        embs = self.model(**batch_data) 
        preds = self.classifier(embs)
        loss = 0
        
        
        if 'LC' in preds.keys():
            partial_loss = self.loss(preds['LC'],  labels.long())
            loss+=partial_loss
            self.LC_valid_metrics(preds['LC'], labels.long())
            self.log_dict(self.LC_valid_metrics, on_step=False, on_epoch=True)
            self.validation_cm(preds['LC'],labels.long())
            #print(torch.argmax(preds['LC'],dim = -1).shape)
           # hier_map_preds = self.map_label_tensor(torch.argmax(preds['LC'],dim = -1))
           # hier_map_labels = self.map_label_tensor(labels.long())

           # df = pd.DataFrame({'preds',preds["LC"].detach().numpy(),
           #                    'preds_hier',hier_map_preds.detach().numpy(),
           #                    'labels_hier', hier_map_labels.detach().numpy(),
           #                     'labels',labels.detach().numpy()})
           # df.query('labels_hier == 0')
            self.log(f"loss_validation/lc",partial_loss,on_step=False, on_epoch=True, sync_dist=True)

            #self.log('validation/f1_score_harmonic',2 / (self.LC_valid_metrics['f1_macro'].compute()**-1 + self.f1_hier_macro_val(hier_map_preds, hier_map_labels)**-1), on_step=False, on_epoch=True, sync_dist=True)
            #self.log('validation/f1_hier_macro',self.f1_hier_macro_val(hier_map_preds, hier_map_labels), on_step=False, on_epoch=True)
           #self.log(, on_step=False, on_epoch=True)
        if 'TAB' in preds.keys():
            self.TAB_valid_metrics(preds['TAB'], labels.long()) 
            self.log_dict(self.TAB_valid_metrics, on_step=False, on_epoch=True)
            partial_loss = self.loss(preds['TAB'],  labels.long())
            loss+=partial_loss
            self.validation_cm(preds['TAB'],labels.long())
           # hier_map_preds = self.map_label_tensor(torch.argmax(preds['TAB'],dim = -1))
           # hier_map_labels = self.map_label_tensor(labels.long())
            self.log(f"loss_validation/tab",partial_loss,on_step=False, on_epoch=True, sync_dist=True)

            #self.log('validation/f1_score_harmonic',2 / (self.TAB_valid_metrics['f1_macro'].compute()**-1 + self.f1_hier_macro_val(hier_map_preds, hier_map_labels)**-1), on_step=False, on_epoch=True, sync_dist=True)
        if 'MIX' in preds.keys():
            self.MIX_valid_metrics(preds['MIX'], labels.long())
            self.log_dict(self.MIX_valid_metrics, on_step=False, on_epoch=True)
            partial_loss = self.loss(preds['MIX'],  labels.long())
            loss+=partial_loss
            self.epoch_labels = (
            torch.concat([self.epoch_labels, labels.detach()])
            if self.epoch_labels is not None
            else labels.detach()
            )
            self.validation_cm(preds['MIX'],labels.long())
            self.log(f"loss_validation/mix",partial_loss,on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"validation_pr_diff",(self.MIX_valid_metrics['precision'].compute() -self.MIX_valid_metrics['recall'].compute() ),on_step=False, on_epoch=True, sync_dist=True)
        
        #loss = loss / len(self.modalities)
        self.log(f"loss_validation/total",loss,on_step=False, on_epoch=True, sync_dist=True)
        return loss
        

    def on_validation_epoch_end(self):
        tax = ELASTICC_TAXONOMY()
        cm = self.validation_cm.compute().cpu().numpy().astype(float)
        fig = plt.figure(figsize=(12, 10)) 
        
        sns.heatmap(np.round(cm, decimals=2), annot=True, cmap=plt.cm.Blues, ax=fig.add_subplot(111))
        plt.xticks(ticks=range(0, self.classifier.num_classes), rotation=45, labels=tax.keys())
        plt.yticks(ticks=range(0, self.classifier.num_classes), rotation=45, labels=tax.keys())

        
        if len(self.modalities) == 3:
            plt.title(f"F1-Score: {self.MIX_valid_metrics['f1_macro'].compute().item()}")
        else:
            if 'LC' in self.modalities:
                plt.title(f"F1-Score: {self.LC_valid_metrics['f1_macro'].compute().item()}")
            elif 'TAB' in self.modalities:
                plt.title(f"F1-Score: {self.TAB_valid_metrics['f1_macro'].compute().item()}")

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
        self.warmup = 100
        optimizer = Lion(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)

        constant = ConstantLR(optimizer,1)  
        #cosine = CosineAnnealingWarmRestarts(optimizer,T_0=100,eta_min=1e-6)                                         
        linear = LinearLR(optimizer, start_factor=1e-2, total_iters=self.warmup)
        cosine = CosineAnnealingWarmRestarts(optimizer,T_0=self.warmup,eta_min=1e-5)                                         

        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[linear,cosine],
                    milestones=[self.warmup]
                )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]    
    
    def map_label_tensor(self,labels):
        mapping_dict = {
            0: 1, 1: 1, 3: 1, 5: 1, 8: 1,
            2: 2, 6: 2, 7: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2,
            4: 0, 9: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0
        }
        mapping_tensor = torch.tensor([mapping_dict.get(int(label), -1) for label in labels], device = labels.device)
        return mapping_tensor
    
    def init_metrics(self,report_lc, report_tab, report_mix):
        thr = 0.5
        metrics = torchmetrics.MetricCollection({
                'acc': torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.classifier.num_classes, threshold=thr),
                

                'f1_macro': torchmetrics.classification.F1Score(task="multiclass", num_classes=self.classifier.num_classes, average="macro", threshold=thr),
                'f1_weighted': torchmetrics.classification.F1Score(task="multiclass", num_classes=self.classifier.num_classes, average="weighted", threshold=thr),
            'recall': torchmetrics.classification.Recall(task="multiclass", num_classes=self.classifier.num_classes, average="macro", threshold=thr),
            'precision': torchmetrics.classification.Precision(task="multiclass", num_classes=self.classifier.num_classes, average="macro", threshold=thr),
            })
        self.validation_cm = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=self.classifier.num_classes, normalize='true', threshold=thr)
       # self.f1_hier_macro_val =  torchmetrics.classification.F1Score(task="multiclass", num_classes=3, average="macro", threshold=thr)
       # self.f1_hier_macro_train =  torchmetrics.classification.F1Score(task="multiclass", num_classes=3, average="macro", threshold=thr)
        #self.f1_8 =  torchmetrics.classification.F1Score(task="multiclass", num_classes=3, average="macro")
        #self.prcurve = torchmetrics.classification.MulticlassPrecisionRecallCurve(num_classes=self.classifier.num_classes,average = 'macro')
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
    

    def get_confusion_matrix(self,preds,
                            target, 
                            taxonomy,
                            dataset_type:str, 
                            plot_title:str, 
                            order_classes:list[str]):
            

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
            ax.set_title(f'{plot_title}: {dataset_type} | macro f1: {np.round(f1_,4)}', fontsize=16, pad=13)
            ax.set_xlabel('Predicted label', fontsize=16, labelpad=13)  # Label del eje x
            ax.set_ylabel('True label', fontsize=16, labelpad=13)        # Label del eje y

            #ax.xaxis.label.set_size(16)
            #ax.yaxis.label.set_size(16)
            #ax.xaxis.labelpad = 13
            #ax.yaxis.labelpad = 13
            return fig