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


import matplotlib.pyplot as plt
import io


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import glob

    
class ClassifierModule(pl.LightningModule):
    def __init__(self,model,classifier, loss, load_ckpt = None, freeze_transformer = False, **kwargs):
        super().__init__()
        self.gradients_ = None

        self.model = model
        self.classifier  =classifier
        self.init_model()

        self.warmup = 0
        self.loss = loss
        self.load_ckpt = load_ckpt
        self.learning_rate = kwargs['learning_rate']
        self.freeze_transformer = freeze_transformer
        metrics = torchmetrics.MetricCollection({
            'acc': torchmetrics.classification.Accuracy(task="multiclass", num_classes=classifier.num_classes),
            'f1': torchmetrics.classification.F1Score(task="multiclass", num_classes=classifier.num_classes, average="macro"),
        'recall': torchmetrics.classification.Recall(task="multiclass", num_classes=classifier.num_classes, average="macro")
        })
        
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='validation/')
        if self.load_ckpt is not None:
            print("LOADING CKPT!!!")
            _ckpt = glob.glob(self.load_ckpt+ "*.ckpt")[0]
            checkpoint_ = torch.load(_ckpt)
            weights = OrderedDict()
            for key in checkpoint_["state_dict"].keys():
                if 'projection' in key:
                    continue
                else:
                    weights[key.replace("model.", "")] = checkpoint_["state_dict"][key]
            self.model.load_state_dict(weights, strict=True)
            print("loaded chekcpoint")
        if self.freeze_transformer:
            for param in self.model.parameters():
                param.requires_grad = False

    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                
    def gradfilter_ema(self,
        m: nn.Module,
        grads: Optional[Dict[str, torch.Tensor]] = None,
        alpha: float = 0.95,
        lamb: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        if grads is None:
            grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

        for n, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
                p.grad.data = p.grad.data + grads[n] * lamb

        return grads
    
    def on_after_backward(self) -> None:
        self.gradients = self.gradfilter_ema(m=self.model,
                                        grads = self.gradients_)

    def training_step(self, batch_data, batch_idx):
        labels = batch_data.pop('labels')
        emb = self.model(**batch_data) 
        pred = self.classifier(emb)
        if pred is None:
            raise ValueError("Invalid prediction.")

        self.train_metrics(pred,  labels.long())
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)
        loss = 0
        loss_dic = {}
        for y, y_type in zip([pred], ["lc"]):
            if y is not None:
                partial_loss = self.loss(y,  labels.long())
                loss += partial_loss
                loss_dic.update({f"loss_train/{y_type}": partial_loss})
        
        loss_dic.update({f"loss_train/total": loss})
        if batch_idx % 100 == 0:
            for harmonic in range(4):
                for enc in range(1):
                    self.logger.experiment.add_histogram(f'HARMONICS/alpha_cos_harmonics_{harmonic}',self.model.time_encoder.time_encoders[enc].alpha_cos,self.global_step)
                    self.logger.experiment.add_histogram(f'HARMONICS/alpha_sin_harmonics_{harmonic}',self.model.time_encoder.time_encoders[enc].alpha_sin,self.global_step) 
            self.logger.experiment.add_histogram(f'out_emb',emb,self.global_step)
            
        self.log_dict(loss_dic)
        return loss
    
    def validation_step(self, batch_data, batch_idx):
        labels = batch_data.pop('labels')
        emb = self.model(**batch_data) 
        pred = self.classifier(emb)
        if pred is None:
            raise ValueError("Invalid prediction.")


        self.valid_metrics(pred,  labels.long())
        self.log_dict(self.valid_metrics, on_epoch=True, sync_dist=True)

        loss = 0
        loss_dic = {}
        for y, y_type in zip([pred], ["lc"]):
            
            partial_loss = self.loss(y,  labels.long())
            loss += partial_loss
            loss_dic.update({f"loss_validation/{y_type}": partial_loss})

        loss_dic.update({f"loss_validation/total": loss})
        self.log_dict(loss_dic,sync_dist= True)
        return loss
    def test_step(self, batch_data, batch_idx):
        input_dict = self.get_input_data(batch_data)

        pred = self.model(**input_dict)
        

        if pred is None:
            raise ValueError("Invalid prediction.")

        """ labels """
        y_true = batch_data["labels"].long()

        loss = 0
        loss_dic = {}
        for y, y_type in zip([pred], ["lc"]):
            partial_loss = self.loss(y, y_true)
            loss += partial_loss
            loss_dic.update({f"loss_test/{y_type}": partial_loss})

        loss_dic.update({f"loss_test/total": loss})
        self.log_dict(loss_dic,sync_dist=True)

        return loss_dic
        if self.use_lightcurves_err:
            input_dict.update({"data_err": batch_data["data_err"].float()})

    def configure_optimizers(self):
       
        
        optimizer = optim.AdamW(self.parameters(), 
                                lr = self.learning_rate)
        constant = ConstantLR(optimizer,1)  
        cosine = CosineAnnealingWarmRestarts(optimizer,T_0=1200,eta_min=1e-5)                                         

        #cosine = CosineAnnealingWarmRestarts(optimizer,T_0=9600,eta_min=1e-5)                                         

        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[constant,constant],
                    milestones=[self.warmup]
                )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]    
