import os
from typing import Dict, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchmetrics

import pytorch_lightning as pl
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR
from tqdm import tqdm  
from ....layers.selfsupervised.tabular import TabularClassifier

class LitTAB(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.gradients_ = None

        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]
        #self.model = LightCurveClassifier(**kwargs)
        self.model = TabularClassifier(**kwargs)
           
        self.warmup = 0

        self.use_lightcurves = self.general_["use_lightcurves"]
        self.use_lightcurves_err = self.general_["use_lightcurves_err"]
        self.use_metadata = self.general_["use_metadata"]
        self.use_features = self.general_["use_features"]
        metrics = torchmetrics.MetricCollection({
            'acc': torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.general_["num_classes"]),
            'f1': torchmetrics.classification.F1Score(task="multiclass", num_classes=self.general_["num_classes"], average="macro"),
            'recall': torchmetrics.classification.Recall(task="multiclass", num_classes=self.general_["num_classes"], average="macro")
        })

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='validation/')


        self.use_cosine_decay = kwargs["general"]["use_cosine_decay"]
        self.gradient_clip_val = (
            1.0 if kwargs["general"]["use_gradient_clipping"] else 0
        )

        
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
        pred = self.model(**batch_data)
    
        if pred is None:
            raise ValueError("Invalid prediction.")
 

        self.train_metrics(pred,  batch_data["labels"].long())
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)

        loss = 0

        loss_dic = {}
        for y, y_type in zip([pred], ["lc"]):
            if y is not None:
                partial_loss = F.cross_entropy(y,  batch_data["labels"].long())
                loss += partial_loss
                loss_dic.update({f"loss_train/{y_type}": partial_loss})
        
        loss_dic.update({f"loss_train/total": loss})
        self.log_dict(loss_dic)
 
        return loss
     
    def validation_step(self, batch_data, batch_idx):
        pred = self.model(**batch_data)
        if pred is None:
            raise ValueError("Invalid prediction.")


        self.valid_metrics(pred,  batch_data["labels"].long())
        self.log_dict(self.valid_metrics, on_epoch=True)

        loss = 0
        loss_dic = {}
        for y, y_type in zip([pred], ["lc"]):
            
            partial_loss = F.cross_entropy(y,  batch_data["labels"].long())
            loss += partial_loss
            loss_dic.update({f"loss_validation/{y_type}": partial_loss})

        loss_dic.update({f"loss_validation/total": loss})
        self.log_dict(loss_dic,sync_dist= True)
        return 0

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
            partial_loss = F.cross_entropy(y, y_true)
            loss += partial_loss
            loss_dic.update({f"loss_test/{y_type}": partial_loss})

        loss_dic.update({f"loss_test/total": loss})
        self.log_dict(loss_dic,sync_dist=True)

        return loss_dic

    def configure_optimizers(self):
    
        optimizer = optim.AdamW(self.parameters(),
                                lr = self.general_['lr'])  
        #constant = ConstantLR(optimizer,1)                                                                                           
        #scheduler = SequentialLR(
        #            optimizer,
        #            schedulers=[constant,constant],
        #            milestones=[self.warmup]
       #         )

        return [optimizer]#, [{'scheduler': scheduler, 'interval': 'step'}]


    def get_input_data(self, batch_data):
        input_dict = {}

        if self.use_lightcurves:
            input_dict.update(
                {
                    "data": batch_data["data"].float(),
                    "time": batch_data["time"].float(),
                    "mask": batch_data["mask"].float(),
                }
            )

        if self.use_lightcurves_err:
            input_dict.update({"data_err": batch_data["data_err"].float()})

        tabular_features = []
        if self.use_metadata:
            tabular_features.append(batch_data["metadata_feat"].float().unsqueeze(2))

        if self.use_features:
            tabular_features.append(batch_data["extracted_feat"].float().unsqueeze(2))

        if tabular_features:
            input_dict["tabular_feat"] = torch.cat(tabular_features, axis=1)

        return input_dict
