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
from pytorch_metric_learning.miners import AngularMiner

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import glob
from copy import deepcopy 
from io import BytesIO
from PIL import Image
import torchvision.transforms as T
class ClassifierModuleTriplet(pl.LightningModule):
    def __init__(self,model,classifier, loss,
                 experiment_type:str, 
                 load_ckpt = None, 
                 freeze_transformer = False, **kwargs):
        super().__init__()
        self.gradients_ = None

        self.model = model
        self.classifier  = classifier
        self.init_model()

        self.warmup = 0
        self.loss = loss
        self.load_ckpt = load_ckpt
        self.learning_rate = kwargs['learning_rate']
        self.freeze_transformer = freeze_transformer
 
        parse_exp_type = experiment_type.split('_')
        self.modalities = []
        self.modalities+= ['LC'] if 'LC' in parse_exp_type else []
        self.modalities+= ['TAB'] if 'MD' in parse_exp_type or 'FEAT' in parse_exp_type else []
        self.modalities+= ['MIX'] if ('MD' in parse_exp_type or 'FEAT' in parse_exp_type) and ('LC' in parse_exp_type) else []
        
        self.miner = AngularMiner(angle = 20)
        self.loss = nn.TripletMarginLoss()
        metrics = torchmetrics.MetricCollection({
                'acc': torchmetrics.classification.Accuracy(task="multiclass", num_classes=22),
                'f1': torchmetrics.classification.F1Score(task="multiclass", num_classes=22, average="macro"),
            'recall': torchmetrics.classification.Recall(task="multiclass", num_classes=22, average="macro"),
            })
        self.validation_cm = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=22, normalize='true')
        if 'LC' in self.modalities:
            self.LC_train_metrics = metrics.clone(prefix=f'{'training/LC/'}')
            self.LC_valid_metrics = metrics.clone(prefix=f'{'validation/LC/'}')
        if 'TAB' in self.modalities:
            self.TAB_train_metrics = metrics.clone(prefix=f'{'training/TAB/'}')
            self.TAB_valid_metrics = metrics.clone(prefix=f'{'validation/TAB/'}')
        if 'MIX' in self.modalities:
            self.MIX_train_metrics = metrics.clone(prefix=f'{'training/MIX/'}')
            self.MIX_valid_metrics = metrics.clone(prefix=f'{'validation/MIX/'}')



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
                nn.init.normal_(p,std = 0.1)
                
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
        embs = self.model(**batch_data) 
        
        
        if isinstance(embs,dict):
            preds = self.classifier(**embs)
        else:
            preds = self.classifier(embs)
        # pred can be a single tensor or a dict of tensors depending on how many modalities or classifier type
        if isinstance(preds,dict):
            loss = 0
            if 'LC' in preds.keys():
                self.LC_train_metrics(preds['LC'], labels.long())
                self.log_dict(self.LC_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                partial_loss = self.loss(anc,pos,neg)
                loss+=partial_loss
            if 'TAB' in preds.keys():
                self.TAB_train_metrics(preds['TAB'], labels.long())
                self.log_dict(self.TAB_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                partial_loss = self.loss(anc,pos,neg)
                loss+=partial_loss
            if 'MIX' in preds.keys():
                self.MIX_train_metrics(preds['MIX'], labels.long())
                self.log_dict(self.MIX_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                partial_loss = self.loss(anc,pos,neg)
                loss+=partial_loss
            self.log("loss_train/total", loss,on_step=True, on_epoch=True)
            return loss
        else:
            if 'LC' in self.modalities:
                self.LC_train_metrics(preds, labels.long())
                self.log_dict(self.LC_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                loss = self.loss(anc,pos,neg)
            if 'TAB' in self.modalities:
                self.TAB_train_metrics(preds, labels.long())
                self.log_dict(self.TAB_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                loss = self.loss(anc,pos,neg)
            if 'MIX' in self.modalities:
                self.MIX_train_metrics(preds, labels.long())
                self.log_dict(self.MIX_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                loss = self.loss(anc,pos,neg)
            self.log(f"loss_train/total",loss,on_step=True, on_epoch=True)
            return loss
        
    def on_validation_epoch_start(self):
        self.epoch_labels = None
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch_data, batch_idx):
        labels = batch_data.pop('labels')
        embs = self.model(**batch_data) 
         
        if isinstance(embs,dict):
            preds = self.classifier(**embs)
        else:
            preds = self.classifier(embs)
        # pred can be a single tensor or a dict of tensors depending on how many modalities or classifier type
        if isinstance(preds,dict):
            loss = 0
            if 'LC' in preds.keys():
                self.LC_train_metrics(preds['LC'], labels.long())
                self.log_dict(self.LC_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                partial_loss = self.loss(anc,pos,neg)
                loss+=partial_loss
            if 'TAB' in preds.keys():
                self.TAB_train_metrics(preds['TAB'], labels.long())
                self.log_dict(self.TAB_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                partial_loss = self.loss(anc,pos,neg)
                loss+=partial_loss
            if 'MIX' in preds.keys():
                self.MIX_train_metrics(preds['MIX'], labels.long())
                self.log_dict(self.MIX_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                partial_loss = self.loss(anc,pos,neg)
                loss+=partial_loss
            self.log("loss_validation/total", loss,on_step=True, on_epoch=True)
            return loss

        else:
            if 'LC' in self.modalities:
                self.LC_train_metrics(preds, labels.long())
                self.log_dict(self.LC_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                loss = self.loss(anc,pos,neg)
            if 'TAB' in self.modalities:
                self.TAB_train_metrics(preds, labels.long())
                self.log_dict(self.TAB_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                loss = self.loss(anc,pos,neg)
            if 'MIX' in self.modalities:
                self.MIX_train_metrics(preds, labels.long())
                self.log_dict(self.MIX_train_metrics, on_step=True, on_epoch=True)
                anc,pos,neg = self.miner.mine(embs,labels.long())
                loss = self.loss(anc,pos,neg)
                #self.validation_cm(preds['MIX'],labels.long())
        self.log(f"loss_validation/total",loss,on_step=False, on_epoch=True)
        
        return loss
        

    def on_validation_epoch_end(self):
        if 'MIX' in self.modalities:
            cm = self.validation_cm.compute().cpu().numpy().astype(float)
            fig = plt.figure(figsize=(12, 10)) 
            sns.heatmap(np.round(cm, decimals=2), annot=True, cmap=plt.cm.Blues, ax=fig.add_subplot(111))
            plt.xticks(ticks=range(0, 22), rotation=45, labels=ZTF_TAXONOMY().keys())
            plt.yticks(ticks=range(0, 22), rotation=45, labels=ZTF_TAXONOMY().keys())
            plt.title(f"F1-Score: {self.MIX_valid_metrics['f1'].compute().item()}")
            plt.tight_layout()

            # Convert the Matplotlib figure to a tensor
            buf = BytesIO()
            fig.savefig(buf, format='png',dpi = 100,pad_inches = 0.05) #png
            buf.seek(0)
            image = Image.open(buf)
            image_tensor = T.ToTensor()(image)  # Convert PIL image to torch tensor (C, H, W)
            self.logger.experiment.add_image('validation cm', image_tensor, self.global_step)
            plt.close(fig)  # Close the figure to free memory
        return super().on_validation_epoch_end()
    def test_step(self, batch_data, batch_idx):
        pass

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
