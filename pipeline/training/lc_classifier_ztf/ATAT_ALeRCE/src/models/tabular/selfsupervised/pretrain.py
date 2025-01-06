from src.augmentations import LightCurveTransform as LC
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict, Optional, Literal 
from src.layers.selfsupervised.tabular import TabularProjector
import pytorch_lightning as pl  
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR 

from src.losses.VICReg import VICReg 


class LitPreTrainVICREG(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        print(kwargs)
        self.gradients_ = None  
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

        self.loss = VICReg()
        self.model = TabularProjector(**self.feature_)
 
        self.warmup = 0

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
        
    def get_gradients(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        return grads
    
    def training_step(self, batch, batch_idx):
         
        batch_data,aug_batch_data= batch
        
        x_emb = self.model(**batch_data)
        aug_x_emb = self.model(**aug_batch_data)
        loss_dict = self.loss(x_emb,aug_x_emb)
        with torch.no_grad():
            loss_dict = {f'loss_train/{key}': value for key, value in loss_dict.items()}
            self.log_dict(loss_dict,on_epoch=False,on_step=True)
        loss = loss_dict['loss_train/loss']
        return loss
     
    def validation_step(self, batch, batch_idx):
        batch_data,aug_batch_data= batch

        x_emb = self.model(**batch_data)
        aug_x_emb = self.model(**aug_batch_data)
        loss_dict = self.loss(x_emb,aug_x_emb)
        with torch.no_grad():
            loss_dict = {f'loss_validation/{key}': value for key, value in loss_dict.items()}
            self.log_dict(loss_dict,on_epoch=True,on_step=False)
        return 0
    
    def test_step(self, batch, batch_idx):
        pass

        return 0
    
    def configure_optimizers(self):
       
        self.learning_rate = self.general_['lr']
        
        optimizer = optim.AdamW(self.parameters(), 
                                lr = self.learning_rate)
        constant = ConstantLR(optimizer,1)  
        cosine = CosineAnnealingWarmRestarts(optimizer,T_0=1200,eta_min=1e-5)                                         
        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[cosine,cosine],
                    milestones=[self.warmup]
                )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]