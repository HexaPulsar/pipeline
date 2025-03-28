from src.augmentations import LightCurveTransform as LC
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict, Optional, Literal
from src.layers.selfsupervised.multimodal import ATATProjector
import pytorch_lightning as pl  
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR, LinearLR

from src.losses.VICReg import VICReg 


class LitPreTrainLCMD(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.gradients_ = None  
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["tab"]

        self.loss = VICReg(15,35,1)
        self.model = ATATProjector(**kwargs)
        self.warmup = 0

    def training_step(self, batch, batch_idx):
        batch_data= batch
        x_emb,f_emb = self.model(**batch_data)
        #aug_x_emb,aug_f_emb = self.model(**aug_batch_data)
        loss_dict = self.loss(x_emb,f_emb)
        with torch.no_grad():
            loss_dict = {f'loss_train/{key}': value for key, value in loss_dict.items()}
            self.log_dict(loss_dict,on_epoch=False,on_step=True)
        loss = loss_dict['loss_train/loss']
        return loss
    
     
    def validation_step(self, batch, batch_idx):
        batch_data= batch
        x_emb,f_emb = self.model(**batch_data)
        #aug_x_emb = self.model(**aug_batch_data)
        loss_dict = self.loss(x_emb,f_emb)
        with torch.no_grad():
            loss_dict = {f'loss_validation/{key}': value for key, value in loss_dict.items()}
            self.log_dict(loss_dict,on_epoch=True,on_step=False)
        return 0
    
    def test_step(self, batch, batch_idx):
        pass

        return 0
    def configure_optimizers(self):
        self.warmup = 336*10
        self.learning_rate = self.general_['lr']
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Create a linear warmup scheduler starting from 1e-7
        
        #warmup = LinearLR(optimizer, 
        #                start_factor=1e-5/self.learning_rate,  # Start from 1e-7
        #                end_factor=1.0,
        #                total_iters=self.warmup)

        # Keep your cosine annealing scheduler
        cosine = CosineAnnealingWarmRestarts(optimizer, T_0=336*5, eta_min=1e-5)
        constant = ConstantLR(optimizer,1)  

        # Create sequential scheduler with warmup followed by cosine
        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[constant,constant],
                    milestones=[self.warmup]
                )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]