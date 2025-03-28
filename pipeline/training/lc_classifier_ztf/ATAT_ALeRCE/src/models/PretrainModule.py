from src.augmentations import LightCurveTransform as LC
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict, Optional, Literal
import pytorch_lightning as pl  
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR, LinearLR
import logging

class PretrainModule(pl.LightningModule):
    def __init__(self,model,loss,lr = 0.001, **kwargs):
        """
        Batch must be a tuple of batches: (batch, augmented batch)
        Model output must be of shape (bsz,embeddings)
        Loss input is (embedding_batch_1, embedding_batch_2)
        Args:
            model (_type_): _description_
            loss (_type_): _description_
           
        """
        super().__init__()
        self.gradients_ = None
        self.lr = lr
        self.model = model
        self.loss = loss
        self.init_model()
        logging.debug('using learning rate {}'.format(self.lr))
    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                
    def training_step(self, batch, batch_idx):
        loss_dict = self.loss(self.model(**batch[0]),self.model(**batch[1]))
        with torch.no_grad():
            self.log_dict({f'loss_train/{key}': value for key, value in loss_dict.items()},on_epoch=False,on_step=True)
            #self.add_histogram('alpha_cos_harmonics_mean',self.model.time_encoder.time_encoders[0].alpha_cos.mean(dim = 0),on_epoch=False,on_step=True)
            #self.add_histogram('alpha_sin_harmonics_mean',self.model.time_encoder.time_encoders[0].alpha_sin.mean(dim = 0),on_epoch=False,on_step=True) 
        return loss_dict['loss']
     
    def validation_step(self, batch, batch_idx):
        loss_dict = self.loss( self.model(**batch[0]),self.model(**batch[1]))
        with torch.no_grad():
            self.log_dict({f'loss_validation/{key}': value for key, value in loss_dict.items()},on_epoch=True,on_step=False)
        return loss_dict['loss']
    
    def test_step(self, batch, batch_idx):
        return 0
    def configure_optimizers(self):
        warmup = 0
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        cosine = CosineAnnealingWarmRestarts(optimizer, T_0=int(1e4), eta_min=2e-3)
        constant = ConstantLR(optimizer,1)  
        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[cosine,cosine],
                    milestones=[warmup]
                )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]