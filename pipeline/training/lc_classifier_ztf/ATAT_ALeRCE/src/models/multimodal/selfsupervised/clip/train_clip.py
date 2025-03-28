from src.augmentations import LightCurveTransform as LC
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict, Optional, Literal
from src.layers.selfsupervised.multimodal import ATATProjector
import pytorch_lightning as pl  
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR, LinearLR

from src.losses.CYCLIP import CyCLIP


class CLIPLCMD(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.gradients_ = None  
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["tab"]

        self.model = ATATProjector(**kwargs)
        self.warmup = 0
        self.loss = CyCLIP(
                    config_dict= {'cross_modal':True,
                    'in_modal':True,
                    'cylambda_1': 0.025,
                    'cylambda_2':0.025},
                    config_logger = {
                        
                        'cross_contrastive_lc':False,
                        'cross_contrastive_ft':False,
                        "cross_contrastive_loss":  True,
                        "cross_cyclic_loss_l2": True,
                        
                        'inmodal_contrastive_lc':False,
                        'inmodal_contrastive_ft':False,
                        "inmodal_contrastive_loss":  False,
                        "inmodal_cyclic_loss_l1": True,

                        "total_cyclic_loss": False,  
                        "total_contrastive_loss": False,
                        "total_loss": True,
                        })
    def training_step(self, batch_data, batch_idx):
        batch_data,aug_batch_data = batch_data 
         
         
        x_emb,f_emb = self.model(**batch_data)
        x_emb_aug,f_emb_aug = self.model(**aug_batch_data)
        x_emb = torch.cat([x_emb,x_emb_aug],dim=0)
        f_emb = torch.cat([f_emb,f_emb_aug],dim=0)
        cyclip_dict = self.loss(x_emb,f_emb,self.model)
        loss = cyclip_dict['total_loss']
        self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data,0,4.605) 
        
        # Create the cyclip_dict as before
        cyclip_dict = {f'loss_train/{key}': value for key, value in cyclip_dict.items()}
        cyclip_dict.update({"temp_value": self.model.logit_scale.item()})
       
        # Log all metrics in cyclip_dict
        self.log_dict(cyclip_dict, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch_data, batch_idx):
        batch_data,aug_batch_data = batch_data 
        
        x_emb,f_emb = self.model(**batch_data)
        x_emb_aug,f_emb_aug = self.model(**aug_batch_data)
        x_emb = torch.cat([x_emb,x_emb_aug],dim=0)
        f_emb = torch.cat([f_emb,f_emb_aug],dim=0)
        cyclip_dict = self.loss(x_emb,f_emb,self.model)
        loss = cyclip_dict['total_loss']
        self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data,0,4.605) 
        # Create the cyclip_dict as before
        cyclip_dict = {f'loss_validation/{key}': value for key, value in cyclip_dict.items()}
        # Log all metrics in cyclip_dict
        self.log_dict(cyclip_dict, on_step=False, on_epoch=True)
 
        return loss
    
    def test_step(self, batch_data, batch_idx):
        pass

    def configure_optimizers(self):
        #params = filter(lambda p: p.requires_grad, self.parameters())

        self.learning_rate = self.general_['lr']
        
        optimizer = optim.AdamW(self.parameters(), 
                                lr = self.learning_rate)      
        constant = ConstantLR(optimizer,1)                                                                                           
        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[constant,constant],
                    milestones=[self.warmup]
                )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

