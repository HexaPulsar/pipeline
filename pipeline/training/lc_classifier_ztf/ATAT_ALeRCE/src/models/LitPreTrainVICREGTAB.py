from copy import deepcopy
import inspect
import logging
import os
import random
import numpy as np
from src.augmentations import TabularTransformations as TAB

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
 
from typing import Dict, Optional, Literal
from src.layers.cATAT import cATAT, SingleBranch
import pytorch_lightning as pl  
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR 

from src.losses.VICReg import VICReg 
from torchvision.transforms import Compose,RandomApply

 
def save_transforms_to_file(transform_dict, filepath: str) -> None:
    """
    Save multiple transform compositions to a text file.
    
    Args:
        transform_dict: Dictionary with names and their corresponding transforms
                       e.g., {'train': train_transforms, 'val': val_transforms}
        filepath: Path where to save the text file
    """
    with open(filepath, 'w') as f:
        f.write("Transforms Configuration\n")
        f.write("=" * 50 + "\n\n")
        
        # Process each set of transforms
        for set_name, transform_compose in transform_dict.items():
            f.write(f"{set_name.upper()} TRANSFORMS\n")
            f.write("-" * 50 + "\n")
            
            transforms_list = transform_compose.transforms
            
            for i, transform in enumerate(transforms_list, 1):
                # Get the transform class name
                transform_name = transform.__class__.__name__
                
                # Get the initialization parameters
                params = inspect.signature(transform.__class__).parameters
                actual_params = {
                    key: getattr(transform, key) 
                    for key in params.keys() 
                    if hasattr(transform, key)
                }
                
                # Write transform details
                f.write(f"{i}. {transform_name}\n")
                if actual_params:
                    f.write("   Parameters:\n")
                    for param_name, param_value in actual_params.items():
                        f.write(f"   - {param_name}: {param_value}\n")
                f.write("\n")
            
            f.write("\n")


class LitPreTrainVICREGTAB(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__() 
        
        print(kwargs)
        self.gradients_ = None   
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

        self.loss = VICReg()  
        self.kwargs = kwargs
        self.model = SingleBranch(type = 'tab',**self.feature_) 
 
        self.warmup = 0
        
        self.transforms_aug_tab = Compose([
            RandomApply([TAB.GaussianNoise()],p = 0.5),
            RandomApply([TAB.GaussianNoise()],p = 0.5)
                                            ])
        self.transforms_data_tab =  Compose([
            RandomApply([TAB.GaussianNoise()],p = 0.5),
            RandomApply([TAB.GaussianNoise()],p = 0.5)
                                            ])
        transform_sets = {
            '1': self.transforms_data_tab,
            '2': self.transforms_aug_tab
        }

        # Save both transform sets to file
        #save_transforms_to_file(transform_sets, f"{self.kwargs['save_dir']}/transforms_config.txt")

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
        #print(f"Model device: {self.device}") 
        batch_data,aug_batch_data= batch
        #print(batch_data['data'].shape)
        batch_data = self.transforms_data_tab(batch_data)
        aug_batch_data = self.transforms_aug_tab(aug_batch_data) 
        
        batch_data = {k: batch_data[k].float() for k in  batch_data.keys()} 
        aug_batch_data = {k: aug_batch_data[k].float() for k in  aug_batch_data.keys()} 


        input_dict = {key: torch.cat([batch_data[key], aug_batch_data[key]], dim=0) for key in batch_data.keys()}
        
        x_emb = self.model(**input_dict)
        x_emb,aug_x_emb = torch.chunk(x_emb,2, dim = 0) 
        contrastive_loss = self.loss(x_emb,aug_x_emb)    

        loss_dict = contrastive_loss  
        loss_dict = {f'loss_train/{key}': value for key, value in loss_dict.items()}
        self.log_dict(loss_dict,on_epoch=False,on_step=True)
        
        return loss_dict['loss_train/loss']
     
    def validation_step(self, batch, batch_idx):
        batch_data,aug_batch_data= batch

        batch_data = self.transforms_data_tab(batch_data)
        aug_batch_data = self.transforms_aug_tab(aug_batch_data) 

        batch_data = {k: batch_data[k].float() for k in  batch_data.keys()} 
        aug_batch_data = {k: aug_batch_data[k].float() for k in  aug_batch_data.keys()} 

        input_dict = {key: torch.cat([batch_data[key], aug_batch_data[key]], dim=0) for key in batch_data.keys()}
        x_emb = self.model(**input_dict)
        x_emb,aug_x_emb = torch.chunk(x_emb,2, dim = 0) 
        
        loss_dict = self.loss(x_emb,aug_x_emb)  
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
        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[constant,constant],
                    milestones=[self.warmup]
                )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

