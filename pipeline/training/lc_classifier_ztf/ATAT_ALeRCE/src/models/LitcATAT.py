import glob
from collections import OrderedDict
from copy import deepcopy
import os 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict, Optional, Literal

import pytorch_lightning as pl
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR
from src.losses.CYCLIP import CyCLIP
from src.layers.cATAT import cATAT
from torchvision.transforms import Compose, RandomApply
from src.augmentations import TabularTransformations as TAB

from src.augmentations import LightCurveTransform as LC



class LitcATAT(pl.LightningModule):
    def __init__(self, lc_ckpt,ft_ckpt,**kwargs):
        super().__init__()
        self.gradients_ = None   
        self.model = cATAT(**kwargs)
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

        
        self.loss = CyCLIP(
            config_dict= {'cross_modal':True,
               'in_modal':True,
               'cylambda_1': 0.25,
               'cylambda_2':0.25},
            config_logger = {
                 
                'cross_contrastive_lc':False,
                'cross_contrastive_ft':False,
                "cross_contrastive_loss":  False,
                "cross_cyclic_loss_l2": True,
                
                'inmodal_contrastive_lc':False,
                'inmodal_contrastive_ft':False,
                "inmodal_contrastive_loss":  False,
                "inmodal_cyclic_loss_l1": True,

                "total_cyclic_loss": False,  
                "total_contrastive_loss": False,
                "total_loss": True,
                })

        self.use_lightcurves = self.general_["use_lightcurves"]
        self.use_lightcurves_err = self.general_["use_lightcurves_err"]
        self.use_metadata = self.general_["use_metadata"]
        self.use_features = self.general_["use_features"]
        self.warmup = 0

        self.use_cosine_decay = kwargs["general"]["use_cosine_decay"]
        self.gradient_clip_val = (
            1.0 if kwargs["general"]["use_gradient_clipping"] else 0
        )
            
        # TODO: correct transformation pipeline and augs
         
        self.transforms = Compose([RandomApply([TAB.GaussianNoise()],p = 0.5),
                                    RandomApply([TAB.GaussianNoise()],p = 0.5),
                                    LC.OnlyMaskPadding(),
                                    RandomApply([LC.Scale(0.5,1.5),],p =0.5),
                                    RandomApply([LC.GaussianNoise(),],p =0.5),
                                    RandomApply([LC.TimeWarp(0.9,1.2),],p =0.5),
                                    RandomApply([LC.SequenceShift((-5,0)),],p =0.5),
                                    ])

############LC LOAD CKPT##
        lc_out_path = glob.glob(lc_ckpt+ "*.ckpt")[0]
        checkpoint_ = torch.load(lc_out_path)
        weights = OrderedDict()
        for key in checkpoint_["state_dict"].keys():
            if 'projection' in key:
                continue
            else:    
                weights[key.replace("model.transformer.", "")] = checkpoint_["state_dict"][key] 
        self.model.LC.load_state_dict(weights, strict=True)


############FT LOAD CKPT##
        ft_out_path = glob.glob(ft_ckpt+ "*.ckpt")[0]
        checkpoint_ = torch.load(ft_out_path)
        weights = OrderedDict()
        for key in checkpoint_["state_dict"].keys():
            if 'projection' in key:
                continue
            else:    
                weights[key.replace("model.transformer.", "")] = checkpoint_["state_dict"][key] 
        self.model.TAB.load_state_dict(weights, strict=True)

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
    
    def training_step(self, batch_data, batch_idx):
        batch_data,aug_batch_data = batch_data
        aug_batch_data = self.transforms(aug_batch_data) 
        
        batch_data = {k: batch_data[k].float() for k in  batch_data.keys()} 
        aug_batch_data = {k: aug_batch_data[k].float() for k in  aug_batch_data.keys()} 
        input_dict = {key: torch.cat([batch_data[key], aug_batch_data[key]], dim=0) for key in batch_data.keys()}
        #print(input_dict.keys())
        x_emb,f_emb = self.model(**input_dict)

        cyclip_dict = self.loss(x_emb,f_emb,self.model)
        loss = cyclip_dict['total_loss']
        self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data,0,4.605) 
        

        # Create the cyclip_dict as before
        cyclip_dict = {f'alignment/loss_train/{key}': value for key, value in cyclip_dict.items()}
        cyclip_dict.update({"alignment/temp_value": self.model.logit_scale.item()})
       
        # Log all metrics in cyclip_dict
        self.log_dict(cyclip_dict, on_step=True, on_epoch=True)
 
        return loss

    def validation_step(self, batch_data, batch_idx):
        batch_data,aug_batch_data = batch_data
        aug_batch_data = self.transforms(aug_batch_data) 
        
        batch_data = {k: batch_data[k].float() for k in  batch_data.keys()} 
        aug_batch_data = {k: aug_batch_data[k].float() for k in  aug_batch_data.keys()} 
        input_dict = {key: torch.cat([batch_data[key], aug_batch_data[key]], dim=0) for key in batch_data.keys()}

        x_emb,f_emb = self.model(**input_dict)

        cyclip_dict = self.loss(x_emb,f_emb,self.model)
        loss = cyclip_dict['total_loss']
        self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data,0,4.605) 
        

        # Create the cyclip_dict as before
        cyclip_dict = {f'alignment/loss_validation/{key}': value for key, value in cyclip_dict.items()}

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

