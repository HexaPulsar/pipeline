from copy import deepcopy
import os 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchmetrics
from collections import OrderedDict
from typing import Dict, Optional, Literal

import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR
from src.training.schedulers import cosine_decay_ireyes

from src.layers import cATAT  
class LitcATAT(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.gradients_ = None   
        print(kwargs)
        self.catat = cATAT(**kwargs)
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

        self.cyclip_dict = {'inmodal':True,
                              'cylambda1':0.25,
                              'cylambda2':0.25,}    # TODO: make into arguments    
        self.loss = CyCLIP(criterion='',**self.cyclip_dict)

        self.use_lightcurves = self.general_["use_lightcurves"]
        self.use_lightcurves_err = self.general_["use_lightcurves_err"]
        self.use_metadata = self.general_["use_metadata"]
        self.use_features = self.general_["use_features"]
 

        self.use_cosine_decay = kwargs["general"]["use_cosine_decay"]
        self.gradient_clip_val = (
            1.0 if kwargs["general"]["use_gradient_clipping"] else 0
        )
            
        # TODO: correct transformation pipeline and augs
         
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
        self.gradients = self.gradfilter_ema(m=self.atat,
                                        grads = self.gradients_)
    
    def get_gradients(self):
        grads = []
        for param in self.catat.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        return grads
    
    def align_loss(self,x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self,x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    
    def training_step(self, batch_data, batch_idx):
        input_dict = self.get_input_data(batch_data)
         
        if self.cyclip_dict['inmodal']:
           # print('entered inmodal')

            aug_input_dict = deepcopy(input_dict) 
            input_dict = {key: torch.cat([input_dict[key], aug_input_dict[key]], dim=0) for key in input_dict}

        x_emb,f_emb = self.catat(**input_dict)
        #print(x_emb.shape)
        cyclip_dict = self.loss(x_emb,f_emb,self.catat)
        loss = cyclip_dict['loss']
        
         
        self.catat.logit_scale.data = torch.clamp(self.catat.logit_scale.data,0,4.605) 
         
        

        # Create the cyclip_dict as before
        cyclip_dict = {f'pretrain/loss_train/{key}': value for key, value in cyclip_dict.items()}
        cyclip_dict.update({"pretrain/other/temp_value": self.catat.logit_scale.item()})
        cyclip_dict.update({"pretrain/other/alignment": self.align_loss(x_emb, f_emb)})

        # Log all metrics in cyclip_dict
        self.log_dict(cyclip_dict, on_step=True, on_epoch=True)
 
        self.log('loss_train_epoch', cyclip_dict['pretrain/loss_train/loss'], 
                    on_step=False, on_epoch=True)

        

        return loss

    def validation_step(self, batch_data, batch_idx):
        input_dict = self.get_input_data(batch_data)
 

        if self.cyclip_dict['inmodal']:
            aug_input_dict = deepcopy(input_dict) 

            input_dict = {key: torch.cat([input_dict[key], aug_input_dict[key]], dim=0) for key in input_dict}

        x_emb,f_emb = self.catat(**input_dict)

        cyclip_dict = self.loss(x_emb,f_emb,self.catat)
        loss = cyclip_dict['loss']

        self.catat.logit_scale.data = torch.clamp(self.catat.logit_scale.data,0,4.605) 
        cyclip_dict = {f'pretrain/loss_validation/{key}': value for key, value in cyclip_dict.items()}

        # Log all metrics in cyclip_dict
        self.log_dict(cyclip_dict, on_step=False, on_epoch=True)

        # Optionally, log the main validation loss separately if you want it in the progress bar
        if 'pretrain/loss_validation/loss' in cyclip_dict:
            self.log('val_loss', cyclip_dict['pretrain/loss_validation/loss'], 
                    on_step=False, on_epoch=True, prog_bar=True)
            
        return loss

    def test_step(self, batch_data, batch_idx):
        pass

    def configure_optimizers(self):
        #params = filter(lambda p: p.requires_grad, self.parameters())


        weight_decay_parameters = []
        no_weight_decay_parameters = []
        no_weight_decay_param_names = []  
        for name, parameter in self.catat.named_parameters():
            if(all(key not in name for key in ["bn", "layer_norm","batch_norm","norm", "bias", "logit_scale"]) and parameter.requires_grad):
                weight_decay_parameters.append(parameter)
                
            if(any(key in name for key in ["bn","layer_norm", "batch_norm", "norm", "bias", "logit_scale"]) and parameter.requires_grad):
                no_weight_decay_parameters.append(parameter)
                no_weight_decay_param_names.append(name) 
        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, 
                                        "weight_decay": 0}, 

                                        {"params": weight_decay_parameters, 
                                        "weight_decay":0.1}], 
                                        lr = self.general_["lr"],
                                         )      
        if self.use_cosine_decay:
            scheduler = LambdaLR(
                optimizer,
                lambda epoch: cosine_decay_ireyes(
                    epoch, warm_up_epochs=10, decay_steps=150, alpha=0.05
                ),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        else:
            return optimizer

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

