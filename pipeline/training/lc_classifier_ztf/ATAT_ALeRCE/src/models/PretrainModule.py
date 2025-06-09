from src.augmentations import LightCurveTransform as LC
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict, Optional, Literal
import pytorch_lightning as pl  
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR, LinearLR
import logging
from sklearn.metrics.pairwise import cosine_similarity
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
        logging.debug('using learning rate {}'.format(self.lr))
        self.init_model()
        
    def init_model(self):
        for name, p in self.loss.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean = 0.0, std = 0.1)
        for name, p in self.model.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                if 'token_lc.token' in name:
                        nn.init.uniform_(p)
        
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

    def training_step(self, batch, batch_idx):
        x = self.model(**batch[0])
        y = self.model(**batch[1])
        loss_dict = self.loss(x,y)
        with torch.no_grad():
            for key,value in loss_dict.items():
                if 'emb_corr' in key:
                    self.logger.experiment.add_histogram(key, value,self.global_step)
                elif 'percent' in key:
                    self.log(f'{key}', value ,on_epoch=False,on_step=True)
                else:
                    self.log(f'loss_train/{key}', value ,on_epoch=False,on_step=True)
           # if batch_idx % 10 == 0:
                #for harmonic in range(4):
                #    self.logger.experiment.add_histogram(f'HARMONICS/alpha_cos_harmonics_{harmonic}',self.model.time_encoder.time_encoders[harmonic].alpha_cos,self.global_step)
                #    self.logger.experiment.add_histogram(f'HARMONICS/alpha_sin_harmonics_{harmonic}',self.model.time_encoder.time_encoders[harmonic].alpha_sin,self.global_step) 
                
                #self.logger.experiment.add_histogram(f'token/x',self.model.token_lc.token,self.global_step)
                #self.logger.experiment.add_histogram(f'token/x',self.model.token_lc.token,self.global_step)
                #self.log("tmax_", self.model.time_encoder.time_encoders.0.Tmax.item() ,on_epoch=False,on_step=True)
                #self.log('tmax_1', self.model.time_encoder.time_encoders.1.Tmax ,on_epoch=False,on_step=True)

                #self.logger.experiment.add_histogram(f'token/x',self.model.token_tab.token,self.global_step)
                #self.logger.experiment.add_histogram(f'token/y',self.model.token_tab.token,self.global_step)

                #self.logger.experiment.add_histogram(f'out_emb/x',x,self.global_step)
                #self.logger.experiment.add_histogram(f'out_emb/y',y,self.global_step)
                #self.logger.experiment.add_histogram(f'cos_similarity',cosine_similarity(x,y),self.global_step)
        return loss_dict['loss']
     
    
    def validation_step(self, batch, batch_idx):
        embs = self.model(**batch[0])
        loss_dict = self.loss( embs ,self.model(**batch[1]))
        with torch.no_grad():
           for key,value in loss_dict.items():
                if 'emb_corr' not in key:
                    self.log(f'loss_validation/{key}', value ,on_epoch=True,on_step=False)
        return loss_dict['loss']

    def test_step(self, batch, batch_idx):
        return 0
    

    def configure_optimizers(self):
        warmup = 0
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        cosine = CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-8)
        constant = ConstantLR(optimizer,1)  
        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[constant,constant],
                    milestones=[warmup]
                )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]