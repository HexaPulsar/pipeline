     
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torchmetrics import Accuracy

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter('logit_scale',nn.Parameter( torch.log(torch.tensor(1/0.07)))  )
    
    def forward(self, m1_emb, m2_emb):
        # Initialize loss tensors 
        #acc = {}
        loss_dict = {}
        # Prepare target tensor
        target = torch.arange(m1_emb.shape[0]).long().to(device=m1_emb.device, non_blocking=True)
        # Compute logits
        logits_m1_per_m2 = self.logit_scale.exp() * m1_emb @ m2_emb.t()
        #acc['acc/cross'] = self.compute_top1_accuracy(logits_ft_per_lc)
        logits_m2_per_m1 = logits_m1_per_m2.t() 
        loss =  (self.criterion(logits_m2_per_m1, target)  + self.criterion(logits_m1_per_m2, target))/2 
        
        loss_dict.update({'loss':loss})
        return loss_dict

