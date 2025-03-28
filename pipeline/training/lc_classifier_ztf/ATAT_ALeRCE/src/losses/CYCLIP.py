     
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torchmetrics import Accuracy

class CyCLIP(torch.nn.Module):

    def __init__(self,config_dict,config_logger,**kwargs):
        super().__init__()
        self.config_dict = config_dict
        self.config_logger = config_logger
        self.criterion = nn.CrossEntropyLoss()  

    def compute_top1_accuracy(self, contrastive_matrix):
            """
            Function to compute Top-1 accuracy given a contrastive matrix.
            """
            targets = torch.arange(contrastive_matrix.size(0), device=contrastive_matrix.device)
            _, top_1_predictions = torch.max(contrastive_matrix, dim=1)
            accuracy = (top_1_predictions == targets).float().mean()
            return accuracy
    
    def contrastive_matrix_alignment(matrix: torch.Tensor) -> torch.Tensor:
        """
        Measures the alignment of a contrastive matrix using cosine similarity.
        
        Args:
            matrix (torch.Tensor): A 2D tensor of shape (N, D), where N is the number of vectors
                                and D is the dimension of each vector.
        
        Returns:
            torch.Tensor: The mean pairwise cosine similarity, representing alignment.
        """
        # Normalize the matrix rows to unit vectors
        matrix = F.normalize(matrix, p=2, dim=1)
        
        # Compute cosine similarity matrix
        cosine_sim_matrix = torch.mm(matrix, matrix.T)  # (N, N) matrix
        
        # Extract upper triangular values excluding the diagonal
        N = matrix.shape[0]
        mask = torch.triu(torch.ones(N, N, device=matrix.device), diagonal=1).bool()
        mean_alignment = cosine_sim_matrix[mask].mean()

        return mean_alignment

    def forward(self, m1_emb, m2_emb, model):
        # Initialize loss tensors 
        
        #acc = {}
        loss_dict = {}
        # Handle input embeddings based on inmodal flag
        if self.config_dict['in_modal']:
            lc_embeds, augmented_lc_embeds = torch.chunk(m1_emb, 2)
            ft_embeds, augmented_ft_embeds = torch.chunk(m2_emb, 2)
        else:
            lc_embeds = m1_emb
            ft_embeds = m2_emb
            
        # Prepare target tensor
        target = torch.arange(lc_embeds.shape[0]).long().to(device=lc_embeds.device, non_blocking=True)
        
        # Compute logits
        logits_ft_per_lc = model.logit_scale.exp() * lc_embeds @ ft_embeds.t()
        #acc['acc/cross'] = self.compute_top1_accuracy(logits_ft_per_lc)
        logits_lc_per_ft = logits_ft_per_lc.t()
        loss_dict.update({'contrastive_alignment':self.compute_top1_accuracy(lc_embeds @ ft_embeds.t()) })
        #loss_dict.update({'corr_lc':torch.corrcoef(lc_embeds) })
        #loss_dict.update({'corr_tab':torch.corrcoef(ft_embeds) })
        self.batch_size = len(logits_lc_per_ft)
  

        # Compute losses
        if self.config_dict['in_modal']:
            # Inmodal augmented logits
            logits_lc_per_augmented_lc = model.logit_scale.exp() * lc_embeds @ augmented_lc_embeds.t()
            logits_ft_per_augmented_ft = model.logit_scale.exp() * ft_embeds @ augmented_ft_embeds.t()
            loss_dict.update({'alig_lclc':self.compute_top1_accuracy(logits_lc_per_augmented_lc) })
            loss_dict.update({'alig_tabtab':self.compute_top1_accuracy(logits_ft_per_augmented_ft) })
            cross_contrastive_lc = self.criterion(logits_ft_per_lc, target) / 2
            cross_contrastive_ft = self.criterion(logits_lc_per_ft, target) / 2
            total_cross_contrastive_loss = cross_contrastive_lc + cross_contrastive_ft
            
            inmodal_contrastive_lc = self.criterion(logits_lc_per_augmented_lc, target) / 2
            inmodal_contrastive_ft = self.criterion(logits_ft_per_augmented_ft, target) / 2
            total_inmodal_contrastive_loss = inmodal_contrastive_lc + inmodal_contrastive_ft
            
            total_contrastive_loss = (total_cross_contrastive_loss + total_inmodal_contrastive_loss) / 2
        else:
            # Simple cross-modal contrastive loss when no augmentation
            total_contrastive_loss  = (self.criterion(logits_ft_per_lc, target) + self.criterion(logits_lc_per_ft, target)) / 2
        
            

        # Cyclic losses
        if self.config_dict['cylambda_1'] > 0:
            logits_lc_per_lc = model.logit_scale.exp() * lc_embeds @ lc_embeds.t()
            logits_ft_per_ft = model.logit_scale.exp() * ft_embeds @ ft_embeds.t()
            inmodal_cyclic_loss = (logits_lc_per_lc - logits_ft_per_ft).square().mean() / (model.logit_scale.exp() * model.logit_scale.exp()) * self.batch_size

        if self.config_dict['cylambda_2'] > 0:
            cross_cyclic_loss = (logits_ft_per_lc - logits_lc_per_ft).square().mean() / (model.logit_scale.exp() * model.logit_scale.exp()) * self.batch_size
            #inconsistency: cant turn of 1 lambda while keeping 2 lambda on
            total_cyclic_loss = self.config_dict['cylambda_1'] * inmodal_cyclic_loss + self.config_dict['cylambda_2'] * cross_cyclic_loss
        
            total_loss = total_contrastive_loss + total_cyclic_loss

       
        loss_dict.update({'cross_contrastive_lc': cross_contrastive_lc}) if self.config_logger['cross_contrastive_lc'] else None
        loss_dict.update({'cross_contrastive_ft': cross_contrastive_ft}) if self.config_logger['cross_contrastive_ft'] else None
        loss_dict.update({'cross_contrastive_loss': total_cross_contrastive_loss}) if self.config_logger['cross_contrastive_loss'] else None  
        loss_dict.update({'cross_cyclic_loss_l2': self.config_dict['cylambda_2'] * cross_cyclic_loss}) if self.config_logger['cross_cyclic_loss_l2'] else None  
        
        loss_dict.update({'inmodal_contrastive_lc': inmodal_contrastive_lc}) if self.config_logger['inmodal_contrastive_lc'] else None
        loss_dict.update({'inmodal_contrastive_ft': inmodal_contrastive_ft}) if self.config_logger['inmodal_contrastive_ft'] else None
        loss_dict.update({'inmodal_contrastive_loss': total_inmodal_contrastive_loss}) if self.config_logger['inmodal_contrastive_loss'] else None
        loss_dict.update({'inmodal_cyclic_loss_l1': self.config_dict['cylambda_1'] * inmodal_cyclic_loss}) if self.config_logger['inmodal_cyclic_loss_l1'] else None
        
        loss_dict.update({'total_cyclic_loss': total_cyclic_loss}) if self.config_logger['total_cyclic_loss'] else None
        loss_dict.update({'total_contrastive_loss':total_contrastive_loss }) if self.config_logger['total_contrastive_loss'] else None
        loss_dict.update({'total_loss': total_loss}) if self.config_logger['total_loss'] else None
        
        return loss_dict

