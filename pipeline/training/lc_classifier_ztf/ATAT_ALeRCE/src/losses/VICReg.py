import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICReg(nn.Module):
    def __init__(self, inv_coeff,var_coeff,cov_coeff, batch_size):
        super().__init__()
        self.inv = inv_coeff
        self.var = var_coeff
        self.cov = cov_coeff
        self.batch_size = batch_size
        print('COEFFS: {} {} {}'.format(self.inv, self.var, self.cov))

    def calculate_cov_loss(self,embedding):
        cov_ = (embedding.T @ embedding) / (self.batch_size - 1)
        return off_diagonal(cov_).pow_(2).sum().div(embedding.size(-1))
    @staticmethod
    def calculate_std_loss(embedding):
        var_ = embedding.var(dim = 0)
        std_ = torch.sqrt(var_ + 1e-4)
        return torch.mean(F.relu(1.0 - std_))/2
    
    def forward(self, x, y):
        loss_dict = {}

        repr_loss = F.mse_loss(x, y, reduction='mean')

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        corr_x = torch.corrcoef(x).mean(dim = 1).flatten()
        corr_y = torch.corrcoef(y).mean(dim = 1).flatten()

        loss_dict['CORR/emb_corr_post_norm_x'] = torch.corrcoef(x).mean(dim = 1).flatten()
        loss_dict['CORR/emb_corr_post_norm_y'] = torch.corrcoef(y).mean(dim = 1).flatten()

        std_loss = self.calculate_std_loss(x) +  self.calculate_std_loss(y)
        cov_loss = self.calculate_cov_loss(x) + self.calculate_cov_loss(y)
        loss_dict['loss'] = self.inv * (repr_loss) + self.var * std_loss + self.cov * cov_loss

        with torch.no_grad():
            loss_dict['not_weighted_inv'] = repr_loss
            loss_dict['not_weighted_1menos_var'] = std_loss
            loss_dict['not_weighted_cov'] = cov_loss
            loss_dict['weighted_inv'] = self.inv * repr_loss
            loss_dict['weighted_1mvar'] = self.var * std_loss
            loss_dict['weighted_cov'] = self.cov * cov_loss
            loss_dict['CORR/99_percentile_x'] = np.percentile(abs(corr_x).cpu().detach().numpy(), 99)
            loss_dict['CORR/99_percentile_y'] = np.percentile(abs(corr_y).cpu().detach().numpy(), 99)

        return loss_dict
