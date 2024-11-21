import torch
import torch.nn.functional as F
import torch.nn as nn


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class VICReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.inv = 25
        self.var = 25
        self.cov = 1 

    def forward(self, x, y):
       
        repr_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (x.shape[0] - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            x.shape[-1]
        ) + off_diagonal(cov_y).pow_(2).sum().div(x.shape[-1])

        loss = (
            self.inv * repr_loss
            + self.var * std_loss
            + self.cov * cov_loss
        )
        return  {'loss': loss.mean(),
                'not_weighted_inv': repr_loss,
                'not_weighted_var': std_loss,
                'not_weighted_cov': cov_loss,
                'weighted_inv': self.inv * repr_loss,
                'weighted_var': self.var * std_loss,
                'weighted_cov': self.cov * cov_loss,
                #'mean_correlation_a': mean_correlation_a,
                #'mean_correlation_b': mean_correlation_b
                
                }   
