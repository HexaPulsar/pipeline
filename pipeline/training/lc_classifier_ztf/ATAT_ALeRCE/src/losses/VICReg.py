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
        self.inv = 5
        self.var = 5
        self.cov = 1

    def forward(self, x, y):
        batch_size, embedding_size = x.shape
        repr_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            embedding_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(embedding_size)

        loss_dict =  {'loss':  (self.inv * repr_loss
                            + self.var * std_loss
                            + self.cov * cov_loss
                        ).mean(),}

        with torch.no_grad():
            loss_dict.update({'not_weighted_inv': repr_loss,
                'not_weighted_var': std_loss,
                'not_weighted_cov': cov_loss,
                'weighted_inv': self.inv * repr_loss,
                'weighted_var': self.var * std_loss,
                'weighted_cov': self.cov * cov_loss})   
        return loss_dict
