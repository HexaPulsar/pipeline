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
        self.inv = 20
        self.var = 35
        self.cov = 1
        
    def forward(self, x, y):
        batch_size, embedding_size = x.shape
        loss_dict = {}
        #repr_loss = torch.sqrt(F.mse_loss(x, y, reduce='mean'))
        repr_loss = F.mse_loss(x, y, reduce='mean')
        
        loss_dict.update({'z_mean_corr_xy': (torch.corrcoef(x).mean() +torch.corrcoef(y).mean())/2})
        #loss_dict.update({'median_corr_xy': (torch.corrcoef(x).mean() +torch.corrcoef(y).mean()/2).median()})
        

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0) 

        var_x = x.var(dim = 0)
        var_y = y.var(dim = 0)

        std_x = torch.sqrt(var_x + 0.0001)
        std_y = torch.sqrt(var_y + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            embedding_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(embedding_size)


       
        loss_dict.update({'loss':  (self.inv * (repr_loss)
                            + self.var * std_loss
                            + self.cov * cov_loss
                        ),})
         

        with torch.no_grad():
            loss_dict.update({'not_weighted_inv': repr_loss,
                'not_weighted_1menos_var': std_loss,
                'not_weighted_cov': cov_loss,
                'weighted_inv': (self.inv * repr_loss).mean(),
                'weighted_1mvar': (self.var * std_loss).mean(),
                #'weighted_std*': self.var * (std_x.mean() + std_y.mean())/2,

                #'weighted_var*':  (var_x.mean() + var_y.mean())/2,
                'weighted_cov': (self.cov * cov_loss).mean(),
                #'mean_var_xy': (var_x.mean() + var_y.mean())/2, 
            })
        return loss_dict
