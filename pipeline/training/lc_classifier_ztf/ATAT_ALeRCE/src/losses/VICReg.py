import torch
import torch.nn.functional as F
import torch.nn as nn


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class VICReg(nn.Module):
    def __init__(self, inv_coeff,var_coeff,cov_coeff):
        super().__init__()
        self.inv = inv_coeff
        self.var = var_coeff
        self.cov = cov_coeff
    @staticmethod
    def calculate_cov_loss(embedding):
        cov_ = (embedding.T @ embedding) / (embedding.size(0) - 1)
        return off_diagonal(cov_).pow_(2).sum().div(embedding.size(-1))
    @staticmethod
    def calculate_std_loss(embedding):
        var_ = embedding.var(dim = 0)
        std_ = torch.sqrt(var_ + 0.0001)
        return torch.mean(F.relu(1.0 - std_))
    
    def forward(self, x, y):
        batch_size, embedding_size = x.shape
        loss_dict = {}
         
        repr_loss = F.mse_loss(x, y, reduce='mean')
        
        #loss_dict.update({'CORR/prenorm_x': torch.corrcoef(x).flatten()})
        #loss_dict.update({'CORR/prenorm_y':  torch.corrcoef(y).flatten()})

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0) 
        #loss_dict.update({'CORR/postnorm_x': torch.corrcoef(x).flatten()})
        #loss_dict.update({'CORR/postnorm_y':  torch.corrcoef(y).flatten()})



        loss_dict.update({'CORR/off_x': off_diagonal( ((x.T @ x) / (x.size(0) - 1)))})
        loss_dict.update({'CORR/off_y': off_diagonal( ((y.T @ y) / (y.size(0) - 1)))})
        std_loss = (self.calculate_std_loss(x) +  self.calculate_std_loss(y))/2
        cov_loss = self.calculate_cov_loss(x) + self.calculate_cov_loss(y)
        loss_dict.update({'loss':  (self.inv * (repr_loss)
                            + self.var * std_loss
                            + self.cov * cov_loss
                        ),})

        with torch.no_grad():
            loss_dict.update({'not_weighted_inv': repr_loss,
                'not_weighted_1menos_var': std_loss,
                'not_weighted_cov': cov_loss,
                'weighted_inv': self.inv * repr_loss,
                'weighted_1mvar': self.var * std_loss,
                'weighted_cov': self.cov * cov_loss, 
            })
        return loss_dict
