import torch
import torch.nn as nn 
import numpy as np

class TimeFilmModified(nn.Module):
    def __init__(self, n_harmonics=16, embedding_size=64, Tmax=1000.0, input_size=1):
        super().__init__()
        # print('n_harmonics:',n_harmonics)
        # self.dropout = nn.Dropout(0.1)
        self.alpha_weights = nn.Parameter(torch.randn(2*n_harmonics, embedding_size))
        self.beta_weights = nn.Parameter(torch.randn(2*n_harmonics, embedding_size))
        self.n_harmonics = n_harmonics
        self.register_buffer("ar", ( 2 * np.pi / Tmax) * torch.arange(n_harmonics).unsqueeze(0).unsqueeze(0))
        # self.embedding_size = embedding_size
        self.linear_proj = nn.Linear(in_features=input_size, out_features=embedding_size, bias=True)

    def concise_sin_cos(self,t):
        t = torch.einsum('bsh,bsh->bsh',self.ar,t.expand(-1,-1,self.n_harmonics))
        #t = self.ar * t.expand(-1, -1, self.n_harmonics)
        return  torch.sin(torch.cat([t,(torch.pi/2) - t],dim = -1))
    
    def get_sin(self, t):
        # t: Batch size x time x dim, dim = 1:
        return torch.sin(torch.einsum('bsh,bsh->bsh',self.ar,t.expand(-1,-1,self.n_harmonics)))
        

    def get_cos(self, t):
        # t: Batch size x time x dim, dim = 1:
        cos_t =  torch.cos(torch.einsum('bsh,bsh->bsh',self.ar,t.expand(-1,-1,self.n_harmonics)))
        cos_t = torch.cat([cos_t[:,0,:].unsqueeze(1), cos_t[:,1:,:].masked_fill_(t[:,1:,:] == 0,0)], dim = 1)
        return cos_t

    #@torch.compile
    def get_sin_cos(self, t):
        return torch.cat([self.get_sin(t), self.get_cos(t)], dim=-1)

    def forward(self, x, t):
        return torch.einsum('bse,bse->bse',self.linear_proj(x),torch.einsum('bsh,he->bse',self.concise_sin_cos(t),self.alpha_weights)).add_(torch.einsum('bsh,he->bse',self.concise_sin_cos(t),self.beta_weights))
    





class TimeFilmModifiedMOD(nn.Module):
    def __init__(self, n_harmonics=16, embedding_size=64, Tmax=1000.0, input_size=1):
        super().__init__()
        self.alpha_weights = nn.Parameter(torch.randn(2*n_harmonics, embedding_size))
        self.beta_weights = nn.Parameter(torch.randn(2*n_harmonics, embedding_size))
        self.n_harmonics = n_harmonics
        self.register_buffer("ar", ( 2 * torch.pi / Tmax) * torch.arange(n_harmonics).unsqueeze(0).unsqueeze(0))
        self.linear_proj = nn.Linear(in_features=input_size, out_features=embedding_size, bias=True)
        
    def concise_sin_cos(self,t):
        t =  self.ar*t.expand(-1,-1,self.n_harmonics)
        return  torch.dstack([torch.sin(t),torch.cos(t)]) 



    def forward(self, x, t):
        alpha = torch.matmul(self.concise_sin_cos(t),self.alpha_weights)
        beta = torch.matmul(self.concise_sin_cos(t),self.beta_weights)
        return (self.linear_proj(x)*alpha) + beta#.masked_fill_(t ==0,0)