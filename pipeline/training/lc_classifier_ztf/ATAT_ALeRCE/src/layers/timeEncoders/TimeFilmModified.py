import torch
import torch.nn as nn 
import numpy as np
import torch
import torch.nn as nn
import math


class TimeFilmModified(nn.Module):
    def __init__(self, n_harmonics=16, embedding_size=64, Tmax=1000.0, input_size=1):
        super(TimeFilmModified, self).__init__()
        self.alpha_sin = nn.Parameter(torch.randn(n_harmonics, embedding_size))
        self.alpha_cos = nn.Parameter(torch.randn(n_harmonics, embedding_size))
        self.beta_sin = nn.Parameter(torch.randn(n_harmonics, embedding_size))
        self.beta_cos = nn.Parameter(torch.randn(n_harmonics, embedding_size))
        self.Tmax = Tmax
        self.n_harmonics = n_harmonics
        self.register_buffer("ar", torch.arange(n_harmonics).unsqueeze(0).unsqueeze(0))
        self.embedding_size = embedding_size
        self.linear_proj = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=embedding_size, bias=True),
        )

    def get_sin(self, t):
        # t: Batch size x time x dim, dim = 1:
        return torch.sin(
            (2 * math.pi * self.ar * t.repeat(1, 1, self.n_harmonics)) / self.Tmax
        )

    def get_cos(self, t):
        # t: Batch size x time x dim, dim = 1:
        return torch.cos(
            (2 * math.pi * self.ar * t.repeat(1, 1, self.n_harmonics)) / self.Tmax
        )

    def get_sin_cos(self, t):
        return self.get_sin(t), self.get_cos(t)

    def forward(self, x, t):
        sin_emb, cos_emb = self.get_sin_cos(t)

        alpha = torch.matmul(sin_emb, self.alpha_sin) + torch.matmul(
            cos_emb, self.alpha_cos
        )
        beta = torch.matmul(sin_emb, self.beta_sin) + torch.matmul(
            cos_emb, self.beta_cos
        )

        return self.linear_proj(x) * alpha + beta

   

class TimeFilmModifiedMOD(nn.Module):
    def __init__(self, n_harmonics=4, embedding_size=64, Tmax=1000.0, input_size=1,dropout=0.01):
        super().__init__()
        self.alpha_weights = nn.Parameter(torch.randn(2*n_harmonics, embedding_size))
        self.beta_weights = nn.Parameter(torch.randn(2*n_harmonics, embedding_size))
        self.n_harmonics = n_harmonics 
        self.linear_proj = nn.Linear(in_features=input_size, out_features=embedding_size, bias=True)
        self.register_buffer("ar", ( 2 * torch.pi / Tmax) *torch.arange(n_harmonics).unsqueeze(0).unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def concise_sin_cos(self,t):         
        t =  self.ar*t.expand(-1,-1,self.n_harmonics)
        return  self.dropout(torch.dstack([self.get_sin(t),self.get_cos(t)]))

    def get_sin(self, t):
            # t: Batch size x time x dim, dim = 1:
            return torch.sin(self.ar * t.expand(-1, -1, self.n_harmonics))
     
    def get_cos(self, t):
        # t: Batch size x time x dim, dim = 1: 
        return torch.cos(self.ar * t.expand(-1, -1, self.n_harmonics)).masked_fill_(t ==0,0)


    def forward(self, x, t):
        alpha = torch.matmul(self.concise_sin_cos(t),self.alpha_weights)
        beta = torch.matmul(self.concise_sin_cos(t),self.beta_weights)
        return (self.linear_proj(self.dropout(x))*alpha).add_(beta)#.masked_fill_(t ==0,0)
    