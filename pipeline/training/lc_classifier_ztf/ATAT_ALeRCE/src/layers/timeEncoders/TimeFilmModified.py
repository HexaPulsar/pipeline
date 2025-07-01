
import torch
import torch.nn as nn

import torch
import torch.nn as nn


class AlphaCoeffs(nn.Module):
    def __init__(self, n_harmonics, embedding_size, dropout):
        super().__init__()
        self.register_parameter('alpha_sincos',nn.Parameter(torch.randn(2*n_harmonics, embedding_size)))
        self.alpha_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,embedding):
        alpha = self.dropout(torch.matmul(embedding, self.alpha_sincos))
        alpha = self.alpha_norm(alpha)
        alpha = nn.functional.gelu(alpha)
        return alpha
    
class BetaCoeffs(nn.Module):
    def __init__(self, n_harmonics, embedding_size, dropout):
        super().__init__()
        self.register_parameter('beta_sincos',nn.Parameter(torch.randn(2*n_harmonics, embedding_size)))
        self.beta_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,embedding):
        beta = self.dropout(torch.matmul(embedding, self.beta_sincos))
        beta = self.beta_norm(beta)
        beta = nn.functional.gelu(beta)
        return beta 
    
class TimeFilmModified(nn.Module):
    def __init__(self, n_harmonics=1, embedding_size=64, Tmax=1000.0, input_size=1):
        super(TimeFilmModified, self).__init__() 
        self.linear_proj_only = False
        if self.linear_proj_only:
            print("USING LINEAR_PROJ ONLY!!!") 
            self.linear_proj = nn.Sequential(nn.Linear(in_features = input_size,out_features = embedding_size,bias = False))
        else:
            p = 1e-5
            self.alpha_coeffs = AlphaCoeffs(n_harmonics,embedding_size, p)
            self.beta_coeffs = BetaCoeffs(n_harmonics,embedding_size,p)
            self.n_harmonics = n_harmonics
            self.register_buffer("ar", torch.arange(1,n_harmonics+1).unsqueeze(0).unsqueeze(0))
            self.register_buffer("Tmax", torch.tensor(Tmax))
            self.register_buffer('const', torch.tensor( 2 * torch.pi ))
            self.linear_proj = nn.Sequential(
                                             nn.Linear(in_features = input_size,out_features = embedding_size,bias = True),
                                             nn.Dropout(p)
                                             )
            self.linear_proj_dxdt = nn.Sequential(nn.Linear(in_features = input_size,out_features = embedding_size,bias = True),
                                             nn.Dropout(p),
                                           
                                             )
            self.linear_proj_dxdtdt = nn.Sequential(nn.Linear(in_features = input_size,out_features = embedding_size,bias = True),
                                             nn.Dropout(p),
                                             )
            self.out = nn.Linear(embedding_size*3, embedding_size)
            self.embedding_size = embedding_size
            self.dropout = nn.Dropout(p=p)
           # self.norm = nn.LayerNorm(n_harmonics)
    @staticmethod
    def get_sin_cos(t):
        sin = torch.sin(t) 
        cos = torch.cos(t)
        return torch.exp(sin).masked_fill_(sin == 0, 0) , torch.exp(cos).masked_fill_(cos == 1, 0) 
    @staticmethod
    def roll_tensor(v):
        v = torch.roll(v, shifts = (1,), dims = 0)
        v[0,:] = 0  
        return v
    
    def dxdt(self,x,t):
        dt = t - self.roll_tensor(t) 
        dx = x - self.roll_tensor(x)
        return dx,dt,dx/dt.masked_fill_(dt == 0, 1) # fil to avoid zero division
        
    
    def forward(self, x, t):  
        dx,dt,dxdt = self.dxdt(x,t) 
        _,_,dxdtdt  = self.dxdt(dx,dt)
        t =self.const * self.ar*t.repeat(1,1,self.n_harmonics) / self.Tmax
        sin_emb, cos_emb = self.get_sin_cos(t)
        emb = torch.concat([sin_emb, cos_emb], axis = -1)
        alpha = self.alpha_coeffs(emb)
        beta = self.beta_coeffs(emb)
        x = torch.concat([self.linear_proj(x),self.linear_proj_dxdt(dxdt),self.linear_proj_dxdtdt(dxdtdt)], axis = -1)
        x = self.out(x)
        out = (x)* alpha +  beta 
        return self.dropout(out)
