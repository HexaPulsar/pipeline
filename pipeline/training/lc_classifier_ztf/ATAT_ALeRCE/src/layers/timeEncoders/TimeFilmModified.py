
import torch
import torch.nn as nn

class TimeFilmModified(nn.Module):
    def __init__(self, n_harmonics=4, embedding_size=64, Tmax=1000.0, input_size=1):
        super(TimeFilmModified, self).__init__() 
        self.linear_proj_only = False
        if self.linear_proj_only:
            print("USING LINEAR_PROJ ONLY!!!") 
            self.linear_proj = nn.Sequential(nn.Linear(in_features = input_size,out_features = embedding_size,bias = False))
        else:
            self.register_parameter('alpha_sin',nn.Parameter(torch.randn(n_harmonics, embedding_size)))
            self.register_parameter('alpha_cos',nn.Parameter(torch.randn(n_harmonics, embedding_size)))
            self.register_parameter('beta_sin',nn.Parameter(torch.randn(n_harmonics, embedding_size)))
            self.register_parameter('beta_cos',nn.Parameter(torch.randn(n_harmonics, embedding_size)))
            #self.linear_r = nn.Linear(in_features=n_harmonics, out_features=embedding_size, bias=False)
            self.n_harmonics = n_harmonics
            self.register_buffer("ar", torch.arange(0,n_harmonics).unsqueeze(0).unsqueeze(0))
            self.register_parameter("Tmax",nn.Parameter(torch.tensor(Tmax, dtype = torch.float)))
            self.register_buffer('const', torch.tensor( 2 * torch.pi ))
            self.linear_proj = nn.Sequential(nn.Linear(in_features = input_size,out_features = embedding_size,bias = True))
            self.embedding_size = embedding_size
            self.dropout = nn.Dropout(p=0.01)

        #self.norm = nn.LayerNorm(embedding_size)
          #  self.lif = snn.Leaky(beta=torch.rand(embedding_size),
          #                     threshold = 1.0,
          #                     learn_beta=True,
           #                    learn_threshold = True, 
           #                    learn_graded_spikes_factor = True,
           #                    spike_grad=None)

    def get_sin_cos(self, t,mask):
        sin = torch.sin(t) 
        cos = torch.cos(t)
        #cos[0:] = torch.masked_fill(cos[0:],cos[0:]== 1, 0)
        return sin, cos

    def gelu_drop(self,coeffs):
        return nn.functional.gelu(coeffs)

    def forward(self, x, t,mask):
            
            sin_emb, cos_emb = self.get_sin_cos((self.const * self.ar*t.repeat(1,1,self.n_harmonics)/ self.Tmax),mask)

             
            alpha = (torch.matmul(sin_emb, self.alpha_sin) + torch.matmul(cos_emb, self.alpha_cos))  
            beta = (torch.matmul(sin_emb, self.beta_sin) + torch.matmul(cos_emb, self.beta_cos)) 
             
            x =  self.linear_proj(x) 
            x = self.dropout(x)
            
            out = x* alpha +  beta
            out = out / out.norm(dim = 1, keepdim = True)
            out = out / out.norm(dim = 2, keepdim = True)
            out = self.gelu_drop(out)
            
            return out