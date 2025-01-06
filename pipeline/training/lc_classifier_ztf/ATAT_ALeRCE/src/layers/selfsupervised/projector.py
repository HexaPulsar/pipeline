import torch
import torch.nn as nn 


class Projector(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,l2norm= False,**kwargs):
        super(Projector,self).__init__()
        self.kwargs  = kwargs
        self.l2norm = l2norm
        self.projection = nn.Sequential(nn.Linear(input_size,hidden_size,bias=True),
                                nn.LayerNorm(hidden_size), 
                                nn.GELU(),
                                nn.Linear(hidden_size,output_size,bias = False),
                                )
    def forward(self,embedding):
        if self.l2norm: 
            return self.projection(embedding / embedding.norm(dim=1, keepdim=True)) / embedding.norm(dim=1, keepdim=True)
        else:
             
            return self.projection(embedding)
       