
import torch.nn as nn 
from .FeedForward import FeedForward 
from .mha import MultiheadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() 
        self.layer_norm = nn.ModuleList([nn.LayerNorm(kwargs['embedding_size']) for _ in range(2)])
        self.attn_forward = MultiheadAttention(**kwargs)
        self.feed_forward = FeedForward(**kwargs)

    def forward(self, x, mask = None, **kwargs):
        norm_x = self.layer_norm[0](x)
        mha = self.attn_forward(**{"query": norm_x , 
                                 "key": norm_x , 
                                 "value": norm_x , 
                                 "mask": mask}) 
        x_ = x  + mha[0]
        norm_x = self.layer_norm[1](x_)  
        x = self.feed_forward(norm_x) + x_ 
        return x
 
