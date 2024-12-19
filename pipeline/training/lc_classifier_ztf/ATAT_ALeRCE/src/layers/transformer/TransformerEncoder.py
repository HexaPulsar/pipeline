
import torch.nn as nn 
from .FeedForward import FeedForward 


class TransformerEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() 
        self.layer_norm = nn.ModuleList([nn.LayerNorm(kwargs['embedding_size']) for _ in range(2)])
        self.attn_forward = DifferentialMultiheadAttention(embed_dim = kwargs['embedding_size'],
                                                           num_heads= kwargs['num_heads'],
                                                 dropout=0.1,
                                                 batch_first=True)
        self.feed_forward = FeedForward(**kwargs)

    def forward(self, x, src_key_padding_mask, **kwargs):
        norm_x = self.layer_norm[0](x)
        mha = self.attn_forward(**{"query": norm_x , 
                                 "key": norm_x , 
                                 "value": norm_x , 
                                 "key_padding_mask": src_key_padding_mask}) 
        x_ = x  + mha[0]
        norm_x = self.layer_norm[1](x_)  
        x = self.feed_forward(norm_x) + x_ 
        return x
 
