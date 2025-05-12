import torch
import torch.nn as nn
from ..utils.Token import Token


import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, length_size, embedding_size, **kwargs):
        super(Embedding, self).__init__()

        self.tab_W_feat = nn.Parameter(torch.randn(1, length_size, embedding_size))
        self.tab_b_feat = nn.Parameter(torch.randn(1, length_size, embedding_size))

    def forward(self, f): 
        return self.tab_W_feat * f + self.tab_b_feat


class TabularTransformer(nn.Module):
    def __init__(self,
        embedding_size= 128,
        embedding_size_sub= 512,
        num_heads= 4,
        num_encoders= 3,
        length_size=7,
        num_bands= 2,
        dropout = 0.00,):

        self.embedding_size = embedding_size
        self.embedding_size_sub = embedding_size_sub
        self.num_heads = num_heads
        self.num_encoders= num_encoders
        self.num_bands = num_bands
        self.length_size =length_size
        
        super().__init__()
        self.embedding_tab = Embedding(
            self.length_size,self.embedding_size
        )  # nn.Linear(kwargs['TAB_ARGS']['length_size'],kwargs['TAB_ARGS']['embedding_size']) #
        self.transformer_tab = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_size,
                nhead=self.num_heads,
                dim_feedforward=self.embedding_size_sub,
                activation="gelu",
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.num_encoders,
        )
        self.token_tab = Token(self.embedding_size)
        self.register_buffer('ones', torch.ones(1,1,1,dtype = float))
        self.dropout = nn.Dropout(dropout)
    def embedding_feats(self, f, tab_mask=None):
        f_mod = self.embedding_tab(**{"f": f})
        return torch.cat([self.token_tab(f.shape[0]), f_mod], axis=1)

    def forward(self, tabular_feat, tab_mask=None, **kwargs):
        
        f_mod=  self.embedding_feats(
            **{"f": tabular_feat, "tab_mask": tab_mask}
        )
        tab_mask = torch.ones(f_mod.shape[:2], device = f_mod.device)
        #dropout token dims
        #tab_mask = self.dropout(tab_mask)
        #tab_mask[0,:]  = 1
        #tab_mask = ~((tab_mask).bool())
        tab_mask = None
        f_mod = f_mod /f_mod.norm(dim = 1,keepdim=True)
        f_emb = self.transformer_tab(**{"src": f_mod, "src_key_padding_mask": tab_mask})[:,0,:]
        #f_emb = self.transformer_tab(**{"src": f_mod})[:,0,:]
        return self.dropout(f_emb)
