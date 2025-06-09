import torch
import torch.nn as nn
from ..timeEncoders import TimeHandler 
from ..utils.Token import Token
from dataclasses import dataclass


class LightCurveTransformer(nn.Module):
    def __init__(self,
        input_size= 1,
        embedding_size= 128,
        embedding_size_sub= 128,
        num_heads= 4,
        num_encoders= 3,
        Tmax= 1500.0,
        num_harmonics= 64,
        pe_type= 'tm',
        num_bands= 2,
        dropout = 0.00,
        checkpoint = None, 
        freeze_weights = False,
        ):
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.embedding_size_sub = embedding_size_sub
        self.num_heads = num_heads
        self.num_encoders= num_encoders
        self.Tmax = Tmax
        self.num_harmonics = num_harmonics
        self.pe_type = pe_type
        self.num_bands = num_bands
        super().__init__()
        self.time_encoder = TimeHandler(self.num_bands,
                                        self.input_size,
                                        self.embedding_size,
                                        self.Tmax,
                                        self.pe_type)
        self.dropout = nn.Dropout(dropout)
        self.transformer_lc = nn.TransformerEncoder(
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
            #norm=nn.LayerNorm(self.embedding_size),
        )
        self.token_lc = Token(self.embedding_size)
        self.register_buffer('ones', torch.ones(1,1,1,dtype = float))
        
    def embedding_light_curve(self, x, t, mask=None, **kwargs):
        x_mod, m_mod, t_mod = self.time_encoder(**{"x": x, "t": t, "mask": mask})
        token = self.token_lc(x.shape[0])
        x_mod = x_mod * m_mod   
        x_mod = torch.cat([token, x_mod], axis=1)
        m_mod = torch.cat(
            [   self.ones.repeat(x.size(0),1,1).bool(),
                m_mod,
            ],
            axis=1,
        )
        assert m_mod.dtype == torch.bool
        return x_mod, m_mod, t_mod

    def forward(self, data, time, mask, **kwargs):
        x_mod, m_mod, _ = self.embedding_light_curve(
            **{"x": data, "t": time, "mask": mask}
        )
        
        x_emb = self.transformer_lc(
            **{"src": x_mod, "src_key_padding_mask": ~(m_mod.squeeze(-1))}
        )
        return x_emb[:,0,:]
    