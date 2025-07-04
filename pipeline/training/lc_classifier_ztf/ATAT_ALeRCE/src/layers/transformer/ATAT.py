import torch
import torch.nn as nn
from ..timeEncoders import TimeHandler 

import torch
import torch.nn as nn


class Token(nn.Module):
    def __init__(self, embedding_size, **kwargs):
        super(Token, self).__init__()

        # self.token = nn.parameter.Parameter(
        #    torch.rand(embedding_size), requires_grad=True
        # )

        self.token = nn.Parameter(torch.rand(embedding_size), requires_grad=True)
        
    def forward(self, n_batch):
        return nn.functional.softmax(self.token).repeat(n_batch, 1, 1)

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
        self.register_buffer('ones', torch.ones(1,1,1,dtype = torch.bool))

    def load_weights(self):
        pass

    def embedding_light_curve(self, x, 
                              t, 
                              mask,
                              metadata = None,
                               coordinates = None,
                                allwise = None, 
                                 timespan = None, **kwargs):
        
        x_mod, m_mod, t_mod = self.time_encoder(x, 
                              t, 
                                mask=mask,
                                metadata = metadata,
                                coordinates = coordinates,
                                allwise = allwise, 
                                timespan = timespan)
        x_norm = torch.sqrt(torch.linalg.norm(x_mod, dim = (1), keepdim = True))
        x_mod = x_mod / (x_norm + 1e-8)

        x_mod = torch.cat([self.token_lc(x.shape[0]), x_mod], axis=1)
        m_mod = torch.cat(
            [   self.ones.repeat(x.size(0),1,1),
                m_mod,
            ],
            axis=1,
        )
        assert m_mod.dtype == torch.bool, 'm_mod type is {}'.format(m_mod.dtype)
        return x_mod, m_mod, t_mod

    def forward(self, data, 
                        time, 
                        mask, 
                        metadata = None,
                        timespan = None,
                        coordinates = None,
                        allwise = None,
                        **kwargs):
        
        x_mod, m_mod, _ = self.embedding_light_curve(x = data,
                                                    t = time, 
                                                    mask=mask,
                                                    metadata = metadata,
                                                    coordinates = coordinates,
                                                    allwise = allwise, 
                                                    timespan = timespan) 
        x_emb = self.transformer_lc(
                                    src = x_mod, 
                                    src_key_padding_mask = ~(m_mod.squeeze(-1))
                                    ) 
        return self.dropout(x_emb[:,0,:])
    
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
        length_size=6,
        num_bands= 2,
        dropout = 0.00,
        checkpoint = None, 
        freeze_weights = False,):

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
        norm=nn.LayerNorm(self.embedding_size),
        )
        self.token_tab = Token(self.embedding_size)
        self.register_buffer('ones', torch.ones(1,1,1,dtype = float))
        self.dropout = nn.Dropout(dropout)

    def embedding_feats(self, f):
        f_mod = self.embedding_tab(**{"f": f})
        f_norm = torch.sqrt(torch.linalg.norm(f_mod, dim = (1), keepdim = True))
        f_mod = f_mod / (f_norm + 1e-8)
        f_mod = torch.cat([self.token_tab(f.shape[0]), f_mod], axis=1)
        return f_mod

    def forward(self, tabular_feat, tab_mask=None, **kwargs):
        f_mod=  self.embedding_feats(
            **{"f": tabular_feat}
        )
        f_emb = self.transformer_tab(**{"src": f_mod, "src_key_padding_mask": tab_mask})
        return self.dropout(f_emb[:,0,:])
 

class Combinator(nn.Module):
    def __init__(self,  lc_model, tab_model,how ='concat', as_dict = True):
        super().__init__()
        self.transformer_lc = lc_model if lc_model is not None else None
        self.transformer_tab = tab_model if tab_model is not None else None
        self.how = how    
        self.as_dict =  as_dict

    def forward(self,data,time,mask, tabular_feat = None,metadata_feat = None,extracted_feat = None, **kwargs):
        
        lc_emb = self.transformer_lc(data,time,mask)
        ft_emb  = self.transformer_tab(tabular_feat)
       # return torch.concat([lc_emb,ft_emb],axis  = -1)
        if self.how == 'concat':
            if self.as_dict:
                return {'LC':lc_emb, "TAB" :ft_emb, "MIX": torch.concat([lc_emb,ft_emb],axis = -1)}
            else: 
                torch.concat([lc_emb,ft_emb],axis = -1)
        elif self.how == 'sum':
            return lc_emb+ft_emb
   