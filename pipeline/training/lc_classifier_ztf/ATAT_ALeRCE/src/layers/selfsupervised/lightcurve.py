
import torch
import torch.nn as nn
from ..timeEncoders import TimeHandler, TimeHandlerMOD
from ..classifiers import TokenClassifier
from ..tokenEmbeddings import Token 
from .projector import Projector



class LightCurveTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveTransformer, self).__init__()
        self.time_encoder = TimeHandlerMOD(**kwargs)
        
        encoder = nn.TransformerEncoderLayer(d_model = kwargs['embedding_size'],
                                                 nhead=kwargs['num_heads'],
                                                 dim_feedforward=kwargs['embedding_size_sub'],
                                                 activation = 'gelu',
                                                 dropout=0.01,
                                                 batch_first=True,
                                                 norm_first=True)
         
        self.transformer_lc = nn.TransformerEncoder(encoder_layer=encoder,num_layers=kwargs['num_encoders'])
        
        self.token_lc = Token(**kwargs)
        self.register_buffer('m_token',torch.ones(1, 1, 1).bool())

    def embedding_light_curve(self, x, t, mask=None, **kwargs):
        x_mod, m_mod, t_mod = self.time_encoder(**{"x": x, "t": t, "mask": mask})
        return torch.cat([self.token_lc(x.shape[0]), x_mod], axis=1),  torch.cat([self.m_token.repeat(x.shape[0],1,1), m_mod], axis=1), t_mod
    
    def forward(self, data, time, mask, **kwargs):
        x_mod, m_mod, _ = self.embedding_light_curve(**{"x": data, "t": time, "mask": mask})
        _ = _.detach()
        with torch.no_grad():
            m_mod =  ~(m_mod.squeeze(-1))
        x_emb = self.transformer_lc(**{"src": x_mod, "src_key_padding_mask":m_mod}) # m_mod.squeeze(-1)
        return x_emb 


class LightCurveClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveClassifier, self).__init__() 
        self.LC = LightCurveTransformer(**kwargs['lc'])
        self.classifier_lc = TokenClassifier(kwargs['lc']['embedding_size'],
                                             num_classes=kwargs['general']['num_classes']
                                             )  
        self.init_model()
        
    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, data,  time, mask=None, **kwargs):
        lc_emb = self.LC(data,time,mask)
        return self.classifier_lc(lc_emb[:,0,:])
    

class LightCurveProjector(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveProjector, self).__init__() 
        self.transformer = LightCurveTransformer(**kwargs)
        self.project = Projector(192,
                                48,
                                48, l2norm = False)
        self.init_model()

    def init_model(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                #nn.init.normal_(p, 0, 0.02)
                nn.init.xavier_normal_(p)
        for p in self.project.parameters():
            if p.dim() > 1:
                #nn.init.normal_(p, 0, 0.02)
                nn.init.xavier_normal_(p)

    def forward(self,**kwargs):
        emb = self.transformer(**kwargs)
        return self.project(emb)[:,0,:]
     