
import torch
import torch.nn as nn
from ..timeEncoders import TimeHandler, TimeHandlerMOD
from ..classifiers import TokenClassifier,MixedClassifier
from ..tokenEmbeddings import Token 
from .projector import Projector



class LightCurveTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveTransformer, self).__init__()
        self.time_encoder = TimeHandler(**kwargs)
        self.transformer_lc = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model = kwargs['embedding_size'],
                                                 nhead=kwargs['num_heads'],
                                                 dim_feedforward=kwargs['embedding_size_sub'],
                                                 activation = 'gelu',
                                                 dropout=0.05,
                                                 batch_first=True,
                                                 norm_first=True,)
                                                 ,num_layers=kwargs['num_encoders'], 
                                                 norm = nn.LayerNorm(kwargs['embedding_size']))
        self.token_lc = Token(**kwargs)
        
    def embedding_light_curve(self, x, t, mask=None, **kwargs):
        x_mod, m_mod, t_mod = self.time_encoder(**{"x": x, "t": t, "mask": mask})
        x_mod = torch.cat([self.token_lc(x.shape[0]), x_mod], axis=1)
        m_mod =  torch.cat([torch.ones(1, 1, 1,device=x.device).bool().repeat(x.shape[0],1,1), m_mod], axis=1)
        return x_mod, m_mod , t_mod
    
    def forward(self, data, time, mask, **kwargs):
        x_mod, m_mod, _ = self.embedding_light_curve(**{"x": data, "t": time, "mask": mask}) 
        m_mod =  ~(m_mod.squeeze(-1))
        
        x_emb = self.transformer_lc(**{"src": x_mod, "src_key_padding_mask":m_mod}) # m_mod.squeeze(-1)
        #x_emb = x_emb / x_emb.norm(dim = 1,keepdim = True)
        return x_emb
    

class LightCurveProjector(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveProjector, self).__init__() 
        self.transformer = LightCurveTransformer(**kwargs)
        self.project = Projector(192,
                            96,
                            96, l2norm = False)
        self.init_model()

    def load_checkpoint(self, checkpoint_path):
        checkpoint_ = torch.load(checkpoint_path)
        weights = OrderedDict()
        for key in checkpoint_["state_dict"].keys():
            if 'projection' in key:
                continue
            else:    
                weights[key.replace("model.transformer.", "")] = checkpoint_["state_dict"][key] 
        self.transformer.load_state_dict(torch.load(checkpoint_path))
        
    def init_model(self): 
        for name,p in self.named_parameters():
            if p.dim( )> 1:
               #nn.init.normal_(p, 0, 0.002)
                nn.init.xavier_normal_(p)
        
     
         
    def forward(self,**kwargs):
        emb = self.transformer(**kwargs)
        projected =self.project(emb[:,0,:]) #.flatten(1)
        return projected
     

class LightCurveClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveClassifier, self).__init__() 
        self.LC = LightCurveTransformer(**kwargs['lc'])
        self.classifier_lc = TokenClassifier(kwargs['lc']['embedding_size'],
                                             num_classes=kwargs['general']['num_classes']
                                             #,dropout=0.01
                                             ) 
        self.classifier_lc = MixedClassifier(kwargs['lc']['embedding_size'],
                                             num_classes=kwargs['general']['num_classes'],
                                             dropout=0.05
                                             )   
        
        self.init_model()
        
    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def load_checkpoint(self, checkpoint_path):
        checkpoint_ = torch.load(checkpoint_path)
        weights = OrderedDict()
        for key in checkpoint_["state_dict"].keys():
            if 'projection' in key:
                continue
            else:    
                weights[key.replace("model.transformer.", "")] = checkpoint_["state_dict"][key] 
        self.LC.load_state_dict(torch.load(checkpoint_path))
        
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
     
