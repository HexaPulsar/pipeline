from collections import OrderedDict
import glob
import torch
import torch.nn as nn
import numpy as np
from .transformer import FeedForward 
from .timeEncoders import TimeHandler, TimeHandlerMOD
from .embeddings import Embedding 
from .transformer.torchimpl import Transformer
from .classifiers import TokenClassifier, MixedClassifier
from .tokenEmbeddings import Token 

class AlignProjector(nn.Module):
    def __init__(self,input_size,output_size,**kwargs):
        super(AlignProjector,self).__init__()
        self.kwargs  = kwargs
        self.projection = nn.Sequential(nn.Linear(input_size,output_size,bias=False))
             
    def forward(self,embedding):
        embedding =   embedding / embedding.norm(dim=1, keepdim=True)
        embedding = self.projection(embedding)
        embedding =   embedding / embedding.norm(dim=1, keepdim=True)
        return embedding

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
       # m_mod =  #invert mask so its compatible with native torch transformer masking 
        x_emb = self.transformer_lc(**{"src": x_mod, "src_key_padding_mask": ~(m_mod.squeeze(-1))}) # m_mod.squeeze(-1)
        return x_emb[:,0,:]

class TabularTransformer(nn.Module):
    def __init__(self, **kwargs):
        print(kwargs)
        super(TabularTransformer, self).__init__()
        self.embedding_ft =   Embedding(**kwargs) #nn.Linear(kwargs['TAB_ARGS']['length_size'],kwargs['TAB_ARGS']['embedding_size']) #
        encoder = nn.TransformerEncoderLayer(d_model = kwargs['embedding_size'],
                                                 nhead=kwargs['num_heads'],
                                                 dim_feedforward=kwargs['embedding_size_sub'],
                                                 activation = 'gelu',
                                                 dropout=0.01,
                                                 batch_first=True,
                                                 norm_first=True)
         
        self.transformer_ft= nn.TransformerEncoder(encoder_layer=encoder,num_layers=kwargs['num_encoders'])
        
        self.token_ft = Token(**kwargs) 
        
    def embedding_feats(self, f):
        f_mod = self.embedding_ft(**{"f": f}) 
        return torch.cat([self.token_ft(f.shape[0]), f_mod], axis=1) 
    
    def forward(self,tabular_feat, **kwargs): 
        f_mod = self.embedding_feats(**{"f": tabular_feat})
        f_emb = self.transformer_ft(**{"src": f_mod, 'src_key_padding_mask':None}) 
        return f_emb[:,0,:]

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
        return self.classifier_lc(lc_emb)

class TabularClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(TabularClassifier, self).__init__()
        self.TAB = TabularTransformer(**kwargs['ft'])
        self.classifier_tab = TokenClassifier(kwargs['ft']['embedding_size'],
                                             num_classes=kwargs['general']['num_classes'],
                                             )  
        self.init_model()
    def init_model(self):
                for p in self.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_normal_(p)
    def forward(self, tabular_feat, **kwargs):
        tab_emb = self.TAB(tabular_feat)
        return self.classifier_tab(tab_emb)
    

class SingleBranch(nn.Module):
    def __init__(self,type:str, **kwargs):
            super(SingleBranch, self).__init__()
            self.kwargs = kwargs
            self.finetune = False
            if type == 'lc':
                self.transformer = LightCurveTransformer(**kwargs) 

                if not self.finetune:
                    
                    self.project = Projector(192,
                                             192,
                                             48, l2norm = False) 
            else:
                self.transformer = TabularTransformer(**kwargs) 
                if not self.finetune:
                    self.project = Projector(128,
                                             48,
                                             48,
                                             l2norm = False) 
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
        return self.project(emb)
    
    def embeddings(self,**kwargs):
        emb = self.transformer(**kwargs)
        return emb    

class ATAT(nn.Module):
    def __init__(self, **kwargs):
        super(ATAT, self).__init__()
        self.kwargs = kwargs 
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]
        self.LC = LightCurveTransformer(**self.lightcv_)
        self.TAB = TabularTransformer(**self.feature_)
        
        # Lightcurve Transformer
        if self.general_["use_lightcurves"]:
             
            self.classifier_lc = TokenClassifier(
                num_classes=self.general_["num_classes"], **kwargs["lc"]
            )
        # Tabular Transformer
        if self.general_["use_metadata"] or self.general_["use_features"]:
            self.classifier_tab = TokenClassifier(
                num_classes=self.general_["num_classes"], **kwargs["ft"]
            )
             

        # Mixed Classifier (Lightcurve and tabular)
        if self.general_["use_lightcurves"] and any(
            [self.general_["use_metadata"], self.general_["use_features"]]
        ):

            input_dim = kwargs["lc"]["embedding_size"] + kwargs["ft"]["embedding_size"]
            self.classifier_mix = MixedClassifier(
                input_dim=input_dim, **kwargs["general"]
            )

        # init model params
        self.init_model()
        
    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                
    def forward(
        self,
        data=None,
        data_err=None,
        time=None,
        tabular_feat=None,
        mask=None,
        **kwargs
        ):
        x_cls, f_cls, m_cls = None, None, None

        if self.general_["use_lightcurves"]:
            if self.general_["use_lightcurves_err"]:
                data = torch.stack((data, data_err), dim=data.dim() - 1)

            
            x_emb = self.LC(data,time,mask)
            x_cls = self.classifier_lc(x_emb)

        if self.general_["use_metadata"] or self.general_["use_features"]:
            
            f_emb = self.TAB(tabular_feat)
            f_cls = self.classifier_tab(f_emb)

        if self.general_["use_lightcurves"] and (
            self.general_["use_metadata"] or self.general_["use_features"]
        ):
            m_cls = self.classifier_mix(
                torch.cat([f_emb, x_emb], axis=1)
            )

        return x_cls, f_cls, m_cls

class cATAT(nn.Module):
    def __init__(self, **kwargs):
        super(cATAT, self).__init__()
        self.kwargs = kwargs 
        
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

  
        self.LC = LightCurveTransformer(**self.lightcv_)
        self.TAB = TabularTransformer(**self.feature_)
        
        # pretraining parameters
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07)) # np.log(kwargs['CYCLIP']['initial_temperature'])
        self.project_lc = AlignProjector(self.lightcv_['embedding_size'],128) #TODO: add tu custom_parser arguments
        self.project_ft = AlignProjector(self.feature_['embedding_size'],128)

        self.init_model()
        
    def init_model(self):
        for p in self.LC.parameters():
            if p.dim() > 1:
                #nn.init.normal_(p, 0, 0.02)
                nn.init.xavier_normal_(p)
        for p in self.TAB.parameters():
            if p.dim() > 1:
                #nn.init.normal_(p, 0, 0.02)
                nn.init.xavier_normal_(p)
        for p in self.project_ft.parameters():
            if p.dim() > 1:
                #nn.init.normal_(p, 0, 0.02)
                nn.init.xavier_normal_(p)
        for p in self.project_lc.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(
        self,
        data=None,
        data_err=None,
        time=None,
        tabular_feat=None,
        mask=None,
        **kwargs
    ):
         
        if self.general_["use_lightcurves"]:
            if self.general_["use_lightcurves_err"]:
                data = torch.stack((data, data_err), dim=data.dim() - 1)
 
            x_emb = self.LC(**{"data": data, "time": time, "mask": mask})
           

        if self.general_["use_metadata"] or self.general_["use_features"]:
            
            f_emb = self.TAB(**{"tabular_feat": tabular_feat})
              
        x_emb = self.project_lc(x_emb)
        f_emb = self.project_ft(f_emb)

        return x_emb,f_emb
 