from collections import OrderedDict
import glob
import torch
import torch.nn as nn
import numpy as np
from .transformer import FeedForward 
from .timeEncoders import TimeHandler
from .embeddings import Embedding
from .transformer import Transformer 
from .classifiers import TokenClassifier, MixedClassifier
from .tokenEmbeddings import Token 

class AlignProjector(nn.Module):
    def __init__(self,input_size,output_size, loss_type = 'CyCLIP',**kwargs):
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
            embedding =   embedding / embedding.norm(dim=1, keepdim=True)
            embedding = self.projection(embedding)
            embedding =   embedding / embedding.norm(dim=1, keepdim=True)
        else:
            embedding = self.projection(embedding)
        return embedding



class LightCurveTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveTransformer, self).__init__()
        self.kwargs  = kwargs
        self.time_encoder = TimeHandler(**kwargs)
        self.transformer_lc = Transformer(**kwargs)
        self.token_lc = Token(**kwargs) 

    def embedding_light_curve(self, x, t, mask=None, **kwargs):
        batch_size = x.shape[0]
        m_token = torch.ones(batch_size, 1, 1).float().to(x.device) 
        x_mod, m_mod, t_mod = self.time_encoder(**{"x": x, "t": t, "mask": mask})
        x_mod = torch.cat([self.token_lc(batch_size), x_mod], axis=1)
        m_mod = torch.cat([m_token, m_mod], axis=1)
        return x_mod, m_mod, t_mod

    def forward(self, data, time, mask, **kwargs):

        x_mod, m_mod, _ = self.embedding_light_curve(**{"x": data, "t": time, "mask": mask})
        x_emb = self.transformer_lc(**{"x": x_mod, "mask": m_mod}) # m_mod.squeeze(-1)
        x_emb =x_emb[:,0,:]  
        return x_emb

class TabularTransformer(nn.Module):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super(TabularTransformer, self).__init__()
        self.embedding_ft =   Embedding(**kwargs) #nn.Linear(kwargs['TAB_ARGS']['length_size'],kwargs['TAB_ARGS']['embedding_size']) #
        self.transformer_ft = Transformer(**kwargs) #TabTransformer() #
 
        self.token_ft = Token(**kwargs) 
        

    def embedding_feats(self, f):
        batch_size = f.shape[0] 
        f_mod = self.embedding_ft(**{"f": f}) 
        f_mod = torch.cat([self.token_ft(batch_size), f_mod], axis=1) 
        
        return f_mod 
    
    def forward(self,f, **kwargs): 
        f_mod = self.embedding_feats(**{"f": f})
        f_emb = self.transformer_ft(**{"x": f_mod, 'mask':None}) 
        f_emb = f_emb / f_emb.norm(dim=1, keepdim=True)
        f_emb = f_emb[:,0,:]  

        return f_emb

class LightCurveClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveClassifier, self).__init__() 
        self.LC = LightCurveTransformer(**kwargs['lc'])
        self.classifier_lc = MixedClassifier(input_dim=kwargs['lc']['embedding_size'],
                                             num_classes=kwargs['general']['num_classes'],
                                             )   
        
    def init_model(self):
            for p in self.classifier_lc.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
            for p in self.LC.parameteself.init_model():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
                     
    def forward(self, data,  time, mask=None, **kwargs):
        lc_emb = self.LC(data,time,mask)
        class_emb = self.classifier_lc(lc_emb)
        return class_emb

class TabularClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(TabularClassifier, self).__init__()
        self.TAB = TabularTransformer(**kwargs)
        self.classifier_tab = MixedClassifier(128,**kwargs)   

    def forward(self, data,  time, mask=None, **kwargs):
        tab_emb = self.TAB(data,time,mask)
        class_emb = self.classifier_tab(tab_emb)
        return class_emb
    


class SingleBranch(nn.Module):
    def __init__(self,type:str, **kwargs):
            super(SingleBranch, self).__init__()
            self.kwargs = kwargs
            self.finetune = False
            if type == 'lc':
                self.transformer = LightCurveTransformer(**kwargs) 

                if not self.finetune:
                    
                    self.project = Projector(192,
                                             48,
                                             48, l2norm = False) 
            else:
                self.transformer = TabularTransformer(**kwargs) 
                if not self.finetune:
                    self.project = Projector(128,
                                             512,
                                             128,
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
        #emb = emb.view(emb.size(0), -1)
        emb = self.project(emb)
        return emb
    
    def embeddings(self,**kwargs):
        emb = self.transformer(**kwargs)
        return emb    



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
        self.project_lc = Projector(self.lightcv_['embedding_size'],128) #TODO: add tu custom_parser arguments
        self.project_ft = Projector(self.feature_['embedding_size'],128)
 

        self.init_model()
        
    def init_model(self): #TODO corregir init por partes para tmfs y projectors
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0, 0.02)
                #nn.init.xavier_normal_(p)



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
 
            x_emb = self.LC(**{"x": data, "t": time, "mask": mask})
           

        if self.general_["use_metadata"] or self.general_["use_features"]:
            
            f_emb = self.TAB(**{"f": tabular_feat})
              
        x_emb = self.project_lc(x_emb)
        f_emb = self.project_ft(f_emb)

        return x_emb,f_emb
 