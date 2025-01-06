import torch
import torch.nn as nn
from .lightcurve import LightCurveTransformer
from .tabular import TabularTransformer



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
    


class ATATAlignment(nn.Module):
    def __init__(self, **kwargs):
        super(ATATAlignment, self).__init__()
        self.kwargs = kwargs 
        
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

  
        self.LC = LightCurveTransformer(**self.lightcv_)
        self.TAB = TabularTransformer(**self.feature_)
        
        # pretraining parameters
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(1/0.07)) # np.log(kwargs['CYCLIP']['initial_temperature'])
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
 