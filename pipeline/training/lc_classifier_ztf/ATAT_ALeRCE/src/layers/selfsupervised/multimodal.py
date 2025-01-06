import torch
import torch.nn as nn

from ..classifiers import TokenClassifier,MixedClassifier
from .lightcurve import LightCurveTransformer
from .tabular import TabularTransformer


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
        #for p in self.classifier_lc.parameters():
        #    if p.dim() > 1:
        #        nn.init.xavier_normal_(p)
        #for p in self.classifier_mix.parameters():
         #   if p.dim() > 1:
        #       nn.init.xavier_normal_(p)
        #for p in self.classifier_tab.parameters():
        #    if p.dim() > 1:
        #        nn.init.xavier_normal_(p)
                
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
            x_cls = self.classifier_lc(x_emb[:,0,:])

        if self.general_["use_metadata"] or self.general_["use_features"]:
            
            f_emb = self.TAB(tabular_feat)
            f_cls = self.classifier_tab(f_emb[:,0,:])

        if self.general_["use_lightcurves"] and (
            self.general_["use_metadata"] or self.general_["use_features"]
        ):
            m_cls = self.classifier_mix(
                torch.cat([f_emb[:,0,:], x_emb[:,0,:]], axis=1)
            )

        return x_cls, f_cls, m_cls
