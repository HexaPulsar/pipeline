import torch
import torch.nn as nn

from ..classifiers import TokenClassifier, MixedClassifier
from .lightcurve import LightCurveTransformer
from .tabular import TabularTransformer
from .projector import VICRegProjector,CLIPProjector

 


class ATATProjector(nn.Module):
    def __init__(self, type,**kwargs):
        super(ATATProjector, self).__init__()
        self.kwargs = kwargs
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["tab"]
        self.LC = LightCurveTransformer(**self.lightcv_)
        self.TAB = TabularTransformer(**self.feature_)
        self.project_lc =  VICRegProjector(128, 32, 32, l2norm=False) if type == type else CLIPProjector(128,256)
        self.project_tab =   VICRegProjector(128, 32, 32, l2norm=False) if type == type else CLIPProjector(128,256)
        # init model params
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07))) # np.log(kwargs['CYCLIP']['initial_temperature'])
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
        x_emb = self.LC(data, time, mask)
        f_emb = self.TAB(tabular_feat)
        x_emb = self.project_lc(x_emb)
        f_emb = self.project_tab(f_emb)
        return x_emb, f_emb


class ATATClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(ATATClassifier, self).__init__()
        self.kwargs = kwargs
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["tab"]
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
                num_classes=self.general_["num_classes"], **kwargs["tab"]
            )

        # Mixed Classifier (Lightcurve and tabular)
        if self.general_["use_lightcurves"] and any(
            [self.general_["use_metadata"], self.general_["use_features"]]
        ):
            input_dim = kwargs["lc"]["embedding_size"] + kwargs["tab"]["embedding_size"]
            self.classifier_mix = MixedClassifier(
                input_dim=input_dim, **kwargs["general"],dropout = 0.3
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

            x_emb = self.LC(data, time, mask)
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
