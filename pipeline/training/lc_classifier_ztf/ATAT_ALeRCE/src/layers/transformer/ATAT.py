import torch
import torch.nn as nn


from .timeEncoders import TimeHandler
from .embeddings import Embedding
#from .transformer import Transformer
from .transformer.torchimpl import Transformer
from .classifiers import TokenClassifier, MixedClassifier
from .tokenEmbeddings import Token
from .lightcurve import LightCurveTransformer 
from .tabular import TabularTransformer

class ATAT(nn.Module):
    def __init__(self,experiment_type:str,
                 lc_args:dict = None, 
                 tab_args:dict = None, 
                 classifier_args: dict = None, **kwargs):
        super(ATAT, self).__init__()
        self.modalities = experiment_type.split('_')

        # Lightcurve Transformer
        if self.general_["use_lightcurves"]:
            self.transformer_lc = LightCurveTransformer()
        # Tabular Transformer
        if self.general_["use_metadata"] or self.general_["use_features"]:
            self.transformer_ft = TabularTransformer()
             

        # Mixed Classifier (Lightcurve and tabular)
        if self.general_["use_lightcurves"] and any(
            [self.general_["use_metadata"], self.general_["use_features"]]
        ):

            input_dim = kwargs["lc"]["embedding_size"] + kwargs["ft"]["embedding_size"]
            self.classifier_mix = MixedClassifier(
                input_dim=input_dim, **kwargs["general"]
            )
  

    def forward(
        self,
        data=None,
        data_err=None,
        time=None,
        tabular_feat=None,
        mask=None,
        **kwargs
    ):
        output= {} 
        if 'LC' in self.modalities:
            if self.general_["use_lightcurves_err"]:
                data = torch.stack((data, data_err), dim=data.dim() - 1)

            x_mod, m_mod, _ = self.embedding_light_curve(
                **{"x": data, "t": time, "mask": mask}
            )
            x_emb = self.transformer_lc(**{"x": x_mod, "mask": ~(m_mod).unsqueeze(-1).bool()})
            output['LC'] =  self.classifier_lc(x_emb)
        if 'MD' in self.modalities or 'FEAT' in self.modalities:
            f_mod = self.embedding_feats(**{"f": tabular_feat})
            f_emb = self.transformer_ft(**{"x": f_mod, "mask": None})
            output['TAB'] = self.classifier_ft(f_emb)
            
        if all(['LC' in self.modalities, 
                ('MD' in self.modalities or 'FEAT' in self.modalities)]):
            output['MM'] =self.classifier_mix(
                torch.cat([f_emb, x_emb], axis=1)
            )
        return output

    def predict_mix(self, data, time, tabular_feat, mask, **kwargs):
        return
        x_mod, m_mod, _ = self.embedding_light_curve(
            **{"x": data, "t": time, "mask": mask}
        )
        x_emb = self.transformer_lc(**{"x": x_mod, "mask": m_mod})

        f_mod = self.embedding_feats(**{"f": tabular_feat})
        f_emb = self.transformer_ft(**{"x": f_mod, "mask": None})

        m_cls = self.classifier_mix(torch.cat([f_emb[:, 0, :], x_emb[:, 0, :]], axis=1))
        m_cls = torch.softmax(m_cls, dim=1)
        return m_cls

    def predict_lc(self, data, time, mask, **kwargs):
        return
        x_mod, m_mod, _ = self.embedding_light_curve(
            **{"x": data, "t": time, "mask": mask}
        )
        x_emb = self.transformer_lc(**{"x": x_mod, "mask": m_mod})
        x_cls = self.classifier_lc(x_emb[:, 0, :])
        x_cls = torch.softmax(x_cls, dim=1)
        return x_cls

    def predict_tab(self, tabular_feat, **kwargs):
        return
        f_mod = self.embedding_feats(**{"f": tabular_feat})
        f_emb = self.transformer_ft(**{"x": f_mod, "mask": None})
        f_cls = self.classifier_ft(f_emb[:, 0, :])
        f_cls = torch.softmax(f_cls, dim=1)
        return f_cls

    def change_clf(self, num_nuevas_clases=22):
        # Reemplazar el clasificador light curve
        embedding_size_lc = self.classifier_lc.output_layer.in_features
        self.classifier_lc = TokenClassifier(embedding_size_lc, num_nuevas_clases)
