import torch
import torch.nn as nn
from ..classifiers import TokenClassifier, MixedClassifier

from ..embeddings import Embedding
from ..tokenEmbeddings import Token
from .projector import VICRegProjector,CLIPProjector



class TabularTransformer(nn.Module):
    def __init__(self, **kwargs):

        super(TabularTransformer, self).__init__()
        self.embedding_tab = Embedding(
            **kwargs
        )  # nn.Linear(kwargs['TAB_ARGS']['length_size'],kwargs['TAB_ARGS']['embedding_size']) #
        self.transformer_tab = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=kwargs["embedding_size"],
                nhead=kwargs["num_heads"],
                dim_feedforward=kwargs["embedding_size_sub"],
                activation="gelu",
                dropout=0.00,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=kwargs["num_encoders"],
            norm=nn.LayerNorm(kwargs["embedding_size"]),
        )
        self.token_tab = Token(**kwargs)
        self.register_buffer("m_token", torch.ones(1, 1, 1).bool())

    def embedding_feats(self, f, tab_mask=None):
        f_mod = self.embedding_tab(**{"f": f})
        if tab_mask is not None:
            return torch.cat([self.token_tab(f.shape[0]), f_mod], axis=1), torch.cat(
                [self.m_token.repeat(tab_mask.shape[0], 1, 1), tab_mask], axis=1
            )

        return torch.cat([self.token_tab(f.shape[0]), f_mod], axis=1), None

    def forward(self, tabular_feat, tab_mask=None, **kwargs):
        f_mod, tab_mask = self.embedding_feats(
            **{"f": tabular_feat, "tab_mask": tab_mask}
        )
        if tab_mask is not None:
            tab_mask = ~(tab_mask.squeeze(-1))
            # print(mask.shape)
        f_emb = self.transformer_tab(**{"src": f_mod, "src_key_padding_mask": tab_mask})
        return f_emb[:,0,:]


class TabularClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(TabularClassifier, self).__init__()
        self.TAB = TabularTransformer(**kwargs["tab"])
        self.classifier_tab = MixedClassifier(
            kwargs["tab"]["embedding_size"],
            num_classes=kwargs["general"]["num_classes"],
            dropout=0.1,
        )
        self.init_model()

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, tabular_feat, tab_mask=None, **kwargs):
        tab_emb = self.TAB(tabular_feat, tab_mask)
        return self.classifier_tab(tab_emb)
