import torch
import torch.nn as nn
from ..timeEncoders import TimeHandler

class Token(nn.Module):
    def __init__(self, embedding_size,dropout, **kwargs):
        super(Token, self).__init__()

        # self.token = nn.parameter.Parameter(
        #    torch.rand(embedding_size), requires_grad=True
        # )
        self.dropout = nn.Dropout(dropout)
        self.token = nn.Parameter(torch.zeros(embedding_size), requires_grad=True)
        #self.token = nn.Parameter(torch.randn(embedding_size) * 0.02, requires_grad=True)
        #self.token = nn.Parameter(torch.rand(embedding_size), requires_grad=True)
        #self.ln = nn.Sequential() #nn.LayerNorm(embedding_size)
    def forward(self, n_batch):

        #return self.ln(self.dropout(self.token.repeat(n_batch, 1, 1).permute(2,1,0)).permute(2,1,0))
        return self.token.repeat(n_batch, 1, 1)


class LightCurveTransformer(nn.Module):
    def __init__(
        self,
        input_size=1,
        embedding_size=128,
        embedding_size_sub=128,
        num_heads=4,
        num_encoders=3,
        Tmax=1500.0,
        num_harmonics=64,
        pe_type="tm",
        num_bands=2,
        dropout=0.00,
        checkpoint=None,
        freeze_weights=False,
        use_velocity=False,
        use_acceleration=False,
        use_stats=False,
        use_metadata=False,
        use_features=False,
        metadata_num_features=6,
        features_num_features=181,
        use_sequence_norm=False,
        use_timefilm_norm=False,
        use_timefilm_gelu=False,
        use_exp: bool = False,
        use_conv:bool = False,
        use_tabular_transformer=False,
    ):
        super().__init__()

        self.time_encoder = TimeHandler(
            num_bands=num_bands,
            input_size=input_size,
            embedding_size=embedding_size,
            projections_inner_size=embedding_size,
            Tmax=Tmax,
            metadata_num_features=metadata_num_features,
            features_num_features=features_num_features,
            use_acceleration=use_acceleration,
            use_velocity=use_velocity,
            use_stats=use_stats,
            use_metadata=use_metadata,
            use_features=use_features,
            use_tabular_transformer=use_tabular_transformer,
            use_timefilm_gelu=use_timefilm_gelu,
            use_timefilm_norm=use_timefilm_norm,
            use_exp=use_exp,
            use_conv = use_conv,
        )

        self.transformer_lc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=num_heads,
                dim_feedforward=embedding_size_sub,
                activation="gelu",
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_encoders,
             norm=nn.LayerNorm(embedding_size),
        )
        self.token_lc = Token(embedding_size, 0.0)
        self.register_buffer("ones", torch.ones(1, 1, 1, dtype=torch.bool))
        self.sequence_norm = use_sequence_norm
        self.dropout = nn.Dropout(dropout)

    def load_weights(self):
        pass

    def embedding_light_curve(self, x, t, mask, metadata=None, features=None, **kwargs):

        x_mod, m_mod, t_mod = self.time_encoder(
            x, t, mask=mask, metadata=metadata, features=features
        )

        if self.sequence_norm:
            x_mod = x_mod / (
            torch.sqrt(torch.linalg.norm(x_mod, dim=(1), keepdim=True)) + 1e-8
            )
        x_mod[x_mod == 0] = -1e9
        x_mod = torch.cat([self.token_lc(x.shape[0]), x_mod], axis=1)
        m_mod = torch.cat(
            [
                self.ones.repeat(x.size(0), 1, 1),
                m_mod,
            ],
            axis=1,
        )
        assert m_mod.dtype == torch.bool, "m_mod type is {}".format(m_mod.dtype)
        return x_mod, m_mod, t_mod

    def forward(self, data, time, mask, metadata=None, features=None, **kwargs):

        x_mod, m_mod, _ = self.embedding_light_curve(
            x=data, t=time, mask=mask, metadata=metadata, features=features
        )
        x_emb = self.transformer_lc(
            src=x_mod, src_key_padding_mask=~(m_mod.squeeze(-1))
        )
        return x_emb#[:, 0, :]


class Embedding(nn.Module):
    def __init__(self, length_size, embedding_size, dropout, **kwargs):
        super(Embedding, self).__init__()
        #self.tab_W_feat = nn.Parameter(torch.randn(1, length_size, embedding_size))
        #self.tab_b_feat = nn.Parameter(torch.randn(1, length_size, embedding_size))
        self.tab_W_feat = nn.Parameter(torch.zeros(1, length_size, embedding_size))
        self.tab_b_feat = nn.Parameter(torch.zeros(1, length_size, embedding_size))
        #self.dropout = nn.Dropout(dropout)
    def forward(self, f):
        return self.tab_W_feat * f + self.tab_b_feat


class TabularTransformer(nn.Module):
    def __init__(
        self,
        embedding_size=128,
        embedding_size_sub=512,
        num_heads=4,
        num_encoders=3,
        length_size=6,
        num_bands=2,
        dropout=0.00,
        checkpoint=None,
        freeze_weights=False,
        sequence_norm=False,
    ):

        self.embedding_size = embedding_size
        self.embedding_size_sub = embedding_size_sub
        self.num_heads = num_heads
        self.num_encoders = num_encoders
        self.num_bands = num_bands
        self.length_size = length_size

        super().__init__()
        self.embedding_tab = Embedding(
            self.length_size, self.embedding_size,
            dropout
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
            # norm=nn.LayerNorm(self.embedding_size),
        )
        self.token_tab = Token(self.embedding_size, dropout)
        self.register_buffer("ones", torch.ones(1, 1, dtype=bool))
        self.dropout = nn.Dropout(dropout)
        self.sequence_norm = sequence_norm

    def embedding_feats(self, f):
        f_mod = self.embedding_tab(**{"f": f.unsqueeze(-1)})

        if self.sequence_norm:
            f_mod = f_mod / (
                torch.sqrt(torch.linalg.norm(f_mod, dim=(1), keepdim=True)) + 1e-8
            )

        f_mod = torch.cat([self.token_tab(f.shape[0]), f_mod], axis=1)

        return f_mod

    def forward(self, tabular_feat, tab_mask=None, **kwargs):

        f_mod = self.embedding_feats(**{"f": tabular_feat})
        if tab_mask is not None:
            tab_mask = torch.cat(
            [
                self.ones.repeat(f_mod.size(0),1),
                tab_mask,
            ],
            axis=1,
        )
        # assert tab_mask is not None
        f_emb = self.transformer_tab(**{"src": f_mod, "src_key_padding_mask": ~tab_mask if tab_mask is not None else tab_mask})
        return self.dropout(f_emb)

class Combinator(nn.Module):
    def __init__(self, lc_model, tab_model, how="concat", as_dict=True):
        super().__init__()
        self.transformer_lc = lc_model if lc_model is not None else None
        self.transformer_tab = tab_model if tab_model is not None else None
        self.how = how
        self.as_dict = as_dict

    def forward(
        self,
        data,
        time,
        mask,
        tabular_feat=None,
        metadata_feat=None,
        extracted_feat=None,
        tab_mask = None,
        **kwargs
    ):

        lc_emb = self.transformer_lc(data, time, mask)
        ft_emb = self.transformer_tab(tabular_feat, tab_mask)
        # return torch.concat([lc_emb,ft_emb],axis  = -1)

        if self.as_dict:
            return {
                "LC": lc_emb,
                "TAB": ft_emb,
                #"MIX": torch.concat([lc_emb, ft_emb], axis=-1),
            }
#
