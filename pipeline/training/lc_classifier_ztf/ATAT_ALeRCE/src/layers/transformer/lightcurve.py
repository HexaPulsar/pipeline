import torch
import torch.nn as nn
from ..timeEncoders import TimeHandler 
from ..utils.Token import Token
from dataclasses import dataclass


class LightCurveTransformer(nn.Module):
    def __init__(self,
        input_size= 1,
        embedding_size= 128,
        embedding_size_sub= 512,
        num_heads= 4,
        num_encoders= 3,
        Tmax= 1500.0,
        num_harmonics= 64,
        pe_type= 'tm',
        num_bands= 2,
        dropout = 0.00,
        ):
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.embedding_size_sub = embedding_size_sub
        self.num_heads = num_heads
        self.num_encoders= num_encoders
        self.Tmax = Tmax
        self.num_harmonics = num_harmonics
        self.pe_type = pe_type
        self.num_bands = num_bands
        super().__init__()
        self.time_encoder = TimeHandler(self.num_bands,
                                        self.input_size,
                                        self.embedding_size,
                                        self.Tmax,
                                        self.pe_type)
        self.dropout = nn.Dropout(dropout)
        self.transformer_lc = nn.TransformerEncoder(
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
            #norm=nn.LayerNorm(self.embedding_size),
        )
        self.token_lc = Token(self.embedding_size)
        self.register_buffer('ones', torch.ones(1,1,1,dtype = float))
    def embedding_light_curve(self, x, t, mask=None, **kwargs):
        x_mod, m_mod, t_mod = self.time_encoder(**{"x": x, "t": t, "mask": mask})
        x_mod = torch.cat([self.token_lc(x.shape[0]), x_mod], axis=1)
        m_mod = torch.cat(
            [   self.ones.repeat(x.size(0),1,1).bool(),
                m_mod,
            ],
            axis=1,
        )
        return x_mod, m_mod, t_mod

    def forward(self, data, time, mask, **kwargs):
        x_mod, m_mod, _ = self.embedding_light_curve(
            **{"x": data, "t": time, "mask": mask}
        )
        x_mod = x_mod /x_mod.norm(dim = 1,keepdim=True)
        x_emb = self.transformer_lc(
            **{"src": x_mod, "src_key_padding_mask": ~(m_mod.squeeze(-1))}
        )[:, 0, :]
        return self.dropout(x_emb)


class LightCurveTransformerALL2(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveTransformerALL2, self).__init__()
        # TODO add assert
        assert kwargs["embedding_size"] % kwargs["num_heads"] == 0
        self.sequence_section = int(kwargs["embedding_size"] / kwargs["num_heads"])
        self.time_encoder = TimeHandler(**kwargs)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.sequence_section,
            nhead=1,
            dim_feedforward=4*self.sequence_section,
            activation="gelu",
            dropout=0.05,
            batch_first=True,
            norm_first=True,
        )
        self.num_heads = kwargs["num_heads"]
        trans = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=kwargs['num_encoders'],
            norm=nn.LayerNorm(self.sequence_section),
        )
        self.encoder_list = nn.ModuleList([trans for _ in range(self.num_heads)])
        self.tokens = nn.ModuleList(
            [Token(self.sequence_section) for _ in range(self.num_heads)]
        )
        
        self.out_norm = nn.LayerNorm(kwargs["embedding_size"])
        self.register_buffer("ones", torch.ones(1, 1, 1, dtype=bool))
    def split(self, x_mod):
        x_mod = x_mod.unsqueeze(-1).reshape(
            x_mod.size(0), x_mod.size(1), self.sequence_section, self.num_heads
        )
        return x_mod

    def embedding_light_curve(self, x, t, mask=None, **kwargs):
        x_mod, m_mod, t_mod = self.time_encoder(**{"x": x, "t": t, "mask": mask})
        m_mod = torch.cat(
            [
                self.ones.expand(x.shape[0], 1, 1),
                m_mod,
            ],
            axis=1,
        )
        return x_mod, m_mod, t_mod

    def add_token(self, x_mod, i):
        return torch.cat([self.tokens[i](x_mod.shape[0]), x_mod], axis=1)

    def forward(self, data, time, mask, **kwargs):
        x_mod, m_mod, _ = self.embedding_light_curve(
            **{"x": data, "t": time, "mask": mask}
        )
        tokens = torch.empty((x_mod.size(0),self.sequence_section,self.num_heads),device = x_mod.device)
        x_mod = x_mod.unsqueeze(-1).reshape(
            x_mod.size(0), x_mod.size(1), self.sequence_section, self.num_heads
        )
        for i in range(self.num_heads):
            in_ = self.add_token(x_mod[:, :, :, i], i)
            tokens[:,:,i] = self.encoder_list[i](
                    src=in_, src_key_padding_mask=~(m_mod.squeeze(-1))
                )[:, 0, :]
        #return self.out_norm(self.out_FC(out))
        return self.out_norm(tokens.flatten(1))

class SectionEmbeddingAttention(nn.Module):
    def __init__(self,d_model,
            nhead,
            dim_feedforward,
            activation="gelu",
            dropout=0.2,
            n_sections = 4,
            batch_first=True,
            norm_first=True,):
        super().__init__()
        self.batch_first =True
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        self.self_attn = self.attn
        self.FC = nn.Linear(n_sections*d_model,n_sections*d_model)

    
    def forward(self,query,key,value,
                key_padding_mask, need_weights=True, 
                attn_mask=None, 
                average_attn_weights=True,
                is_causal=False):
        attn_output, _ = self.attn(query, key, value, key_padding_mask = key_padding_mask)  # (bsz * embed_sec, seqlen, sec_n)
        # Reshape back to original form

        attn_output = attn_output.view(query.size(0)//4, 4, 401, 32)  # (bsz, embed_sec, seqlen, sec_n)
        # Permute back to original shape (bsz, seqlen, 32, 4)
        attn_output = attn_output.permute(0, 2, 3, 1)
        out = self.FC(attn_output.flatten(2))
        out = out.reshape(query.shape)

        return out
    
    
class LightCurveTransformerALL(nn.Module):
    def __init__(self, **kwargs):
        super(LightCurveTransformerALL, self).__init__()
        # TODO add assert

        self.time_encoder = TimeHandler(**kwargs)
        self.num_heads = kwargs["num_heads"]
        # trans = nn.TransformerEncoder(
        #    encoder_layer=encoder_layer,
        #    num_layers=kwargs["num_encoders"],
        #    norm=nn.LayerNorm(self.sequence_section),
        # )
        assert (
            kwargs["embedding_size"] % kwargs["num_heads"] == 0
        ), "embedding_size not div by numheads"
        self.embed_sec = kwargs["embedding_size"] // kwargs["num_heads"]
        self.sec_n = kwargs["num_heads"]
        self.learnable_tokens = nn.Parameter(
            torch.randn(1, 1, self.embed_sec, self.sec_n)
        )  # (1, 1, 32, 4)
        encoder_layer =  encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_sec,
            nhead=1,
            dim_feedforward=8*self.embed_sec,
            activation="gelu",
            dropout=0.05,
            batch_first=True,
            norm_first=True,
        )
        attn = SectionEmbeddingAttention(d_model=self.embed_sec,
            nhead=1,
            dim_feedforward=4*kwargs["embedding_size"]//kwargs['num_heads'],
            activation="gelu",
            dropout=0.2,
            batch_first=True,
            norm_first=True,)
        encoder_layer.self_attn = attn
        self.register_buffer("ones", torch.ones(1, 1, 1, dtype=bool))
        self.transformer_lc = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            # norm=nn.LayerNorm(self.embed_sec),
        )
        self.out_norm = nn.LayerNorm(kwargs["embedding_size"])

    def embedding_light_curve(self, x, t, mask=None, **kwargs):
        x_mod, m_mod, t_mod = self.time_encoder(**{"x": x, "t": t, "mask": mask})
        m_mod = torch.cat(
            [
                self.ones.expand(x.shape[0], 1, 1),
                m_mod,
            ],
            axis=1,
        )
        return x_mod, m_mod, t_mod

    def forward(self, data, time, mask, **kwargs):
        x_mod, m_mod, _ = self.embedding_light_curve(
            **{"x": data, "t": time, "mask": mask}
        )
        x_mod = x_mod.unsqueeze(-1).view(
            x_mod.size(0), x_mod.size(1), self.embed_sec, self.num_heads
        )
        tokens = self.learnable_tokens.expand(
            x_mod.size(0), -1, -1, -1
        )  # Expand to match batch size
        x_mod = torch.cat([tokens, x_mod], dim=1)  # (bsz, 401, 32, 4)
        
        x_mod = x_mod.permute(0, 3, 1, 2)  # (bsz, embed_sec, 401, sec_n)
        bsz, sec_n,seqlen, embed_sec  = x_mod.shape
        x_mod_reshaped = x_mod.reshape(bsz * sec_n, seqlen, embed_sec)
        output = self.transformer_lc(
            src=x_mod_reshaped,
            src_key_padding_mask=~(m_mod.squeeze(-1).repeat(self.sec_n, 1)),
        )  # (bsz * embed_sec, 401, sec_n)
        output = output.view(
            data.size(0), self.sec_n, x_mod.size(2), self.embed_sec
        )  # (bsz, embed_sec, 401, sec_n)
        output = output.permute(0, 2, 3, 1)

        return self.out_norm(output[:, 0, :, :].flatten(1))