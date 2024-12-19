import torch.nn as nn
from .TransformerEncoder import TransformerEncoder
from .TransformerEncoderCnn import TransformerEncoderCnn


class Transformer(nn.Module):
    def __init__(self, encoder_type="Linear", **kwargs):
        super().__init__()
        self.stacked_transformers = nn.ModuleList([TransformerEncoder(**kwargs)
            if encoder_type == "Linear"
            else TransformerEncoderCnn(**kwargs) for _ in range(kwargs["num_encoders"])])

    def forward(self, x, src_key_padding_mask, **kwargs):
        for l in self.stacked_transformers:
            x = l(x, src_key_padding_mask)
        return x
