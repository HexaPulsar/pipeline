import torch.nn as nn 

class Transformer(nn.Module):
    def __init__(
        self,
        num_encoders,
        num_heads,
        embedding_size,
        embedding_size_sub,
        dropout = 0.1,
        **kwargs
    ):
        super().__init__()
        
        encoder = nn.TransformerEncoderLayer(d_model = embedding_size,
                                                 nhead=num_heads,
                                                 dim_feedforward=embedding_size_sub,
                                                 activation = 'gelu',
                                                 dropout=dropout,
                                                 batch_first=True,
                                                 norm_first=True)
        '''
        encoder = DifferentialEncoder(d_model = embedding_size,
                                                 nhead=num_heads,
                                                 dim_feedforward=embedding_size_sub,
                                                 activation = 'gelu',
                                                 dropout=dropout,
                                                 batch_first=True,
                                                 norm_first=True)
        '''
        self.stacked_transformers = nn.TransformerEncoder(encoder_layer=encoder,num_layers=num_encoders)
        

    def forward(self, x, src_key_padding_mask):
        return self.stacked_transformers(x, src_key_padding_mask=src_key_padding_mask)