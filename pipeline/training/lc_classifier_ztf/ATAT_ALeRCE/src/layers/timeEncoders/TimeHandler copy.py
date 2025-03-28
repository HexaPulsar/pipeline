import torch
import torch.nn as nn 
from .TimeFilmModified import TimeFilmModified,SpringEncoderSTEP,SpringEncoderANA
from .PosEmbedding import PosEmbedding
from .PosEmbeddingMLP import PosEmbeddingMLP
from .PosEmbeddingRNN import PosEmbeddingRNN
from .PosConcatEmbedding import PosConcatEmbedding
from .PosEmbeddingCadence import PosEmbeddingCadence
from .tAPE import tAPE
 



class   TimeHandler(nn.Module):
    def __init__(
        self,
        num_bands=2,
        input_size=1,
        embedding_size=64,
        Tmax=1500.0,
        pe_type="tm",
        **kwargs
    ):
        super(TimeHandler, self).__init__() 
        dict_PEs = {
            "spring": SpringEncoderANA,
            "tm": TimeFilmModified,
            "pe": PosEmbedding,
            "pe_cad": PosEmbeddingCadence,
            "mlp": PosEmbeddingMLP,
            "rnn": PosEmbeddingRNN,
            "pe_concat": PosConcatEmbedding,
            "tAPE": tAPE,
        }
        self.embedding_size = embedding_size
        self.time_encoders = nn.ModuleList([dict_PEs[pe_type](
                embedding_size=embedding_size, input_size=input_size, Tmax=Tmax
            ) for _ in range(num_bands)]) 

    def forward(self, x, t, mask, **kwargs):
        bsz,seqlen,channels = x.shape
        x_mod = torch.empty(bsz,seqlen*channels,self.embedding_size,device=x.device, dtype = x.dtype)
        for i in range(x.shape[-1]): 
            x_mod[:,seqlen*i:seqlen*(i+1),:] = self.time_encoders[i](x[:,:,i].unsqueeze(-1), x[:,:,i].unsqueeze(-1),mask[:,:,i].unsqueeze(-1))
            t_mod = t.view(bsz,seqlen*channels,1)
            m_mod = mask.view(bsz,seqlen*channels,1)
        indexes = (t_mod * m_mod + ~(m_mod) * 9999999).argsort(axis=1)
        x_mod = x_mod.gather(1, indexes.repeat(1, 1, x_mod.shape[-1]))
        m_mod  = m_mod.gather(1, indexes)
        t_mod = t_mod.gather(1, indexes)
        return (
            x_mod,
            m_mod,
            t_mod
        )