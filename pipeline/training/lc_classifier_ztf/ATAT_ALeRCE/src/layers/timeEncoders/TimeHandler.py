import torch
import torch.nn as nn 
from .TimeFilmModified import TimeFilmModified, TimeFilmModifiedMOD
from .PosEmbedding import PosEmbedding
from .PosEmbeddingMLP import PosEmbeddingMLP
from .PosEmbeddingRNN import PosEmbeddingRNN
from .PosConcatEmbedding import PosConcatEmbedding
from .PosEmbeddingCadence import PosEmbeddingCadence
from .tAPE import tAPE
 
from ..utils import clones


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
        # general params
        self.num_bands = num_bands
        self.ebedding_size = embedding_size
        self.T_max = Tmax

        dict_PEs = {
            "tm": TimeFilmModified,
            "pe": PosEmbedding,
            "pe_cad": PosEmbeddingCadence,
            "mlp": PosEmbeddingMLP,
            "rnn": PosEmbeddingRNN,
            "pe_concat": PosConcatEmbedding,
            "tAPE": tAPE,
        }

        # tume_encoders
        self.time_encoders = clones(
            dict_PEs[pe_type](
                embedding_size=embedding_size, input_size=input_size, Tmax=Tmax
            ),
            num_bands,
        )

    def forward(self, x, t, mask, **kwargs):
        x_mod = []
        t_mod = []
        m_mod = []

        for i in range(x.shape[-1]):
            slices_x = [slice(None)] * (x.dim() - 1) + [slice(i, i + 1)]
            slices_t = [slice(None)] * (t.dim() - 1) + [slice(i, i + 1)]

            if x.dim() != t.dim():
                x_band = self.time_encoders[i](x[slices_x].squeeze(-1), t[slices_t])
            else:
                x_band = self.time_encoders[i](x[slices_x], t[slices_t])

            t_band = t[slices_t]
            m_band = mask[slices_t]

            x_mod.append(x_band)
            t_mod.append(t_band)
            m_mod.append(m_band)

        x_mod = torch.cat(x_mod, axis=1)
        t_mod = torch.cat(t_mod, axis=1)
        m_mod = torch.cat(m_mod, axis=1)

        # sorted indexes along time, trwoh to the end  new samples
        indexes = (t_mod * m_mod + (1 - m_mod) * 9999999).argsort(axis=1)

        return (
            x_mod.gather(1, indexes.repeat(1, 1, x_mod.shape[-1])),
            m_mod.gather(1, indexes),
            t_mod.gather(1, indexes),
        )


class TimeHandlerMOD(nn.Module):
    def __init__(self, num_bands=2, embedding_size=64, Tmax=1000.0, num_harmonics = 4,**kwargs):
        super().__init__()
        # general params
        self.num_bands = num_bands
        self.time_encoders = nn.ModuleList([TimeFilmModifiedMOD(n_harmonics=num_harmonics,embedding_size=embedding_size, Tmax=Tmax) for _ in range(num_bands)])
        self.embedding_size = embedding_size
    
    def bring_zeros(self,tensor):
        indices = (tensor != 0).type(torch.float32)
        sorted_indices = torch.argsort(indices, dim=1, descending=True) 
        return torch.gather(tensor, 1, sorted_indices)
    
    def forward(self, x, t, mask, **kwargs):
        batch_size, seqlen, channels = x.shape          
        x_mod = torch.empty(batch_size,seqlen*channels,self.embedding_size,device = x.device)
        for i in range(self.num_bands):
            x_mod[:,seqlen*i:seqlen*(i+1),:] = self.time_encoders[i](x[:, :,  i].unsqueeze(-1).clone(), t[:, :, i].unsqueeze(-1).clone())
        with torch.no_grad():
            x_mod = self.bring_zeros(x_mod)
            mask = self.bring_zeros(mask.reshape(batch_size,-1,1))
            t = self.bring_zeros(t.reshape(batch_size,-1,1))
            max_seq = (x_mod).count_nonzero(dim =1).max()      
        return x_mod[:,:max_seq,:], mask[:,:max_seq,:],t[:,:max_seq,:]