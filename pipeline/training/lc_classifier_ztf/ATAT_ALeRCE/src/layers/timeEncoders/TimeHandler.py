import torch
import torch.nn as nn 
from .TimeFilmModified import EarlyFusionEncoder 

class   TimeHandler(nn.Module):
    def __init__(
        self,
        num_bands=2,
        input_size=1,
        embedding_size=64,
        Tmax=1500.0,
        pe_type="tm",
    ):
        super(TimeHandler, self).__init__() 
        dict_PEs = {
            "tm": EarlyFusionEncoder,
        }
        self.embedding_size = embedding_size
        self.time_encoders = nn.ModuleList([dict_PEs[pe_type](
                embedding_size=embedding_size, input_size=input_size, Tmax=Tmax
            ) for _ in range(num_bands)]) 
        
    def forward(self, 
                x, 
                t, 
                mask,
                metadata = None,
                coordinates = None, 
                allwise = None, 
                timespan = None):
        
        x_mod = []
        t_mod = []
        m_mod = []

        for i in range(x.shape[-1]):
            slices_x = [slice(None)] * (x.dim() - 1) + [slice(i, i + 1)]
            slices_t = [slice(None)] * (t.dim() - 1) + [slice(i, i + 1)]
            slices_m = [slice(None)] * (mask.dim() - 1) + [slice(i, i + 1)]

            if x.dim() != t.dim():
                x_band = self.time_encoders[i](x[slices_x], 
                                               t[slices_t], 
                                               metadata = metadata, 
                                               coordinates = coordinates,
                                               timespan = timespan,
                                               allwise = allwise)
            else:
                x_band = self.time_encoders[i](x[slices_x], 
                                               t[slices_t], 
                                               metadata = metadata, 
                                               coordinates = coordinates,
                                               timespan = timespan,
                                               allwise = allwise)
            
            t_band = t[slices_t]
            m_band = mask[slices_m]
            
            x_mod.append(x_band)
            t_mod.append(t_band)
            m_mod.append(m_band)

        x_mod = torch.cat(x_mod, axis=1)
        m_mod = torch.cat(m_mod, axis=1)
        t_mod = torch.cat(t_mod, axis=1)
        
        return (
            x_mod,
            m_mod,
            t_mod
        )