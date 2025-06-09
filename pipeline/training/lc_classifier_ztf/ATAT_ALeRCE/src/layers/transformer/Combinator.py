import torch.nn as nn

import torch

class Combinator(nn.Module):
    def __init__(self, lc_model, tab_model):
        super().__init__()
        self.transformer_lc = lc_model
        self.transformer_tab = tab_model
    
    def forward(self,data,time,mask, tabular_feat = None,metadata_feat = None,extracted_feat = None, **kwargs):
        lc_emb = self.transformer_lc(data,time,mask)
        ft_emb  = self.transformer_tab(tabular_feat)
        return torch.concat([lc_emb,ft_emb],axis  = -1)
        #return {'LC':lc_emb, }#"TAB" :ft_emb, "MIX": torch.concat([lc_emb,ft_emb],axis = -1)}

    def predict(self,data,time,mask, tabular_feat = None,metadata_feat = None,extracted_feat = None, **kwargs):
        lc_emb = self.transformer_lc(data,time,mask)
        ft_emb  = self.transformer_tab(tabular_feat)
        return torch.concat([lc_emb,ft_emb],axis  = -1)
        #return {'LC':lc_emb, }#"TAB" :ft_emb, "MIX": torch.concat([lc_emb,ft_emb],axis = -1)}