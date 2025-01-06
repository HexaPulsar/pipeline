 
import torch
import torch.nn as nn
from ..classifiers import TokenClassifier

from ..embeddings import Embedding 
from ..tokenEmbeddings import Token 
from .projector import Projector

class TabularTransformer(nn.Module):
    def __init__(self, **kwargs):
        print(kwargs)
        super(TabularTransformer, self).__init__()
        self.embedding_ft =   Embedding(**kwargs) #nn.Linear(kwargs['TAB_ARGS']['length_size'],kwargs['TAB_ARGS']['embedding_size']) #
        encoder = nn.TransformerEncoderLayer(d_model = kwargs['embedding_size'],
                                                 nhead=kwargs['num_heads'],
                                                 dim_feedforward=kwargs['embedding_size_sub'],
                                                 activation = 'gelu',
                                                 dropout=0.01,
                                                 batch_first=True,
                                                 norm_first=True)
         
        self.transformer_ft= nn.TransformerEncoder(encoder_layer=encoder,num_layers=kwargs['num_encoders'])
        
        self.token_ft = Token(**kwargs) 
        
    def embedding_feats(self, f):
        f_mod = self.embedding_ft(**{"f": f}) 
        return torch.cat([self.token_ft(f.shape[0]), f_mod], axis=1) 
    
    def forward(self,tabular_feat, **kwargs): 
        f_mod = self.embedding_feats(**{"f": tabular_feat})
        f_emb = self.transformer_ft(**{"src": f_mod, 'src_key_padding_mask':None}) 
        return f_emb


class TabularClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(TabularClassifier, self).__init__()
        self.TAB = TabularTransformer(**kwargs['ft'])
        self.classifier_tab = TokenClassifier(kwargs['ft']['embedding_size'],
                                             num_classes=kwargs['general']['num_classes'],
                                             )  
        self.init_model()
    def init_model(self):
                for p in self.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_normal_(p)
    def forward(self, tabular_feat, **kwargs):
        tab_emb = self.TAB(tabular_feat)
        return self.classifier_tab(tab_emb[:,0,:])
    



class TabularProjector(nn.Module):
    def __init__(self, **kwargs):
        super(TabularProjector, self).__init__()
        self.transformer = TabularTransformer(**kwargs)
        self.project = Projector(128,
                                    48,
                                    48,
                                    l2norm = False) 
        self.init_model()
             
    def init_model(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                #nn.init.normal_(p, 0, 0.02)
                nn.init.xavier_normal_(p)
        for p in self.project.parameters():
            if p.dim() > 1:
                #nn.init.normal_(p, 0, 0.02)
                nn.init.xavier_normal_(p)

    def forward(self,**kwargs):
        emb = self.transformer(**kwargs)
        return self.project(emb)[:,0,:]