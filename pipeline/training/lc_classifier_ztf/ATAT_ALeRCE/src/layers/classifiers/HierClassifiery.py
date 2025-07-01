
import torch.nn as nn

import torch.nn


class Hier(nn.Module):
    def __init__(self, embedding_size, num_classes,num_hier = 3,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.fc_hier = nn.Sequential(         nn.Linear(embedding_size, embedding_size*4),
                                        nn.Dropout(0.01),
                                        nn.LayerNorm(embedding_size*4),
                                        nn.GELU(),
                                        nn.Linear(embedding_size*4, embedding_size*4),
                                        nn.Dropout(0.01),
                                        nn.LayerNorm(embedding_size*4),
                                        nn.GELU(),
                                     nn.Linear(embedding_size*4,num_hier))
        self.fc_class =  nn.Sequential(         nn.Linear(embedding_size, embedding_size*4),
                                        nn.Dropout(0.01),
                                        nn.LayerNorm(embedding_size*4),
                                        nn.GELU(),
                                        nn.Linear(embedding_size*4, embedding_size*4),
                                        nn.Dropout(0.01),
                                        nn.LayerNorm(embedding_size*4),
                                        nn.GELU(),
                                     nn.Linear(embedding_size*4,num_classes))
    def forward(self, x, inference = False):
        if inference: 

            pass
        else:
            logits_hier = self.fc_hier(x)
            logits_class = self.fc_class(x)
            return (logits_hier, logits_class)
        


class Hier2(nn.Module):
    def __init__(self, embedding_size, num_classes,num_hier = 3,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.fc_hier = nn.Sequential(         nn.Linear(embedding_size, embedding_size),
                                        nn.Dropout(0.01),
                                        nn.LayerNorm(embedding_size),
                                        nn.GELU(),
                                     nn.Linear(embedding_size,num_hier))
        self.transient =  nn.Sequential(         nn.Linear(embedding_size, embedding_size),
                                        nn.Dropout(0.01),
                                        nn.LayerNorm(embedding_size),
                                        nn.GELU(),
                                     nn.Linear(embedding_size,8))
        self.stoch =  nn.Sequential(         nn.Linear(embedding_size, embedding_size),
                                        nn.Dropout(0.01),
                                        nn.LayerNorm(embedding_size),
                                        nn.GELU(),
                                     nn.Linear(embedding_size,5))
        self.per = nn.Sequential(         nn.Linear(embedding_size, embedding_size),
                                        nn.Dropout(0.01),
                                        nn.LayerNorm(embedding_size),
                                        nn.GELU(),
                                     nn.Linear(embedding_size,9))
    def forward(self, x, inference = False):

            logits_hier = self.fc_hier(x)
            logits_trans = self.transient(x)
            logits_stoch = self.stoch(x)
            logits_per = self.per(x)

            return (logits_hier, logits_trans, logits_stoch, logits_per)
        
