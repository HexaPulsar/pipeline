import torch.nn as nn

import torch.nn


class TokenClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(embedding_size)
        self.output_layer = nn.Sequential(#nn.Linear(embedding_size, embedding_size),
                                          #nn.GELU(),
                                            nn.Linear(embedding_size, num_classes))
    def forward(self, x):
        return self.output_layer(self.norm(x))



class Hier(nn.Module):
    def __init__(self, embedding_size, num_classes,num_hier = 3,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.fc_hier = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                     nn.GELU(),
                                     nn.Linear(embedding_size,num_hier))
        self.fc_class = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                     nn.GELU(),
                                     nn.Linear(embedding_size,num_classes))
    def forward(self, x, inference = False):
        if inference: 

            pass
        else:
            logits_hier = self.fc_hier(x)
            logits_class = self.fc_class(x)
            return (logits_hier, logits_class)
        

class MultimodalClassifier(nn.Module):
    def __init__(self,
                 experiment_type = str,
                 lc_input_size = None,
                 tab_input_size = None,
                 use_lc = False, 
                 use_tab = False,
                 use_mix = False,
                 num_classes = None,
                 dropout=0.01, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.lc_input_size = lc_input_size
        self.tab_input_size = tab_input_size
        self.use_lc = use_lc
        self.use_tab = use_tab
        self.use_mix = use_mix
        parse_exp_type = experiment_type.split('_')
        self.modalities = []
        self.modalities+= ['LC'] if 'LC' in parse_exp_type else []
        self.modalities+= ['TAB'] if 'MD' in parse_exp_type or 'FEAT' in parse_exp_type else []
        self.modalities+= ['MIX'] if ('MD' in parse_exp_type or 'FEAT' in parse_exp_type) and ('LC' in parse_exp_type) else []
        
        if use_lc:
            self.token_lc =  TokenClassifier(lc_input_size,num_classes ) # Hier(lc_input_size, 22, 3) #
        if use_tab:
            self.token_tab = TokenClassifier(tab_input_size,num_classes )
        if use_mix:
            combined = lc_input_size + tab_input_size
            self.net = nn.Sequential(nn.LayerNorm(combined),
                nn.Linear(combined, combined),
                nn.GELU(),
            nn.Linear(combined, num_classes),
            )
            #self.classifier = Hier(combined, 22, 3)

    def forward(self,emb):
       # emb = emb / emb.norm(dim = -1, keepdim = True)
        out_dict= {}
        if self.use_lc:
            lc_class = self.token_lc(emb)
            out_dict.update({'LC':lc_class})

        if self.use_tab:
            tab_class = self.token_tab(emb)
            out_dict.update({'TAB':tab_class})

        if self.use_mix:
            mix_class = self.net(emb)
            out_dict.update({'MIX':mix_class})
        return out_dict
       
    