import torch.nn as nn

import torch.nn


class TokenClassifier(nn.Module):
    def __init__(self, embedding_size, 
                 inner_size,num_classes,
                  dropout = 0.01, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        
        self.output_layer =  nn.Sequential(nn.RMSNorm(embedding_size),
                                        nn.Linear(embedding_size, inner_size),
                                        nn.Dropout(dropout),
                                        nn.GELU(),
                                     nn.Linear(inner_size,num_classes),
                                     #nn.Softmax(dim= -1)
                                     )
    def forward(self, x):
        #norm = torch.sqrt(torch.linalg.norm(x, dim = (1), keepdim = True))
        #x = x / norm
        return self.output_layer(x)


class MultimodalClassifier(nn.Module):
    def __init__(self,
                 experiment_type = str,
                 lc_input_size = None,
                 tab_input_size = None,
                 inner_size = 128,
                 use_lc = False, 
                 use_tab = False,
                 use_mix = False,
                 num_classes = None,
                 combine_logits = False,
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
        self.combine_logits = combine_logits
        
        if use_lc:
            self.token_lc =  TokenClassifier(lc_input_size,inner_size,num_classes, dropout ) # Hier(lc_input_size, 22, 3) #
        if use_tab:
            self.token_tab = TokenClassifier(tab_input_size,inner_size,num_classes, dropout )
        if use_mix:
            combined = lc_input_size + tab_input_size
            self.net = nn.Sequential(#nn.LayerNorm(combined),
                                     nn.Linear(combined, inner_size),
                                        nn.Dropout(dropout),
                                        nn.LayerNorm(inner_size),
                                        nn.GELU(),  
                                     nn.Linear(inner_size,num_classes),
                                     #nn.Softmax(dim =-1)
                                     )
        if self.combine_logits:
            assert all([self.combine_logits, self.use_mix,self.use_lc, self.use_tab]), 'to combine logits use all modalities'
    def forward(self,emb_dict):
       
       # self.logit_scale.data = torch.clamp(self.logit_scale.data,0,4.605) 
        out_dict= {}
        if all([self.use_lc, not self.use_tab, not self.use_mix]):
            lc_class = self.token_lc(emb_dict)# / self.logit_scale
            out_dict.update({'LC':lc_class})

        if all([not self.use_lc, self.use_tab, not self.use_mix]):
            tab_class = self.token_tab(emb_dict)# / self.logit_scale
            out_dict.update({'TAB':tab_class})

        if self.use_mix:
            mix_class = self.net(emb_dict['MIX']) #/ self.logit_scale
            
            if all([self.combine_logits,self.use_lc, self.use_tab]):
                lc_class = self.token_lc(emb_dict['LC'])# / self.logit_scale
                #out_dict.update({'LC':lc_class})
                tab_class =  self.token_tab(emb_dict['TAB']) # / self.logit_scale
                #out_dict.update({'TAB':tab_class})

                out_dict.update({'MIX': (lc_class + tab_class + mix_class)/3})
            else:
                out_dict.update({'MIX':mix_class})
        return out_dict
       
    