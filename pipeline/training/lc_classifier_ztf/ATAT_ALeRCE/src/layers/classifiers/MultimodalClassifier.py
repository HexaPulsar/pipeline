import torch.nn as nn

import torch.nn
'''
self.output_layer =  nn.Sequential(
                                     nn.Linear(embedding_size,embedding_size, bias=True),
                                    nn.Dropout(dropout),
                                    nn.LayerNorm(embedding_size),
                                    nn.GELU(),
                                    nn.Linear(embedding_size,embedding_size, bias=True),
                                    nn.Dropout(dropout),
                                    nn.LayerNorm(embedding_size),
                                    nn.GELU(),
                                    nn.Linear(embedding_size, num_classes)
                                     )
'''
class TokenClassifier(nn.Module):
    def __init__(self, embedding_size,num_classes,
                  dropout = 0.01, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        self.output_layer =  nn.Sequential(nn.Dropout(dropout),

                                     nn.Linear(embedding_size,num_classes, bias=True),
                                     )
    def forward(self, x):

        return self.output_layer(x)


class MultimodalClassifier(nn.Module):
    def __init__(self,
                 experiment_type = str,
                 lc_input_size = None,
                 tab_input_size = None,
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
        print(use_lc, use_tab, use_mix)
        #if combine_logits:
        if use_lc:
            self.token_lc =  TokenClassifier(lc_input_size,num_classes, dropout ) # Hier(lc_input_size, 22, 3) #
        elif use_tab:
            self.token_tab = TokenClassifier(tab_input_size,num_classes, dropout )
        #elif self.combine_logits:
         #   assert all([self.combine_logits, self.use_mix,self.use_lc, self.use_tab]), 'to combine logits use all modalities'
        else:
           # self.mixed_classifier = TokenClassifier(lc_input_size + tab_input_size,num_classes, dropout)
            self.mixed_classifier = nn.Sequential(nn.Linear(lc_input_size + tab_input_size,lc_input_size + tab_input_size),
                                                  nn.Dropout(dropout),
                                                  nn.LayerNorm(lc_input_size + tab_input_size),
                                                  nn.GELU(),
                                                  nn.Linear(lc_input_size + tab_input_size,num_classes),
                                                )
    def forward(self,emb_dict):
        out_dict= {}

        if isinstance(emb_dict,dict):

            if self.combine_logits:
                lc_class = self.token_lc(emb_dict['LC'])# / self.logit_scale
                #out_dict.update({'LC':lc_class})
                tab_class =  self.token_tab(emb_dict['TAB']) # / self.logit_scale
                #out_dict.update({'TAB':tab_class})

                out_dict.update({'MIX': nn.functional.softmax(lc_class[:,0,:], dim = -1) + nn.functional.softmax(tab_class[:,0,:], dim = -1)})
            else:
                emb = torch.concat([emb_dict['LC'][:,0,:] , emb_dict['TAB'][:,0,:] ], dim = -1)
                out_dict.update({'MIX': self.mixed_classifier(emb)})
        else:
            emb_dict = emb_dict[:,0,:]
            #self.temp.data = torch.clamp(self.temp.data,0,4.605)
        # self.logit_scale.data = torch.clamp(self.logit_scale.data,0,4.605)
            if all([self.use_lc, not self.use_tab, not self.use_mix]):
                lc_class = self.token_lc(emb_dict)# / self.logit_scale
                out_dict.update({'LC':lc_class})

            if all([not self.use_lc, self.use_tab, not self.use_mix]):
                tab_class = self.token_tab(emb_dict)# / self.logit_scale
                out_dict.update({'TAB':tab_class})

            if self.use_mix:
                #mix_class = self.net(emb_dict['MIX']) #/ self.logit_scale

                if all([self.combine_logits,self.use_lc, self.use_tab]):
                    lc_class = self.token_lc(emb_dict['LC']) # / self.logit_scale
                    #out_dict.update({'LC':lc_class})
                    tab_class =  self.token_tab(emb_dict['TAB']) # / self.logit_scale
                    #out_dict.update({'TAB':tab_class})

                    out_dict.update({'MIX': lc_class + tab_class})
                else:
                    out_dict.update({'MIX': self.mixed_classifier(emb_dict['MIX'])})
        return out_dict

