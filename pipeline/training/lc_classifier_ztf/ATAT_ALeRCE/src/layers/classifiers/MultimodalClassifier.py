import torch.nn as nn

import torch.nn


class TokenClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(embedding_size)
        self.output_layer = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                          nn.GELU(),
                                          nn.Dropout(0.1),
                                        nn.Linear(embedding_size, num_classes),)

    def forward(self, x):
        return self.output_layer(self.norm(x))


class MultimodalClassifier(nn.Module):
    def __init__(self,lc_input_size = None,
                 tab_input_size = None,
                 num_classes = None,
                 dropout=0.01, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.lc_input_size = lc_input_size
        self.tab_input_size = tab_input_size
        if lc_input_size is not None:
            self.token_lc = TokenClassifier(lc_input_size,num_classes )
        if tab_input_size is not None:
            self.token_tab = TokenClassifier(tab_input_size,num_classes )

        if all([lc_input_size is not None, tab_input_size is not None]):
            combined = lc_input_size + tab_input_size
            self.net = nn.Sequential(nn.LayerNorm(combined),
                nn.Linear(combined, combined//4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(combined//4, combined//4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(combined//4, num_classes),
            )

    def forward(self,emb):
        out_dict = {}

        if isinstance(emb,dict):
            if self.lc_input_size is not None:
                lc_class = self.token_lc(emb['LC'])
                out_dict.update({'LC':lc_class})
            if self.tab_input_size is not None:
                tab_class = self.token_tab(emb['TAB'])
                out_dict.update({'TAB':tab_class})
            if all([self.lc_input_size is not None, self.tab_input_size is not None]):
                mix_class = self.net(emb['MIX'])
                out_dict.update({'MIX':mix_class})
        else:
            if self.lc_input_size is not None:
                lc_class = self.token_lc(emb)
                out_dict.update({'LC':lc_class})
            if self.tab_input_size is not None:
                tab_class = self.token_tab(emb)
                out_dict.update({'TAB':tab_class})
            if all([self.lc_input_size is not None, self.tab_input_size is not None]):
                mix_class = self.net(emb)
                out_dict.update({'MIX':mix_class})
        return out_dict
        
