import torch
import torch.nn as nn


class VICRegProjector(nn.Module):
    def __init__(
        self,vicreg, shape:str = '128-128-128', norm_seqdim=False, norm_embdim = False, **kwargs
    ):
        super(VICRegProjector, self).__init__()
        layers = []
        f = list(map(int, shape.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.LayerNorm(f[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.projection_x = nn.Sequential(*layers)
        self.projection_y = nn.Sequential(*layers)
        self.vicreg = vicreg
    def forward(self, emb_x,emb_y):
        return self.vicreg(self.projection_x(emb_x),self.projection_y(emb_y))


class CLIPProjector(nn.Module):
    def __init__(self, input_size, output_size, l2norm=False, **kwargs):
        super(CLIPProjector, self).__init__()
        self.kwargs = kwargs
        self.l2norm = l2norm
        hidden_size = 128
        self.projection = nn.Sequential(
            # nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, embedding):
        # embedding = embedding / torch.norm(embedding,dim = 1,keepdim=True)
        embedding = self.projection(embedding)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)
        return embedding
