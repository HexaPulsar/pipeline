import torch.nn as nn

class ProjectorBaseModel(nn.Module):
    def __init__(self, model, projector):
        """_summary_

        Args:
            model (_type_): _description_
            projector (_type_): _description_
        """
        super().__init__()
        self.model = model
        self.projector = projector
        self.init_model()
        
    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, data, time, mask=None):
        emb = self.model(data, time, mask)
        return self.projector(emb)

