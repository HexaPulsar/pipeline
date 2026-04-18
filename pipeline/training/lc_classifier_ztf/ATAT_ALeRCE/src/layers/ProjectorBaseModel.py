import torch.nn as nn

class ProjectorBaseModel(nn.Module):
    def __init__(self, model, projector):
        """Projection model combining an embedding model with a projection head.

        Args:
            model: Embedding model that produces feature representations
            projector: Projection head that maps embeddings to a projection space
        """
        super().__init__()
        self.model = model
        self.projector = projector
         

    def forward(self, data, time, mask=None):
        emb = self.model(data, time, mask)
        return self.projector(emb)

