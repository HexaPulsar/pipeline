import torch.nn as nn

class ClassifierBaseModel(nn.Module):
    def __init__(self, model, classifier,loss= nn.CrossEntropyLoss()):
        """_summary_

        Args:
            model (_type_): _description_
            classifier (_type_): _description_
        """
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.loss = loss
        #self.init_model()
        

    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p,std = 0.1)

    def forward(self, data, time, mask=None, labels= None):
        emb = self.model(data, time, mask)
        return self.classifier(emb)
     
    def get_embeddings(self, data, time, mask=None, labels= None):
    # emb = emb / emb.norm(dim = -1, keepdim = True)
        return {"LC":self.model(data, time, mask)}