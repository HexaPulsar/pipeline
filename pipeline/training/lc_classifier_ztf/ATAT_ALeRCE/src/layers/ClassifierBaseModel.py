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
        self.init_model()
        

    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, data, time, mask=None):
        emb = self.model(data, time, mask)
        return self.classifier(emb)

class MultimodalClassifier(nn.Module):
    def __init__(self, lightcuve_classifier,tabular_classifier, multimodal_classifier):
        self.lightcuve_classifier= lightcuve_classifier
        self.tabular_classifier = tabular_classifier
        self.multimodal_classifier = multimodal_classifier
    
    def forward(self,lc_emb,tab_emb):
        torch.concat()
        return loss
        pass