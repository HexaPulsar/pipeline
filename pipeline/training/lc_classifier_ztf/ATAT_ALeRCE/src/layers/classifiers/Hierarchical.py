import torch
import torch.nn as nn
from src.layers.classifiers.MixedClassifier import MixedClassifier
from src.layers.classifiers.TokenClassifier import TokenClassifier

class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim:int, hierarchy:dict, dropout=0.01,**kwargs):
        super().__init__()
        self.hierarchy = hierarchy
        self.num_node_classes = len(list(hierarchy.keys()))
        self.num_leaf_classes = [len(value) for key,value in hierarchy.items()]
        self.inv_hierarchy =  {v: k for k, vals in hierarchy.items() for v in vals}
        assert all([isinstance(i,int) for i in self.num_leaf_classes])
        #print(self.num_leaf_classes,self.num_node_classes)
        assert self.num_node_classes == len(self.num_leaf_classes)
        
        self.node_net= TokenClassifier(input_dim,self.num_node_classes)
        self.leaf_classifiers = nn.ModuleList([TokenClassifier(input_dim,self.num_leaf_classes[i]) for i in range(len(self.num_leaf_classes))])

    def remap_prediction(self,node,belongs_to_leaf):
        out = torch.zeros_like(belongs_to_leaf)
        for real_node,real_class in self.hierarchy.items():
            out[:,list(real_class.values())] = belongs_to_leaf[:,:len(real_class)]
        return out
    
  
    
    def forward(self, x,target):
        belongs_to_node = self.node_net(x) # should return an int in range num_node_classifiers
        belongs_to_node = torch.argmax(belongs_to_node,dim = 1).to(dtype=torch.int)
        node_pred = torch.zeros((x.size(0),sum(self.num_leaf_classes)),device = x.device,dtype = int)
        leaf_pred = torch.zeros((x.size(0),sum(self.num_leaf_classes)),device = x.device,dtype = float)
        node_pred = torch.argmax(self.node_net(x), dim = -1)
        for i in range(self.num_node_classes):
            indices = torch.where(belongs_to_node == i)
            #print(5*'#')
            #print(indices)
            partial_leaf_pred = self.leaf_classifiers[i](x[indices])
            padded = torch.nn.functional.pad(partial_leaf_pred,(0,sum(self.num_leaf_classes)-partial_leaf_pred.size(-1)))
            leaf_pred[indices] = padded.to(dtype = float)

        remapped_leaf_pred = self.remap_prediction(None,leaf_pred)
        remapped_leaf_pred = torch.argmax(remapped_leaf_pred, dim = -1)
        return (node_pred,remapped_leaf_pred)
