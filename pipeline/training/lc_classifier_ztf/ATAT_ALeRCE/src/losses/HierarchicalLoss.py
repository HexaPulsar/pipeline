import torch
import torch.nn as nn

class HierarchicalLoss(nn.Module):
    def __init__(self,taxonomy: dict,node_coeff:int = 0.5, leaf_coeff = 0.5):
        self.taxonomy = taxonomy
        self.node_classes = taxonomy.keys()
        self.leaf_classes =  list(range(sum([len(taxonomy[key]) for key in list(taxonomy.keys())])))
        self.node_coeff= node_coeff
        self.lead_coeff = leaf_coeff
        self.inv_taxonomy =  {v: k for k, vals in taxonomy.items() for v in vals}
        
    def map_leaf_target_to_node_target(leaf_target):
        return 
        pass
    def forward(self,node_leaf_tuple:tuple,target):
        target = torch.tensor(list(map(lambda x: self.inv_taxonomy[x],target)))
        hierarchical_loss = 0
        
        hierarchical_loss += nn.functional(node_pred, node_target) * self.node_coeff
        hierarchical_loss += nn.functional(leaf_pred, leaf_target) * self.lead_coeff
        return hierarchical_loss