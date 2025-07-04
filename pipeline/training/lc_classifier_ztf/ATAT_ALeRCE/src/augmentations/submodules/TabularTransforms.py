
import torch

class TABGaussianNoise:
    def  __init__(self, num_bands,std = 1e-5):
        super().__init__()
        self.num_bands = num_bands
        self.std = std
       
    def __call__(self, sample):
        
        band_data = sample['tabular_feat']
        noise = torch.normal(0,self.std, size=(band_data.size(0),)).to(device=band_data.device, non_blocking=True) 
        sample["tabular_feat"] = band_data + noise
        return sample

