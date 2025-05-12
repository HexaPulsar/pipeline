import torch


class DealWithNaNs:
    def __call__(self, sample):
        sample['tabular_feat'] = sample['tabular_feat'].masked_fill_(torch.isnan(sample['tabular_feat']),0)
        return sample
    
class DealWithInfs:
    def __call__(self, sample):
        data = sample['tabular_feat']
        pos_inf_mask = torch.isinf(data) & (data > 0)
        data[pos_inf_mask] = 9999
        
        # Mask negative infinities
        neg_inf_mask = torch.isinf(data) & (data < 0)
        data[neg_inf_mask] = -9999
        sample['tabular_feat'] = data
        return sample


class TABGaussianNoise:     
    def __init__(self, mean,std):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        x = sample['tabular_feat']  # Shape: [bs, seqlen, channels]
        # Generate Gaussian noise for each channel independently
        noise = torch.normal(self.mean, self.std, size=x.shape).to(device = x.device)
        x_with_noise = x + noise
        #x_with_noise = torch.clip(x_with_noise, 0,1.0)
        sample['tabular_feat'] = x_with_noise
        return sample
    

class RandomMask:
    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Input tensor of shape [bs, seqlen, channels].

        Returns:
            torch.Tensor: Tensor with a random channel zeroed for each sample in the batch.
        """
        mask = ( torch.rand_like(sample['tabular_feat'])>=0.5).bool()

        sample['tab_mask'] = mask
        return sample
